import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import video2text


def fake_torch(cuda=False, xpu=False, mps=False, hip=None):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda, get_device_name=lambda _: "GPU"),
        xpu=SimpleNamespace(is_available=lambda: xpu, get_device_name=lambda _: "Arc"),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        version=SimpleNamespace(hip=hip),
    )


class DeviceTests(unittest.TestCase):
    def test_auto_on_each_backend_and_cpu(self):
        for kwargs, expected in (
            ({}, ("cpu", False)),
            ({"cuda": True}, ("cuda", True)),
            ({"cuda": True, "hip": "7"}, ("cuda", True)),
            ({"xpu": True}, ("xpu", False)),
            ({"mps": True}, ("mps", False)),
        ):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(video2text.resolve_device(fake_torch(**kwargs), "auto", True), expected)

    def test_rocm_alias_uses_pytorch_cuda_api(self):
        torch = fake_torch(cuda=True, hip="7")
        self.assertEqual(video2text.resolve_device(torch, "rocm", False), ("cuda", False))
        self.assertIn("AMD / ROCm", video2text.device_description(torch, "cuda"))

    def test_explicit_cpu_overrides_available_gpu(self):
        self.assertEqual(video2text.resolve_device(fake_torch(cuda=True), "cpu", True), ("cpu", False))

    def test_unavailable_devices_use_cpu(self):
        for device in ("cuda", "rocm", "xpu", "mps"):
            with self.subTest(device=device):
                self.assertEqual(video2text.resolve_device(fake_torch(), device, True), ("cpu", False))

    def test_broken_driver_and_missing_backend_do_not_block_cpu(self):
        torch = SimpleNamespace(cuda=Mock())
        torch.cuda.is_available.side_effect = RuntimeError("driver unavailable")
        self.assertEqual(video2text.available_devices(torch), ["cpu"])

    def test_gpu_error_releases_resources_before_cpu_retry(self):
        events = []
        def action(device, fp16):
            events.append((device, fp16))
            if device != "cpu":
                raise NotImplementedError("sparse operation not supported")
            return {"text": "listo"}
        result, device = video2text.run_with_cpu_fallback(
            action, "mps", False, lambda: events.append("release"), Mock()
        )
        self.assertEqual(result, {"text": "listo"})
        self.assertEqual(device, "cpu")
        self.assertEqual(events, [("mps", False), "release", ("cpu", False)])

    def test_gpu_out_of_memory_retries_once_and_disables_fp16(self):
        action = Mock(side_effect=[RuntimeError("out of memory"), "ok"])
        release = Mock()
        self.assertEqual(video2text.run_with_cpu_fallback(action, "cuda", True, release, Mock()), ("ok", "cpu"))
        self.assertEqual(action.call_args_list, [call("cuda", True), call("cpu", False)])
        release.assert_called_once_with()

    def test_successful_gpu_does_not_retry(self):
        action, release = Mock(return_value="ok"), Mock()
        self.assertEqual(video2text.run_with_cpu_fallback(action, "xpu", False, release, Mock()), ("ok", "xpu"))
        action.assert_called_once_with("xpu", False)
        release.assert_not_called()

    def test_cpu_failure_is_not_retried(self):
        action = Mock(side_effect=RuntimeError("CPU failure"))
        with self.assertRaisesRegex(RuntimeError, "CPU failure"):
            video2text.run_with_cpu_fallback(action, "cpu", False, Mock(), Mock())
        action.assert_called_once()

    def test_second_failure_and_unrelated_errors_are_not_hidden(self):
        action = Mock(side_effect=[RuntimeError("GPU"), RuntimeError("CPU")])
        with self.assertRaisesRegex(RuntimeError, "CPU"):
            video2text.run_with_cpu_fallback(action, "cuda", True, Mock(), Mock())
        self.assertEqual(action.call_count, 2)
        action = Mock(side_effect=FileNotFoundError("audio"))
        with self.assertRaises(FileNotFoundError):
            video2text.run_with_cpu_fallback(action, "cuda", True, Mock(), Mock())
        action.assert_called_once()

    def test_folder_opener_for_linux_and_macos_preserves_spaces(self):
        path = Path("folder with spaces")
        for platform, opener in (("linux", "xdg-open"), ("darwin", "open")):
            with self.subTest(platform=platform), patch.object(video2text.sys, "platform", platform), patch("video2text.subprocess.Popen") as popen:
                video2text.open_folder(path)
                popen.assert_called_once_with([opener, str(path)])


if __name__ == "__main__":
    unittest.main()
