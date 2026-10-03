import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import video2text


@unittest.skipUnless(
    os.environ.get("VIDEO2TEXT_RUN_WHISPER_INTEGRATION") == "1",
    "Prueba de modelo desactivada; define VIDEO2TEXT_RUN_WHISPER_INTEGRATION=1.",
)
class WhisperIntegrationTests(unittest.TestCase):
    def test_transcribes_real_ogg_with_installed_model(self):
        whisper_module = video2text.load_whisper()
        torch_module = video2text.load_torch()
        ffmpeg_path = video2text.require_ffmpeg()

        with tempfile.TemporaryDirectory() as directory:
            audio_file = Path(directory) / "tono.ogg"
            subprocess.run(
                [
                    ffmpeg_path,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:duration=0.5",
                    "-c:a",
                    "libvorbis",
                    "-y",
                    str(audio_file),
                ],
                check=True,
                timeout=30,
            )

            video2text.probe_audio_stream(audio_file)
            requested_device = os.environ.get("VIDEO2TEXT_TEST_DEVICE", "auto")
            device, fp16 = video2text.resolve_device(torch_module, requested_device, True)
            if requested_device != "auto":
                expected = "cuda" if requested_device == "rocm" else requested_device
                self.assertEqual(device, expected, "El backend solicitado no esta disponible.")
                if requested_device == "rocm":
                    self.assertTrue(torch_module.version.hip, "Esta instalacion no usa ROCm.")
            model_name = os.environ.get("VIDEO2TEXT_TEST_MODEL", "turbo")
            model = whisper_module.load_model(model_name, device=device)
            with torch_module.inference_mode():
                result = video2text.transcribe_media(
                    model,
                    audio_file,
                    "es",
                    fp16,
                    None,
                )

        self.assertIn("text", result)
        self.assertIn("segments", result)
        self.assertEqual(result.get("language"), "es")


if __name__ == "__main__":
    unittest.main()
