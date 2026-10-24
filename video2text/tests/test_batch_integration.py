"""Real Whisper batch validation without creating a Tk window or downloading models."""

import os
import subprocess
import sys
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import video2text
from transcription_queue import BatchQueue


@unittest.skipUnless(
    os.environ.get("VIDEO2TEXT_RUN_WHISPER_INTEGRATION") == "1",
    "Prueba de lote real desactivada; define VIDEO2TEXT_RUN_WHISPER_INTEGRATION=1.",
)
class BatchIntegrationTests(unittest.TestCase):
    def test_real_mixed_batch_reuses_model_and_continues_after_invalid_audio(self):
        cache_root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        cached_model = cache_root / "whisper" / "tiny.pt"
        if not cached_model.is_file():
            self.skipTest("tiny.pt no está en la caché local; esta prueba no descarga modelos.")

        whisper = video2text.load_whisper()
        torch = video2text.load_torch()
        ffmpeg = video2text.require_ffmpeg()
        video2text.require_ffprobe()
        app = video2text.Video2TextApp.__new__(video2text.Video2TextApp)
        app._batch = BatchQueue()
        app._loaded_model = None
        app._loaded_model_key = None
        app._closing = False
        app._stop_requested = threading.Event()
        app._set_status = Mock()
        app._append_console = Mock()
        app._finish_batch = Mock()
        app._ui = lambda callback: callback()
        events = []
        app._on_item_changed = lambda item: events.append((item.path.name, item.status))
        load_model = whisper.load_model

        def load_local_model(name, *, device):
            self.assertEqual(name, "tiny")
            # Loading a filename bypasses Whisper's downloader completely.
            return load_model(str(cached_model), device=device)

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(4, previous_threads))
        try:
            with tempfile.TemporaryDirectory(prefix="video2text-batch-") as directory:
                root = Path(directory)
                audio_file = root / "grabacion.wav"
                video_file = root / "grabacion.mp4"
                broken_file = root / "roto.wav"
                with wave.open(str(audio_file), "wb") as audio:
                    audio.setnchannels(1)
                    audio.setsampwidth(2)
                    audio.setframerate(16000)
                    audio.writeframes(b"\x00\x00" * 8000)
                broken_file.write_bytes(b"invalid audio data")
                subprocess.run(
                    [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                     "color=c=black:s=32x32:r=10:d=0.5", "-i", str(audio_file),
                     "-c:v", "mpeg4", "-q:v", "10", "-c:a", "aac", "-shortest", "-y", str(video_file)],
                    check=True, capture_output=True, timeout=30,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                )
                items, _ = app._batch.add([audio_file, broken_file, video_file])
                with patch.object(whisper, "load_model", side_effect=load_local_model) as model_load:
                    app._worker_process("tiny", "es", "cpu", False, None)

                self.assertEqual(app._batch.summary(), {
                    "total": 3, "pending": 0, "processing": 0, "completed": 2, "error": 1,
                }, [call.args[0] for call in app._append_console.call_args_list])
                self.assertEqual(events, [
                    ("grabacion.wav", "processing"), ("grabacion.wav", "completed"),
                    ("roto.wav", "processing"), ("roto.wav", "error"),
                    ("grabacion.mp4", "processing"), ("grabacion.mp4", "completed"),
                ])
                model_load.assert_called_once_with("tiny", device="cpu")
                self.assertIsNotNone(app._loaded_model)
                self.assertEqual(app._loaded_model_key, ("tiny", "cpu"))
                self.assertTrue(items[1].error)
                self.assertIsNone(items[1].outputs)
                outputs = [path for item in (items[0], items[2]) for path in item.outputs]
                self.assertEqual(len(set(outputs)), 4)
                self.assertEqual({path.name for path in outputs}, {
                    "grabacion.wav.txt", "grabacion.wav.srt", "grabacion.mp4.txt", "grabacion.mp4.srt",
                })
                self.assertTrue(all(path.is_file() for path in outputs))
                app._finish_batch.assert_called_once()
                self.assertEqual(app._finish_batch.call_args.args[1], "")
        finally:
            app._loaded_model = None
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
