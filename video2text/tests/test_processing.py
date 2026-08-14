import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import video2text


class ProcessingTests(unittest.TestCase):
    def test_bundled_resource_uses_pyinstaller_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(video2text.sys, "_MEIPASS", directory, create=True):
                resource = video2text.bundled_resource("totext.ico")

        self.assertEqual(resource, Path(directory) / "totext.ico")

    def test_find_media_binary_prefers_packaged_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            binary_name = "ffmpeg.exe" if video2text.sys.platform == "win32" else "ffmpeg"
            binary_path = Path(directory) / binary_name
            binary_path.touch()
            with patch.object(video2text.sys, "_MEIPASS", directory, create=True):
                resolved = video2text.find_media_binary("ffmpeg")

        self.assertEqual(resolved, str(binary_path))

    def test_resolve_media_file_accepts_audio_and_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            filenames = (
                "grabacion.mp3",
                "voz.OGG",
                "audio.wav",
                "reunion.m4a",
                "video.mp4",
            )
            for filename in filenames:
                media_file = root / filename
                media_file.touch()
                self.assertEqual(video2text.resolve_media_file(media_file), media_file)

    def test_resolve_media_file_rejects_unsupported_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            media_file = Path(directory) / "notas.txt"
            media_file.touch()

            with self.assertRaisesRegex(ValueError, r"\.mp3"):
                video2text.resolve_media_file(media_file)

    def test_legacy_video_resolver_accepts_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            audio_file = Path(directory) / "grabacion.flac"
            audio_file.touch()

            self.assertEqual(video2text.resolve_video_file(audio_file), audio_file)

    def test_probe_audio_stream_returns_first_audio_track(self):
        completed = SimpleNamespace(
            returncode=0,
            stdout='{"streams":[{"codec_name":"vorbis","channels":2,"sample_rate":"48000"}]}',
            stderr="",
        )
        with patch("video2text.subprocess.run", return_value=completed) as run:
            stream = video2text.probe_audio_stream(Path("grabacion.ogg"), "ffprobe")

        self.assertEqual(stream["codec_name"], "vorbis")
        self.assertIn("grabacion.ogg", run.call_args.args[0])

    def test_probe_audio_stream_rejects_media_without_audio(self):
        completed = SimpleNamespace(returncode=0, stdout='{"streams":[]}', stderr="")
        with patch("video2text.subprocess.run", return_value=completed):
            with self.assertRaisesRegex(ValueError, "pista de audio"):
                video2text.probe_audio_stream(Path("video.mp4"), "ffprobe")

    def test_probe_audio_stream_reports_execution_failure(self):
        with patch("video2text.subprocess.run", side_effect=OSError("fallo del sistema")):
            with self.assertRaisesRegex(RuntimeError, "No se pudo ejecutar FFprobe"):
                video2text.probe_audio_stream(Path("audio.ogg"), "ffprobe")

    def test_normalize_language_accepts_spanish_names(self):
        whisper_module = SimpleNamespace(
            tokenizer=SimpleNamespace(TO_LANGUAGE_CODE={}, LANGUAGES={"es": "spanish"})
        )

        self.assertEqual(video2text.normalize_language(whisper_module, "Español"), "es")
        self.assertEqual(video2text.normalize_language(whisper_module, "castellano"), "es")

    def test_transcribe_media_passes_current_whisper_arguments(self):
        model = Mock()
        model.transcribe.return_value = {"text": "contenido", "segments": []}
        media_file = Path("grabacion.ogg")

        result = video2text.transcribe_media(model, media_file, "es", True, "Contexto")

        self.assertEqual(result["text"], "contenido")
        model.transcribe.assert_called_once_with(
            str(media_file),
            language="es",
            fp16=True,
            initial_prompt="Contexto",
            word_timestamps=False,
            verbose=None,
        )

    def test_srt_timestamp_clamps_negative_values(self):
        self.assertEqual(video2text.srt_timestamp(-1), "00:00:00,000")
        self.assertEqual(video2text.srt_timestamp(3661.234), "01:01:01,234")

    def test_build_outputs_filters_invalid_segments(self):
        result = {
            "text": " Hola mundo ",
            "segments": [
                {"start": -1, "end": 1.25, "text": " Hola "},
                {"start": "incorrecto", "end": 2, "text": "descartar"},
                {"start": 2, "end": 1, "text": "mundo"},
                {"start": 3, "end": 4, "text": "  "},
            ],
        }

        txt, srt = video2text.build_transcript_outputs(result)

        self.assertEqual(txt, "Hola mundo\n")
        self.assertIn("00:00:00,000 --> 00:00:01,250", srt)
        self.assertIn("00:00:02,000 --> 00:00:02,000", srt)
        self.assertNotIn("descartar", srt)

    def test_transcript_falls_back_to_segment_text(self):
        txt, _ = video2text.build_transcript_outputs(
            {"segments": [{"start": 0, "end": 1, "text": "Texto recuperado"}]}
        )
        self.assertEqual(txt, "Texto recuperado\n")

    def test_write_windows_text_is_atomic_and_uses_crlf(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "salida.txt"
            video2text.write_windows_text(output, "línea 1\nlínea 2\n")

            self.assertEqual(output.read_bytes(), "línea 1\r\nlínea 2\r\n".encode())
            self.assertFalse((output.parent / ".salida.txt.tmp").exists())

    def test_save_transcript_outputs_uses_audio_basename(self):
        with tempfile.TemporaryDirectory() as directory:
            media_file = Path(directory) / "entrevista.ogg"
            txt_file, srt_file = video2text.save_transcript_outputs(
                media_file,
                {
                    "text": "Texto",
                    "segments": [{"start": 0, "end": 1, "text": "Texto"}],
                },
            )

            self.assertEqual(txt_file.name, "entrevista.txt")
            self.assertEqual(srt_file.name, "entrevista.srt")
            self.assertTrue(txt_file.exists())
            self.assertTrue(srt_file.exists())


if __name__ == "__main__":
    unittest.main()
