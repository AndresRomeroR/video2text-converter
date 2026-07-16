import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import video2text


class ProcessingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
