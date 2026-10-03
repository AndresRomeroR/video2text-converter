import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import video2text


class NoConsoleTests(unittest.TestCase):
    def test_existing_console_streams_are_preserved(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(sys, "stdout", stdout), patch.object(sys, "stderr", stderr):
            video2text.ensure_standard_streams()
            self.assertIs(sys.stdout, stdout)
            self.assertIs(sys.stderr, stderr)

    def test_missing_streams_accept_output_and_initialization_is_idempotent(self):
        with patch.object(sys, "stdout", None), patch.object(sys, "stderr", None):
            video2text.ensure_standard_streams()
            stdout, stderr = sys.stdout, sys.stderr
            try:
                print("Progreso")
                stderr.write("Descargando modelo\n")
                stderr.flush()
                video2text.ensure_standard_streams()
                self.assertIs(sys.stdout, stdout)
                self.assertIs(sys.stderr, stderr)
            finally:
                stdout.close()
                stderr.close()

    def test_whisper_download_with_progress_without_console(self):
        # Ejecuta el descargador real de Whisper con una respuesta local pequena.
        # Se conserva tqdm, la escritura del archivo y la verificacion SHA256.
        whisper = video2text.load_whisper()
        payload = b"model fixture" * 1024
        checksum = hashlib.sha256(payload).hexdigest()
        url = f"https://example.invalid/{checksum}/fixture.pt"
        response = io.BytesIO(payload)
        response.info = lambda: {"Content-Length": str(len(payload))}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys, "stdout", None), patch.object(sys, "stderr", None):
                video2text.ensure_standard_streams()
                stdout, stderr = sys.stdout, sys.stderr
                try:
                    with patch.object(whisper.urllib.request, "urlopen", return_value=response):
                        downloaded = whisper._download(url, directory, in_memory=False)
                    self.assertEqual(Path(downloaded).read_bytes(), payload)
                finally:
                    stdout.close()
                    stderr.close()


if __name__ == "__main__":
    unittest.main()
