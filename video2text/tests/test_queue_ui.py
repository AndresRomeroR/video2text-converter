import sys
import tempfile
import threading
import tkinter as tk
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import video2text
from transcription_queue import BatchQueue


class QueueUiTests(unittest.TestCase):
    def make_app(self):
        # Build only the controller state: no native Tk window or model loading.
        app = video2text.Video2TextApp.__new__(video2text.Video2TextApp)
        app._running = False
        app._batch = BatchQueue()
        app._loaded_model = None
        app._loaded_model_key = None
        return app

    def test_drop_parser_preserves_all_paths_spaces_and_accents(self):
        master = tk.Tcl()
        data = "{C:/Reuniones con espacios/entrevista.mp3} {C:/Vídeos/demo.mp4} C:/voz.wav"

        self.assertEqual(video2text._parse_drop_files(master, data), [
            "C:/Reuniones con espacios/entrevista.mp3", "C:/Vídeos/demo.mp4", "C:/voz.wav",
        ])
        self.assertEqual(video2text._parse_drop_file(master, data),
                         "C:/Reuniones con espacios/entrevista.mp3")
        self.assertEqual(video2text._parse_drop_files(master, ""), [])
        self.assertEqual(video2text._parse_drop_files(master, "{malformed"), [])

    def test_file_picker_adds_the_entire_mixed_selection(self):
        app = self.make_app()
        selected = ("C:/Audios/uno.mp3", "C:/Videos/dos.mp4", "C:/Audios/tres.wav")
        received = []
        app._add_files = lambda paths: received.extend(paths)

        with patch.object(video2text.filedialog, "askopenfilenames", return_value=selected) as picker:
            app._on_select_file()

        picker.assert_called_once()
        self.assertEqual(received, [Path(path) for path in selected])

    def test_drop_adds_the_entire_mixed_selection(self):
        app = self.make_app()
        app.tk = tk.Tcl().tk
        received = []
        app._add_files = lambda paths: received.extend(paths)

        app._on_drop_file(SimpleNamespace(data="{C:/Mis audios/uno.mp3} C:/dos.mp4"))

        self.assertEqual(received, [Path("C:/Mis audios/uno.mp3"), Path("C:/dos.mp4")])

    def test_files_cannot_be_added_while_batch_is_running(self):
        app = self.make_app()
        app._running = True
        app._add_files = Mock()

        with patch.object(video2text.filedialog, "askopenfilenames") as picker:
            drop_result = app._on_drop_file(SimpleNamespace(data="C:/uno.mp3 C:/dos.mp4"))
            app._on_select_file()

        self.assertEqual(drop_result, "refuse_drop")
        app._add_files.assert_not_called()
        picker.assert_not_called()

    def test_add_files_keeps_valid_entries_and_reports_invalid_once(self):
        app = self.make_app()
        app._render_item = Mock()
        app.tabs = Mock()
        app.file_tree = Mock()
        app._set_status = Mock()
        app._append_console = Mock()
        app._refresh_summary = Mock()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio, video, text = (root / name for name in ("audio.mp3", "video.mp4", "notas.txt"))
            for path in (audio, video, text):
                path.touch()
            with patch.object(video2text.messagebox, "showwarning") as warning:
                app._add_files([audio, text, video, audio])

        self.assertEqual([item.path.name for item in app._batch.items], ["audio.mp3", "video.mp4"])
        self.assertEqual(app._render_item.call_count, 2)
        self.assertIn("1 duplicados omitidos", app._set_status.call_args.args[0])
        warning.assert_called_once()
        self.assertIn("1 archivos", warning.call_args.args[1])

    def test_overall_progress_counts_success_and_failure_as_finished(self):
        app = self.make_app()
        added, _ = app._batch.add(Path(name) for name in ("uno.mp3", "dos.mp4", "tres.wav", "cuatro.ogg"))
        for item, status in zip(added, ("completed", "error", "processing", "pending")):
            item.status = status
        app._running = True
        for name in ("lbl_summary", "pbar", "btn_process", "btn_retry", "btn_clear_finished"):
            setattr(app, name, Mock())
        app._update_selection_controls = Mock()

        app._refresh_summary()

        app.pbar.config.assert_called_once_with(maximum=4, value=2)
        summary = app.lbl_summary.config.call_args.kwargs["text"]
        for label in ("4 archivos", "1 pendientes", "1 en proceso", "1 completados", "1 errores"):
            self.assertIn(label, summary)
        self.assertEqual(app.btn_process.config.call_args.kwargs["state"], "disabled")
        self.assertEqual(app.btn_retry.config.call_args.kwargs["state"], "disabled")

    def test_worker_callback_freezes_row_and_counts_before_tk_receives_it(self):
        app = self.make_app()
        item = app._batch.add([Path("grabacion.wav")])[0][0]
        item.status = "processing"
        callbacks = []
        app._ui = callbacks.append
        app._render_item = Mock()
        app._refresh_summary = Mock()
        app.file_tree = Mock()
        app.file_tree.selection.return_value = ()

        app._on_item_changed(item)
        item.status = "completed"
        item.elapsed = 12.3
        item.outputs = (Path("grabacion.txt"), Path("grabacion.srt"))
        app._on_item_changed(item)
        callbacks[0]()

        snapshot = app._render_item.call_args.args[0]
        self.assertEqual(snapshot.status, "processing")
        self.assertEqual(snapshot.elapsed, 0.0)
        self.assertIsNone(snapshot.outputs)
        self.assertEqual(app._refresh_summary.call_args.args[0]["processing"], 1)
        self.assertEqual(app._refresh_summary.call_args.args[0]["completed"], 0)
        callbacks[1]()
        self.assertEqual(app._render_item.call_args.args[0].status, "completed")
        self.assertEqual(app._refresh_summary.call_args.args[0]["completed"], 1)

    def test_same_stem_audio_and_video_can_keep_distinct_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_outputs = video2text.save_transcript_outputs(
                root / "reunion.final.mp3", {"text": "Audio", "segments": []}, include_extension=True,
            )
            video_outputs = video2text.save_transcript_outputs(
                root / "reunion.final.mp4", {"text": "Video", "segments": []}, include_extension=True,
            )

            self.assertEqual([path.name for path in audio_outputs],
                             ["reunion.final.mp3.txt", "reunion.final.mp3.srt"])
            self.assertEqual([path.name for path in video_outputs],
                             ["reunion.final.mp4.txt", "reunion.final.mp4.srt"])
            self.assertEqual(audio_outputs[0].read_text(encoding="utf-8"), "Audio\n")
            self.assertEqual(video_outputs[0].read_text(encoding="utf-8"), "Video\n")

    def test_worker_prepares_once_and_continues_after_a_failed_file(self):
        app = self.make_app()
        items, _ = app._batch.add([Path("roto.mp4"), Path("correcto.wav")])
        app._closing = False
        app._stop_requested = threading.Event()
        app._set_status = Mock()
        app._append_console = Mock()
        app._resolve_device = Mock(return_value=("cpu", False))
        outputs = (Path("correcto.txt"), Path("correcto.srt"))
        app._transcribe_item = Mock(side_effect=[RuntimeError("Sin audio"), outputs])
        app._on_item_changed = Mock()
        callbacks = []
        app._ui = callbacks.append
        app._finish_batch = Mock()
        whisper = object()

        with patch.object(video2text, "load_whisper", return_value=whisper) as load, \
                patch.object(video2text, "require_ffmpeg") as ffmpeg, \
                patch.object(video2text, "require_ffprobe") as ffprobe, \
                patch.object(video2text, "normalize_language", return_value="es"):
            app._worker_process("tiny", "español", "auto", True, "Contexto")

        load.assert_called_once()
        ffmpeg.assert_called_once()
        ffprobe.assert_called_once()
        self.assertEqual([call.args[0] for call in app._transcribe_item.call_args_list], items)
        self.assertEqual([item.status for item in items], ["error", "completed"])
        self.assertEqual(items[1].outputs, outputs)
        for call in app._transcribe_item.call_args_list:
            self.assertEqual(call.args[1:], (whisper, "tiny", "es", "cpu", False, "Contexto"))
        app._finish_batch.assert_not_called()
        self.assertEqual(len(callbacks), 1)
        callbacks[0]()
        self.assertEqual(app._finish_batch.call_args.args[1], "")

    def test_setup_failure_preserves_pending_files_and_finishes_on_ui_thread(self):
        app = self.make_app()
        app._batch.add([Path("uno.mp3"), Path("dos.mp4")])
        app._set_status = Mock()
        app._append_console = Mock()
        app._transcribe_item = Mock()
        app._finish_batch = Mock()
        callbacks = []
        app._ui = callbacks.append

        with patch.object(video2text, "load_whisper", side_effect=RuntimeError("Whisper no disponible")):
            app._worker_process("tiny", "es", "cpu", False, None)

        self.assertEqual(app._batch.summary()["pending"], 2)
        app._transcribe_item.assert_not_called()
        app._finish_batch.assert_not_called()
        callbacks[0]()
        self.assertEqual(app._finish_batch.call_args.args[1], "Whisper no disponible")

    def test_cpu_fallback_is_reused_by_remaining_files_in_the_batch(self):
        app = self.make_app()
        app._batch.add([Path("uno.mp3"), Path("dos.mp4"), Path("tres.wav")])
        app._closing = False
        app._stop_requested = threading.Event()
        app._set_status = Mock()
        app._append_console = Mock()
        app._resolve_device = Mock(return_value=("cuda", True))
        app._on_item_changed = Mock()
        app._ui = Mock()
        devices = []

        def transcribe(item, whisper, model, language, device, fp16, prompt):
            devices.append((device, fp16))
            # The first item's GPU inference recovered by loading the CPU model.
            app._loaded_model_key = (model, "cpu")
            return item.path.with_suffix(".txt"), item.path.with_suffix(".srt")

        app._transcribe_item = transcribe
        with patch.object(video2text, "load_whisper"), \
                patch.object(video2text, "require_ffmpeg"), \
                patch.object(video2text, "require_ffprobe"), \
                patch.object(video2text, "normalize_language", return_value="es"):
            app._worker_process("tiny", "es", "auto", True, None)

        self.assertEqual(devices, [("cuda", True), ("cpu", False), ("cpu", False)])
        self.assertEqual(app._batch.summary()["completed"], 3)


if __name__ == "__main__":
    unittest.main()
