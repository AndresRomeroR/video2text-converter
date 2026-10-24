import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transcription_queue import BatchQueue


class BatchQueueTests(unittest.TestCase):
    def test_mixed_audio_and_video_are_processed_in_order(self):
        batch = BatchQueue()
        added, duplicates = batch.add(
            [Path("entrevista.mp3"), Path("video.mp4"), Path("reunion.wav")]
        )
        events = []

        def process(item):
            self.assertEqual(batch.summary()["processing"], 1)
            events.append(("process", item.path.name))
            return item.path.with_suffix(".txt"), item.path.with_suffix(".srt")

        batch.run(process, lambda item: events.append((item.status, item.path.name)))

        self.assertEqual(duplicates, 0)
        self.assertEqual(
            events,
            [(state, name) for name in ("entrevista.mp3", "video.mp4", "reunion.wav")
             for state in ("processing", "process", "completed")],
        )
        self.assertEqual(batch.summary(), {
            "total": 3, "pending": 0, "processing": 0, "completed": 3, "error": 0,
        })
        self.assertEqual(added[0].outputs[0].name, "entrevista.txt")
        self.assertTrue(all(item.elapsed >= 0 for item in added))

    def test_failed_file_does_not_block_following_files(self):
        batch = BatchQueue()
        added, _ = batch.add([Path("roto.mp4"), Path("correcto.wav")])
        summaries = []

        def process(item):
            if item.path.name == "roto.mp4":
                raise ValueError("El archivo no tiene pista de audio")
            return item.path.with_suffix(".txt"), item.path.with_suffix(".srt")

        batch.run(process, lambda _: summaries.append(batch.summary()))

        self.assertEqual(added[0].error, "El archivo no tiene pista de audio")
        self.assertIsNone(added[0].outputs)
        self.assertEqual(added[1].status, "completed")
        self.assertEqual(summaries[2], {
            "total": 2, "pending": 0, "processing": 1, "completed": 0, "error": 1,
        })
        self.assertEqual(batch.summary()["error"], 1)

    def test_duplicates_are_ignored_within_and_between_selections(self):
        batch = BatchQueue()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            added, duplicates = batch.add([
                root / "video.mp4", root / "." / "video.mp4", root / "audio.wav",
            ])
            self.assertEqual([item.path.name for item in added], ["video.mp4", "audio.wav"])
            self.assertEqual(duplicates, 1)
            again, duplicates = batch.add([root / "video.mp4"])
            self.assertEqual(again, [])
            self.assertEqual(duplicates, 1)
            self.assertEqual(len({item.id for item in batch.items}), 2)

    def test_next_run_only_processes_new_pending_items(self):
        batch = BatchQueue()
        batch.add([Path("primero.mp3")])
        process = Mock(side_effect=lambda item: (
            item.path.with_suffix(".txt"), item.path.with_suffix(".srt"),
        ))
        batch.run(process, Mock())
        batch.add([Path("segundo.mp4")])
        batch.run(process, Mock())
        batch.run(process, Mock())

        self.assertEqual([call.args[0].path.name for call in process.call_args_list],
                         ["primero.mp3", "segundo.mp4"])
        self.assertEqual(batch.summary()["completed"], 2)

    def test_stop_between_files_preserves_pending_items(self):
        batch = BatchQueue()
        added, _ = batch.add([Path("primero.mp3"), Path("segundo.mp3")])
        process = Mock(return_value=(Path("salida.txt"), Path("salida.srt")))

        batch.run(process, Mock(), lambda: batch.summary()["completed"] == 1)

        process.assert_called_once_with(added[0])
        self.assertEqual([item.status for item in added], ["completed", "pending"])

    def test_remove_and_clear_finished_preserve_active_and_pending_items(self):
        batch = BatchQueue()
        added, _ = batch.add([Path(name) for name in ("a.wav", "b.mp4", "c.mp3", "d.ogg")])
        for item, status in zip(added, ("processing", "pending", "completed", "error")):
            item.status = status

        removed = batch.clear_finished()
        self.assertEqual([item.id for item in removed], [added[2].id, added[3].id])
        removed = batch.remove([item.id for item in added])
        self.assertEqual([item.id for item in removed], [added[1].id])
        self.assertEqual(batch.items, [added[0]])


if __name__ == "__main__":
    unittest.main()
