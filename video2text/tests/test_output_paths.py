import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from output_paths import plan_output_paths
from transcription_queue import QueueItem


class OutputPathTests(unittest.TestCase):
    def make_items(self, directory, *names):
        return [QueueItem(str(index), Path(directory) / name)
                for index, name in enumerate(names)]

    def test_ordinary_files_keep_their_stems_and_folders(self):
        with tempfile.TemporaryDirectory() as directory:
            items = self.make_items(directory, "entrevista.mp3", "reunion.mp4")
            planned = plan_output_paths(items)
            self.assertEqual(planned["0"], (Path(directory) / "entrevista.txt", Path(directory) / "entrevista.srt"))
            self.assertEqual(planned["1"][0].name, "reunion.txt")
            self.assertIsNone(items[0].outputs)

    def test_matching_stems_retain_the_input_extensions(self):
        with tempfile.TemporaryDirectory() as directory:
            items = self.make_items(directory, "clip.mp3", "clip.wav")
            planned = plan_output_paths(items)
            self.assertEqual([planned[item.id][0].name for item in items],
                             ["clip.mp3.txt", "clip.wav.txt"])
            self.assertEqual(planned["0"][1].name, "clip.mp3.srt")

    def test_collisions_propagate_through_extension_like_stems(self):
        with tempfile.TemporaryDirectory() as directory:
            items = self.make_items(directory, "clip.mp3", "clip.wav", "clip.mp3.mp4",
                                    "clip.mp3.mp4.wav")
            planned = plan_output_paths(items)
            self.assertEqual([planned[item.id][0].name for item in items],
                             ["clip.mp3.txt", "clip.wav.txt", "clip.mp3.mp4.txt", "clip.mp3.mp4.wav.txt"])
            self.assertEqual(len({path for pair in planned.values() for path in pair}), 8)

    def test_same_stem_in_different_folders_has_no_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            items = self.make_items(directory, "one/clip.mp3", "two/clip.wav")
            planned = plan_output_paths(items)
            self.assertEqual([planned[item.id][0].name for item in items], ["clip.txt", "clip.txt"])

    def test_completed_destinations_are_preserved_between_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            items = self.make_items(directory, "clip.mp3", "clip.wav", "clip.wav.mp4")
            completed = items[0]
            completed.status = "completed"
            completed.outputs = (Path(directory) / "clip.txt", Path(directory) / "clip.srt")
            planned = plan_output_paths(items)
            self.assertEqual(planned[completed.id], completed.outputs)
            self.assertEqual(planned["1"][0].name, "clip.wav.txt")
            self.assertEqual(planned["2"][0].name, "clip.wav.mp4.txt")

    def test_numeric_fallback_avoids_completed_and_other_preferred_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            items = self.make_items(directory, "old.wav", "older.wav", "clip.mp3", "clip.mp3 (2).mp4")
            items[0].status = "completed"
            items[0].outputs = (Path(directory) / "clip.txt", Path(directory) / "clip.srt")
            items[1].status = "completed"
            items[1].outputs = (Path(directory) / "clip.mp3.txt", Path(directory) / "clip.mp3.srt")
            planned = plan_output_paths(items)
            self.assertEqual(planned["2"][0].name, "clip.mp3 (3).txt")
            self.assertEqual(planned["3"][0].name, "clip.mp3 (2).txt")
            self.assertEqual(planned, plan_output_paths(items))


if __name__ == "__main__":
    unittest.main()
