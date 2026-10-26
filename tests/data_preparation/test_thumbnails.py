import tempfile
import unittest
from pathlib import Path

import cv2

from stillface.config import DatasetPaths
from stillface.data_preparation.thumbnails import generate_thumbnails, middle_frame
from tests.helpers import make_video


class TestThumbnails(unittest.TestCase):
    def test_middle_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = make_video(Path(tmp) / "v.mp4", seconds=1, size="64x48")
            self.assertEqual(middle_frame(video).shape, (48, 64, 3))
            self.assertIsNone(middle_frame(Path(tmp) / "none.mp4"))

    def test_generate_for_all_sessions_skips_broken_ones(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = DatasetPaths(Path(tmp) / "raw", Path(tmp) / "out")
            make_video(paths.visualize_dir("1") / "session_stillface.mp4", seconds=1)
            paths.visualize_dir("2").mkdir(parents=True)  # no overview video
            self.assertEqual(generate_thumbnails(paths), 1)
            for png in (paths.thumbnail_path("1"), paths.thumbnails_dir / "1.png"):
                self.assertIsNotNone(cv2.imread(str(png)))
            self.assertEqual(generate_thumbnails(paths, ["1"]), 1)

    def test_no_sessions_yet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(generate_thumbnails(DatasetPaths(Path(tmp), Path(tmp) / "out")), 0)


if __name__ == "__main__":
    unittest.main()
