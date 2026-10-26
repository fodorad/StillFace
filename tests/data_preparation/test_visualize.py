import tempfile
import unittest
from pathlib import Path

from avst.io import get_video_frame_size

from stillface.data_preparation.visualize import (
    create_stacked_videos,
    find_cut_videos,
    stack_grid,
    stack_vertical,
)
from tests.helpers import make_video


class TestVisualize(unittest.TestCase):
    def test_stack_vertical_doubles_height_and_keeps_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a, b = (make_video(Path(tmp) / f"{n}.mp4", seconds=1) for n in "ab")
            dst = Path(tmp) / "o" / "s.mp4"
            self.assertTrue(stack_vertical(a, b, dst))
            self.assertEqual(get_video_frame_size(dst), (64, 96))
            self.assertTrue(stack_vertical(a, b, dst))
            self.assertFalse(stack_vertical(a, Path(tmp) / "none.mp4", Path(tmp) / "x.mp4"))

    def test_grid_replaces_missing_cameras_with_black(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            baby = make_video(Path(tmp) / "baby.mp4", seconds=1)
            dst = Path(tmp) / "grid.mp4"
            self.assertTrue(stack_grid({"baby": baby, "mother": None}, dst, audio_source="door"))
            self.assertEqual(get_video_frame_size(dst), (128, 96))
            self.assertTrue(stack_grid({"baby": baby}, dst))

    def test_grid_without_videos_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(stack_grid({"baby": None}, Path(tmp) / "g.mp4"))

    def test_find_and_stack_all_phases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cut_dir = Path(tmp) / "cut"
            for camera in ("mother", "baby"):
                make_video(cut_dir / f"{camera}_play.mp4", seconds=1)
            clips = find_cut_videos(cut_dir)
            self.assertEqual(set(clips["baby"]), {"play"})
            self.assertEqual(clips["window"], {})
            create_stacked_videos(clips, Path(tmp) / "vis")
            self.assertTrue((Path(tmp) / "vis" / "mother-baby_play.mp4").exists())
            self.assertTrue((Path(tmp) / "vis" / "session_play.mp4").exists())
            self.assertFalse((Path(tmp) / "vis" / "mother-baby_reunion.mp4").exists())


if __name__ == "__main__":
    unittest.main()
