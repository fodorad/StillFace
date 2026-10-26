import tempfile
import unittest
from pathlib import Path

from avst.io import get_video_fps

from stillface.data_preparation.video import (
    convert_to_fps,
    run_ffmpeg,
    to_seconds,
    video_duration,
)
from tests.helpers import make_video


class TestVideo(unittest.TestCase):
    def test_to_seconds(self) -> None:
        self.assertEqual(to_seconds("1:30"), 90)
        self.assertEqual(to_seconds("1:01:30"), 3690)
        for bad in ("90", "a:b", "nan"):
            with self.assertRaises(ValueError):
                to_seconds(bad)

    def test_duration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = make_video(Path(tmp) / "v.mp4", seconds=2)
            self.assertAlmostEqual(video_duration(video), 2.0, delta=0.2)
            self.assertEqual(video_duration(Path(tmp) / "missing.mp4"), 0.0)

    def test_convert_to_fps_changes_rate_and_keeps_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = make_video(Path(tmp) / "src.mp4", fps=25)
            dst = convert_to_fps(src, Path(tmp) / "out" / "dst.mp4", 60)
            self.assertAlmostEqual(get_video_fps(dst), 60, delta=0.1)
            mtime = dst.stat().st_mtime_ns
            self.assertEqual(convert_to_fps(src, dst, 60), dst)
            self.assertEqual(dst.stat().st_mtime_ns, mtime)

    def test_convert_failure_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                convert_to_fps(Path(tmp) / "none.mp4", Path(tmp) / "o.mp4", 60)

    def test_run_ffmpeg_reports_failure(self) -> None:
        self.assertFalse(run_ffmpeg(["-i", "does_not_exist.mp4", "out.mp4"]))


if __name__ == "__main__":
    unittest.main()
