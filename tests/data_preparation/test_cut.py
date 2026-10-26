import tempfile
import unittest
from pathlib import Path

import pandas as pd

from stillface.config import DatasetPaths
from stillface.constants import PHASES
from stillface.data_preparation.cut import cut, cut_all, cut_all_phases, cut_clip
from stillface.data_preparation.metadata import PhaseWindows
from stillface.data_preparation.video import video_duration
from tests.helpers import make_video

WINDOWS = PhaseWindows(
    {
        "baseline": ("0:00", "0:01"),
        "play": ("0:01", "0:02"),
        "stillface": ("0:02", "0:03"),
        "reunion": ("0:03", "0:04"),
    }
)


class TestCut(unittest.TestCase):
    def test_cut_clip_duration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = make_video(Path(tmp) / "s.mp4", seconds=4)
            dst = Path(tmp) / "o" / "c.mp4"
            self.assertTrue(cut_clip(src, dst, "0:01", "0:03"))
            self.assertAlmostEqual(video_duration(dst), 2.0, delta=0.3)
            self.assertTrue(cut_clip(src, dst, "0:01", "0:03"), "existing clip is kept")

    def test_cut_clip_rejects_bad_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = make_video(Path(tmp) / "s.mp4", seconds=2)
            self.assertFalse(cut_clip(Path(tmp) / "none.mp4", Path(tmp) / "a.mp4", "0:00", "0:01"))
            self.assertFalse(cut_clip(src, Path(tmp) / "b.mp4", "nan", "nan"))
            self.assertFalse(cut_clip(src, Path(tmp) / "c.mp4", "0:30", "0:40"))

    def test_cut_all_phases_skips_absent_cameras_and_phases_past_the_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            synced, out = Path(tmp) / "synced", Path(tmp) / "cut"
            make_video(synced / "baby.mp4", seconds=2.5)
            clips = cut_all_phases(synced, WINDOWS, out)
            self.assertEqual(list(clips), ["baby"])
            self.assertEqual(set(clips["baby"]), {"baseline", "play", "stillface"})

    def test_cut_session_with_visualization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = DatasetPaths(Path(tmp) / "raw", Path(tmp) / "out")
            make_video(paths.synced_dir("7") / "mother.mp4", seconds=4.5)
            make_video(paths.synced_dir("7") / "baby.mp4", seconds=4.5)
            clips = cut(paths, "7", WINDOWS, visualize=True)
            self.assertEqual(set(clips), {"mother", "baby"})
            self.assertTrue((paths.visualize_dir("7") / "mother-baby_play.mp4").exists())
            self.assertTrue((paths.visualize_dir("7") / "session_reunion.mp4").exists())

    def test_cut_all_records_failed_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = DatasetPaths(Path(tmp) / "raw", Path(tmp) / "out")
            paths.raw_dir.mkdir(parents=True)
            row = {"ID": [4], "Auto": ["y"]}
            for phase in PHASES:
                row[f"{phase}_start_(MM:SS)"], row[f"{phase}_end_(MM:SS)"] = ["0:00"], ["0:01"]
            pd.DataFrame(row).to_excel(paths.metadata_path, index=False)
            make_video(paths.synced_dir("4") / "baby.mp4", seconds=2)
            paths.visualize_dir("4").parent.mkdir(parents=True, exist_ok=True)
            paths.visualize_dir("4").write_text("blocks the output folder")

            cut_all(paths, visualize=True)

            self.assertEqual(
                (paths.logs_dir / "failed_cut_sessions.txt").read_text().split(), ["4"]
            )
            self.assertFalse((paths.logs_dir / "cut_sessions.txt").exists())


if __name__ == "__main__":
    unittest.main()
