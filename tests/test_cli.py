import tempfile
import unittest
from pathlib import Path

import pandas as pd

from stillface.cli import build_parser, main
from stillface.constants import PHASES
from tests.helpers import make_video


def write_metadata(path: Path) -> None:
    row = {
        "ID": 9, "Auto": "y", "Session_date_(YYYY-HH-DD)": "2024-01-01",
        "Session_hour_(HH:MM)": "10:00", "mother": "y", "baby": "y", "window": "n",
        "door": "y", "polar_mother": "y", "polar_baby": "y", "offset_mother-baby_(ms)": 0.0,
    }  # fmt: skip
    for i, phase in enumerate(PHASES):
        row[f"{phase}_start_(MM:SS)"], row[f"{phase}_end_(MM:SS)"] = f"0:0{i}", f"0:0{i + 1}"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_excel(path, index=False)


class TestCli(unittest.TestCase):
    def test_subcommand_is_required(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args([])

    def test_check_missing_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw, out = Path(tmp) / "raw", Path(tmp) / "out"
            write_metadata(raw / "metadata_database.xlsx")
            code = main(
                ["--raw-dir", str(raw), "--processed-dir", str(out), "prepare", "check-missing"]
            )
            self.assertEqual(code, 0)
            report = pd.read_csv(out / "missing_files_report.csv")
            self.assertEqual(report["missing"].tolist(), ["camera {window}"])

    def test_cut_single_session_then_thumbnails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw, out = Path(tmp) / "raw", Path(tmp) / "out"
            write_metadata(raw / "metadata_database.xlsx")
            make_video(out / "Sessions" / "9" / "synced" / "baby.mp4", seconds=5)
            base = ["--raw-dir", str(raw), "--processed-dir", str(out), "prepare"]
            self.assertEqual(main([*base, "cut", "--session-id", "9", "--visualize"]), 0)
            self.assertTrue((out / "Sessions" / "9" / "cut" / "baby_play.mp4").exists())
            self.assertEqual(main([*base, "thumbnails", "--session-id", "9"]), 0)
            self.assertTrue((out / "Thumbnails" / "9.png").exists())

    def test_cut_and_sync_all_use_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw, out = Path(tmp) / "raw", Path(tmp) / "out"
            write_metadata(raw / "metadata_database.xlsx")
            make_video(out / "Sessions" / "9" / "synced" / "baby.mp4", seconds=5)
            base = ["--raw-dir", str(raw), "--processed-dir", str(out), "-v", "prepare"]
            self.assertEqual(main([*base, "cut"]), 0)
            self.assertTrue((out / "logs" / "cut_sessions.txt").exists())
            self.assertEqual(main([*base, "sync"]), 0)  # offset already known -> nothing pending

    def test_sync_single_session_without_cameras(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = [
                "--raw-dir",
                tmp,
                "--processed-dir",
                tmp,
                "prepare",
                "sync",
                "--session-id",
                "1",
            ]
            self.assertEqual(main(args), 0)


if __name__ == "__main__":
    unittest.main()
