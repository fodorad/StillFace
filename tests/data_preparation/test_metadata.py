import tempfile
import unittest
from pathlib import Path

import pandas as pd

from stillface.constants import PHASES
from stillface.data_preparation.metadata import (
    load_metadata,
    missing_recordings,
    phase_windows,
    sessions_to_cut,
    sessions_to_sync,
)


def make_table() -> pd.DataFrame:
    row = {
        "ID": 1, "Auto": "y", "Session_date_(YYYY-HH-DD)": "2024-03-13",
        "Session_hour_(HH:MM)": "13:45", "mother": "y", "baby": "y", "window": "n",
        "door": "y", "polar_mother": "n", "polar_baby": "y", "offset_mother-baby_(ms)": None,
    }  # fmt: skip
    for i, phase in enumerate(PHASES):
        row[f"{phase}_start_(MM:SS)"] = f"{i}:00"
        row[f"{phase}_end_(MM:SS)"] = f"{i}:30"
    done = {**row, "ID": 2, "offset_mother-baby_(ms)": 120.0}
    manual = {**row, "ID": 3, "Auto": "n"}
    return pd.DataFrame([row, done, manual])


class TestMetadata(unittest.TestCase):
    def test_load_csv_and_xlsx_roundtrip(self) -> None:
        table = make_table()
        with tempfile.TemporaryDirectory() as tmp:
            csv, xlsx = Path(tmp) / "m.csv", Path(tmp) / "m.xlsx"
            table.to_csv(csv, index=False)
            table.to_excel(xlsx, index=False)
            self.assertEqual(len(load_metadata(csv)), 3)
            self.assertEqual(len(load_metadata(xlsx)), 3)

    def test_load_rejects_missing_and_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_metadata(Path(tmp) / "none.csv")
            other = Path(tmp) / "m.json"
            other.write_text("{}")
            with self.assertRaises(ValueError):
                load_metadata(other)

    def test_missing_recordings_lists_flagged_only(self) -> None:
        report = missing_recordings(make_table())
        self.assertEqual(
            set(report.loc[report["ID"] == 1, "missing"]), {"camera {window}", "HRV {polar_mother}"}
        )

    def test_session_selection(self) -> None:
        table = make_table()
        self.assertEqual(sessions_to_sync(table), ["1"])
        self.assertEqual(sessions_to_cut(table), ["1", "2"])

    def test_phase_windows(self) -> None:
        windows = phase_windows(make_table(), "1").windows
        self.assertEqual(windows["play"], ("1:00", "1:30"))
        self.assertEqual(list(windows), list(PHASES))
        with self.assertRaises(KeyError):
            phase_windows(make_table(), "99")


if __name__ == "__main__":
    unittest.main()
