import tempfile
import unittest
from pathlib import Path

from stillface.features.run_record import (
    RunRecord,
    current_git_sha,
    installed_version,
    read_run_record,
    write_run_record,
)


class TestRunRecord(unittest.TestCase):
    def test_roundtrip(self) -> None:
        record = RunRecord(
            session_id="7",
            model_variant="vitl14_inout_childplay",
            config={"version": 1, "sample_rate_hz": 30.0},
            exordium_version="1.2.3",
            stillface_version="0.2.0",
            git_sha="abc123",
            clips=["window_play.mp4"],
            flags={"roles_ambiguous": ["door/baseline"]},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x" / "run.json"
            write_run_record(record, path)
            self.assertEqual(read_run_record(path), record)

    def test_optional_fields_default_to_empty(self) -> None:
        record = RunRecord(session_id="7", model_variant="none", config={})
        self.assertIsNone(record.exordium_version)
        self.assertEqual(record.clips, [])
        self.assertEqual(record.flags, {})

    def test_git_sha_is_a_hash_inside_a_checkout_and_none_outside(self) -> None:
        sha = current_git_sha(Path(__file__).parent)
        self.assertIsNotNone(sha)
        self.assertRegex(sha, r"^[0-9a-f]{40}$")
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(current_git_sha(Path(tmp)))

    def test_installed_version(self) -> None:
        self.assertRegex(installed_version("pandas"), r"^\d+\.")
        self.assertIsNone(installed_version("definitely-not-installed-package"))


if __name__ == "__main__":
    unittest.main()
