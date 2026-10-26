import os
import unittest
from pathlib import Path

from stillface.config import PROCESSED_ENV_VAR, RAW_ENV_VAR, DatasetPaths


class TestDatasetPaths(unittest.TestCase):
    def setUp(self) -> None:
        self.paths = DatasetPaths(Path("raw"), Path("out"))

    def test_raw_and_processed_layouts_are_separate(self) -> None:
        self.assertEqual(self.paths.original_dir("1"), Path("raw/Sessions/1/original"))
        self.assertEqual(self.paths.synced_dir("1"), Path("out/Sessions/1/synced"))
        self.assertEqual(self.paths.cut_dir("1"), Path("out/Sessions/1/cut"))
        self.assertEqual(self.paths.normalized_dir("1"), Path("out/Sessions/1/normalized"))
        self.assertEqual(self.paths.visualize_dir("1"), Path("out/Sessions/1/visualize"))
        self.assertEqual(self.paths.thumbnail_path("1"), Path("out/Sessions/1/thumbnail.png"))
        self.assertEqual(self.paths.metadata_path, Path("raw/metadata_database.xlsx"))
        self.assertEqual(self.paths.thumbnails_dir, Path("out/Thumbnails"))
        self.assertEqual(self.paths.logs_dir, Path("out/logs"))

    def test_default_uses_data_links(self) -> None:
        saved = {k: os.environ.pop(k, None) for k in (RAW_ENV_VAR, PROCESSED_ENV_VAR)}
        try:
            paths = DatasetPaths.default()
        finally:
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value
        self.assertEqual(paths.raw_dir, Path("data/DB/ELTE-PPK_StillFace"))
        self.assertEqual(paths.processed_dir, Path("data/DB_PROCESSED/ELTE-PPK_StillFace"))

    def test_default_honors_environment(self) -> None:
        os.environ[RAW_ENV_VAR], os.environ[PROCESSED_ENV_VAR] = "/r", "/p"
        try:
            paths = DatasetPaths.default()
        finally:
            del os.environ[RAW_ENV_VAR], os.environ[PROCESSED_ENV_VAR]
        self.assertEqual((paths.raw_dir, paths.processed_dir), (Path("/r"), Path("/p")))


if __name__ == "__main__":
    unittest.main()
