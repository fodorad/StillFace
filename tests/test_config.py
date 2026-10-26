import os
import tempfile
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


class TestPhaseClips(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.paths = DatasetPaths(self.tmp / "raw", self.tmp / "out")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_phase_clip_points_into_the_raw_processed_folder(self) -> None:
        self.assertEqual(
            self.paths.phase_clip("7", "window", "play"),
            self.tmp / "raw" / "Sessions" / "7" / "processed" / "window_play.mp4",
        )

    def test_phase_clip_rejects_unknown_names(self) -> None:
        with self.assertRaises(ValueError):
            self.paths.phase_clip("7", "ceiling", "play")
        with self.assertRaises(ValueError):
            self.paths.phase_clip("7", "baby", "dinner")

    def test_available_phase_clips_lists_existing_and_ignores_visualize(self) -> None:
        processed = self.paths.raw_dir / "Sessions" / "7" / "processed"
        processed.mkdir(parents=True)
        (processed / "baby_play.mp4").write_bytes(b"")
        (processed / "door_reunion.mp4").write_bytes(b"")
        visualize = self.paths.raw_dir / "Sessions" / "7" / "visualize"
        visualize.mkdir()
        (visualize / "session_play.mp4").write_bytes(b"")

        clips = self.paths.available_phase_clips("7")

        self.assertEqual(set(clips), {("baby", "play"), ("door", "reunion")})
        self.assertEqual(clips[("baby", "play")], processed / "baby_play.mp4")

    def test_available_phase_clips_of_unknown_session_is_empty(self) -> None:
        self.assertEqual(self.paths.available_phase_clips("404"), {})

    def test_feature_folders_are_below_the_processed_root(self) -> None:
        gaze = self.paths.gaze_dir("7")
        self.assertEqual(gaze, self.tmp / "out" / "Sessions" / "7" / "features" / "gaze")
        self.assertEqual(self.paths.gaze_overlay_dir("7"), gaze / "overlay")
        self.assertEqual(self.paths.features_dir("7", "pose").name, "pose")
        for folder in (gaze, self.paths.gaze_overlay_dir("7")):
            self.assertTrue(folder.is_relative_to(self.paths.processed_dir))


if __name__ == "__main__":
    unittest.main()
