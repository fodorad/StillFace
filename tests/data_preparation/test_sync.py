import tempfile
import unittest
from pathlib import Path

import pandas as pd
from avst.io import get_video_fps

from stillface.config import DatasetPaths
from stillface.data_preparation.sync import (
    find_cameras,
    normalize,
    pick_reference,
    sync,
    sync_all,
    sync_auxiliary,
    sync_mother_baby,
)
from tests.helpers import make_noise_video, make_video


class TestSync(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.paths = DatasetPaths(self.tmp / "raw", self.tmp / "out")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_find_cameras_prefers_mts_and_skips_absent(self) -> None:
        original = self.paths.original_dir("1")
        make_video(original / "baby.mp4", seconds=1)
        make_video(original / "window.mp4", seconds=1)
        make_video(original / "window.MTS", seconds=1, fps=25)
        cameras = find_cameras(original)
        self.assertEqual(set(cameras), {"baby", "window"})
        self.assertEqual(cameras["window"].suffix, ".MTS")
        self.assertEqual(find_cameras(self.tmp / "none"), {})

    def test_normalize_converts_rate_but_remuxes_matching_rate(self) -> None:
        original = self.paths.original_dir("1")
        slow = make_video(original / "window.MTS", seconds=1, fps=25)
        fast = make_video(original / "door.mp4", seconds=1, fps=60)
        out = normalize({"window": slow, "door": fast}, self.paths.normalized_dir("1"))
        self.assertAlmostEqual(get_video_fps(out["window"]), 60, delta=0.1)
        self.assertAlmostEqual(get_video_fps(out["door"]), 60, delta=0.1)
        self.assertTrue(fast.exists(), "originals are untouched")
        self.assertEqual(out["door"].parent, self.paths.normalized_dir("1"))

    def test_pick_reference_priority(self) -> None:
        cams = {"window": Path("w_src.mp4")}
        self.assertEqual(pick_reference({"mother": Path("m"), "baby": Path("b")}, cams), Path("b"))
        self.assertEqual(pick_reference({"mother": Path("m")}, cams), Path("m"))
        self.assertEqual(pick_reference({"window": Path("w")}, cams), Path("w"))
        self.assertEqual(pick_reference({}, cams), Path("w_src.mp4"))
        self.assertIsNone(pick_reference({}, {}))

    def test_single_camera_is_copied(self) -> None:
        baby = make_video(self.tmp / "baby.mp4", seconds=1)
        result = sync_mother_baby(None, baby, self.tmp / "synced")
        self.assertEqual(set(result.videos), {"baby"})
        self.assertIsNone(result.offset_ms)
        self.assertTrue(result.videos["baby"].exists())
        mother = make_video(self.tmp / "mother.mp4", seconds=1)
        self.assertEqual(set(sync_mother_baby(mother, None, self.tmp / "s2").videos), {"mother"})
        self.assertEqual(sync_mother_baby(None, None, self.tmp / "s3").videos, {})

    def test_mother_baby_offset_is_recovered(self) -> None:
        mother = make_noise_video(self.tmp / "mother.mp4", delay_ms=0)
        baby = make_noise_video(self.tmp / "baby.mp4", delay_ms=1000)
        result = sync_mother_baby(mother, baby, self.tmp / "synced")
        self.assertAlmostEqual(abs(result.offset_ms), 1000, delta=50)
        self.assertEqual(set(result.videos), {"mother", "baby"})
        again = sync_mother_baby(mother, baby, self.tmp / "synced")
        self.assertIsNone(again.offset_ms, "existing outputs are not recomputed")

    def test_auxiliary_without_reference_is_copied(self) -> None:
        window = make_video(self.tmp / "window.mp4", seconds=1)
        out = sync_auxiliary("window", window, None, self.tmp / "synced")
        self.assertEqual(out, self.tmp / "synced" / "window.mp4")
        self.assertTrue(out.exists())

    def test_auxiliary_is_aligned_to_reference(self) -> None:
        reference = make_noise_video(self.tmp / "ref.mp4", delay_ms=0)
        door = make_noise_video(self.tmp / "door.mp4", delay_ms=500)
        out = sync_auxiliary("door", door, reference, self.tmp / "synced")
        self.assertTrue(out.exists())

    def test_session_without_cameras_is_empty(self) -> None:
        self.assertEqual(sync(self.paths, "404").videos, {})

    def test_full_session_never_touches_originals(self) -> None:
        original = self.paths.original_dir("5")
        make_noise_video(original / "mother.mp4", delay_ms=0)
        make_noise_video(original / "baby.mp4", delay_ms=1000)
        make_noise_video(original / "window.MTS", delay_ms=500, fps=60)
        before = sorted(p.name for p in original.iterdir())
        result = sync(self.paths, "5")
        self.assertEqual(set(result.videos), {"mother", "baby", "window"})
        self.assertEqual(sorted(p.name for p in original.iterdir()), before)

    def test_sync_all_logs_success_and_failure(self) -> None:
        make_noise_video(self.paths.original_dir("1") / "mother.mp4")
        make_noise_video(self.paths.original_dir("1") / "baby.mp4", delay_ms=500)
        broken = self.paths.original_dir("2")
        broken.mkdir(parents=True)
        (broken / "window.MTS").write_bytes(b"not a video")
        self.paths.raw_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "ID": [1, 2, 3],
                "Auto": ["y", "y", "n"],
                "offset_mother-baby_(ms)": [None, None, None],
            }
        ).to_excel(self.paths.metadata_path, index=False)

        sync_all(self.paths)

        synced = (self.paths.logs_dir / "synced_sessions.txt").read_text().splitlines()
        self.assertEqual([line.split(",")[0] for line in synced], ["1"])
        self.assertEqual((self.paths.logs_dir / "failed_sessions.txt").read_text().split(), ["2"])


if __name__ == "__main__":
    unittest.main()
