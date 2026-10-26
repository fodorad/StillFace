import json
import tempfile
import unittest
from pathlib import Path

import cv2

from stillface.config import DatasetPaths
from stillface.features.gaze.config import GazeConfig
from stillface.features.gaze.pilot import head_overlay_items, head_rows, run_heads
from stillface.features.gaze.prepare import ClipHeads
from stillface.features.gaze.roles import RoleAssignment
from stillface.features.timeline import read_timeline
from tests.features.gaze.fakes import FakeHeadTracker, box, track
from tests.helpers import make_video

SIZE = "160x120"


class PilotCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.paths = DatasetPaths(self.tmp / "raw", self.tmp / "out")
        self.processed = self.paths.raw_dir / "Sessions" / "7" / "processed"
        self.out = self.paths.features_dir("7", "heads")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def clip(self, camera: str, phase: str = "play") -> Path:
        return make_video(
            self.processed / f"{camera}_{phase}.mp4", seconds=1, fps=60, size=SIZE, audio=False
        )


class TestRows(unittest.TestCase):
    def prepared(self) -> ClipHeads:
        return ClipHeads(
            [0, 2], 60.0, RoleAssignment(2, 1),
            {"mother": [box(0.8), None], "baby": [box(0.3), box(0.3)]},
        )  # fmt: skip

    def test_missing_heads_are_not_codable_with_a_reason(self) -> None:
        rows = head_rows("7", "door", "play", self.prepared())
        by_key = {(r.frame_index, r.person): r for r in rows}
        self.assertEqual(len(rows), 4)
        self.assertEqual(by_key[(0, "mother")].gaze_class, "")
        missing = by_key[(2, "mother")]
        self.assertEqual((missing.gaze_class, missing.reason), ("NK", "occluded"))
        self.assertAlmostEqual(by_key[(2, "baby")].time_s, 2 / 60)
        self.assertIsNone(missing.gaze_x)

    def test_close_camera_only_has_its_owner(self) -> None:
        rows = head_rows("7", "baby", "play", self.prepared())
        self.assertEqual({r.person for r in rows}, {"baby"})

    def test_overlay_items(self) -> None:
        items = head_overlay_items("window", self.prepared())
        self.assertEqual(sorted(items), [0, 2])
        self.assertEqual([i.label for i in items[2]], ["NK:no_head", ""])


class TestRunHeads(PilotCase):
    def tracker(self) -> FakeHeadTracker:
        frames = list(range(0, 60, 2))
        both = [track(1, 0.3, frames=frames), track(2, 0.8, frames=frames)]
        return FakeHeadTracker({"window_play.mp4": both, "baby_play.mp4": [track(5, 0.5)]})

    def test_writes_timelines_overlays_grid_and_run_record(self) -> None:
        self.clip("window"), self.clip("baby")
        record = run_heads(self.paths, "7", self.tracker(), GazeConfig())

        rows = read_timeline(self.out / "window_play.csv")
        self.assertEqual({r.person for r in rows}, {"mother", "baby"})
        self.assertEqual(len(rows), 60)
        self.assertTrue((self.out / "overlay" / "window_play.mp4").exists())
        self.assertTrue((self.out / "overlay" / "session_play.mp4").exists())
        saved = json.loads((self.out / "run.json").read_text())
        self.assertEqual(saved["model_variant"], "none")
        self.assertEqual(sorted(record.clips), ["baby_play.mp4", "window_play.mp4"])
        self.assertEqual(saved["config"]["sample_rate_hz"], 30.0)

    def test_camera_and_phase_filters(self) -> None:
        self.clip("window"), self.clip("window", "reunion"), self.clip("baby")
        tracker = self.tracker()
        run_heads(self.paths, "7", tracker, GazeConfig(), cameras=["window"], phases=["play"],
                  overlay=False)  # fmt: skip
        self.assertEqual([c.name for c, _ in tracker.calls], ["window_play.mp4"])
        self.assertFalse((self.out / "overlay").exists())

    def test_finished_outputs_are_skipped_unless_overwrite(self) -> None:
        self.clip("window")
        tracker = self.tracker()
        run_heads(self.paths, "7", tracker, GazeConfig())
        run_heads(self.paths, "7", tracker, GazeConfig())
        self.assertEqual(len(tracker.calls), 1)
        run_heads(self.paths, "7", tracker, GazeConfig(), overwrite=True)
        self.assertEqual(len(tracker.calls), 2)

    def test_ambiguous_roles_are_flagged_in_the_run_record(self) -> None:
        self.clip("window")
        tracker = FakeHeadTracker({"window_play.mp4": [track(1, 0.5), track(2, 0.51)]})
        record = run_heads(self.paths, "7", tracker, GazeConfig(), overlay=False)
        self.assertEqual(record.flags["roles_ambiguous"], ["window_play"])

    def test_roles_json_in_the_output_folder_is_used(self) -> None:
        self.clip("window")
        self.out.mkdir(parents=True)
        (self.out / "roles.json").write_text(
            json.dumps({"window": {"play": {"mother": 1, "baby": 2}}})
        )
        tracker = FakeHeadTracker({"window_play.mp4": [track(1, 0.5), track(2, 0.51)]})
        record = run_heads(self.paths, "7", tracker, GazeConfig(), overlay=False)
        self.assertNotIn("roles_ambiguous", record.flags)

    def test_failing_clip_is_recorded_and_the_rest_continues(self) -> None:
        self.clip("window")
        (self.processed / "baby_play.mp4").write_bytes(b"not a video")
        record = run_heads(self.paths, "7", self.tracker(), GazeConfig(), overlay=False)
        self.assertEqual(len(record.flags["failed"]), 1)
        self.assertIn("baby_play", record.flags["failed"][0])
        self.assertTrue((self.out / "window_play.csv").exists())
        self.assertFalse((self.out / "baby_play.csv").exists())

    def test_overlay_clip_has_the_source_frame_count(self) -> None:
        self.clip("window")
        run_heads(self.paths, "7", self.tracker(), GazeConfig())
        cap = cv2.VideoCapture(str(self.out / "overlay" / "window_play.mp4"))
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 60)
        cap.release()


if __name__ == "__main__":
    unittest.main()
