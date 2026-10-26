import tempfile
import unittest
from pathlib import Path

from stillface.config import DatasetPaths
from stillface.features.gaze.config import GazeConfig
from stillface.features.gaze.pipeline import _decide, read_frames, run_gaze
from stillface.features.gaze.prepare import ClipHeads
from stillface.features.gaze.roles import RoleAssignment
from stillface.features.timeline import read_timeline
from tests.features.gaze.fakes import FakeGazeEstimator, FakeHeadTracker, box, track
from tests.helpers import make_video

FRAMES = list(range(0, 60, 2))
CLOSE_CFG = GazeConfig(partner_zones={"baby": (0.0, 0.0, 0.4, 0.4), "mother": (0.6, 0.0, 1.0, 0.4)})


class PipelineCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.paths = DatasetPaths(self.tmp / "raw", self.tmp / "out")
        self.out = self.paths.gaze_dir("7")
        self.processed = self.paths.raw_dir / "Sessions" / "7" / "processed"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def clip(self, camera: str, phase: str = "play") -> Path:
        return make_video(
            self.processed / f"{camera}_{phase}.mp4", seconds=1, fps=60, size="160x120", audio=False
        )

    def window_tracker(self, mother_frames=FRAMES) -> FakeHeadTracker:
        return FakeHeadTracker(
            {"window_play.mp4": [track(1, 0.3, frames=FRAMES), track(2, 0.8, frames=mother_frames)]}
        )


class TestSideCamera(PipelineCase):
    def test_mutual_gaze_is_classified_per_person_and_frame(self) -> None:
        self.clip("window")
        # person order is (mother, baby): mother looks at the baby head, baby at the mother head
        estimator = FakeGazeEstimator({0: (0.3, 0.4), 1: (0.8, 0.4)})

        run_gaze(self.paths, "7", self.window_tracker(), estimator, GazeConfig(), overlay=False)

        rows = read_timeline(self.out / "window_play.csv")
        self.assertEqual(len(rows), 60)
        by_person = {p: [r for r in rows if r.person == p] for p in ("mother", "baby")}
        self.assertEqual({r.gaze_class for r in by_person["mother"]}, {"BABARA"})
        self.assertEqual({r.gaze_class for r in by_person["baby"]}, {"ANYARA"})
        first = by_person["mother"][0]
        self.assertAlmostEqual(first.gaze_x, 0.3, delta=0.02)
        self.assertGreater(first.partner_mass, 0.8)
        self.assertAlmostEqual(by_person["baby"][1].time_s, 2 / 60)

    def test_gaze_elsewhere_and_out_of_frame(self) -> None:
        self.clip("window")
        estimator = FakeGazeEstimator({0: (0.5, 0.9), 1: (0.5, 0.9)})
        run_gaze(self.paths, "7", self.window_tracker(), estimator, GazeConfig(), overlay=False)
        rows = read_timeline(self.out / "window_play.csv")
        self.assertEqual({r.gaze_class for r in rows if r.person == "mother"}, {"MASHOVA"})
        self.assertEqual({r.gaze_class for r in rows if r.person == "baby"}, {"TARGY_KORNYEZET"})

    def test_missing_head_is_not_codable_and_costs_no_inference(self) -> None:
        self.clip("window")
        estimator = FakeGazeEstimator({0: (0.3, 0.4), 1: (0.8, 0.4)})
        tracker = self.window_tracker(mother_frames=FRAMES[:10])  # mother lost after 10 samples

        run_gaze(self.paths, "7", tracker, estimator, GazeConfig(), overlay=False)

        rows = read_timeline(self.out / "window_play.csv")
        lost = [r for r in rows if r.person == "mother" and r.frame_index >= 20]
        self.assertTrue(lost)
        self.assertEqual(
            {(r.gaze_class, r.reason, r.gaze_x) for r in lost}, {("NK", "no_head", None)}
        )
        # the baby's partner (mother) head is gone too, so the baby row is not codable
        baby_after = [r for r in rows if r.person == "baby" and r.frame_index >= 20]
        self.assertEqual({r.reason for r in baby_after}, {"partner_not_found"})
        self.assertEqual(sum(estimator.calls), 30, "the baby head is on every sampled frame")

    def test_run_record_names_the_model_variant(self) -> None:
        self.clip("window")
        record = run_gaze(self.paths, "7", self.window_tracker(), FakeGazeEstimator({}),
                          GazeConfig(), overlay=False)  # fmt: skip
        self.assertEqual(record.model_variant, "fake")
        self.assertTrue((self.out / "run.json").exists())


class TestCloseCamera(PipelineCase):
    def test_zone_decides_partner_gaze(self) -> None:
        self.clip("baby")
        tracker = FakeHeadTracker({"baby_play.mp4": [track(5, 0.5, frames=FRAMES)]})
        run_gaze(
            self.paths, "7", tracker, FakeGazeEstimator({0: (0.2, 0.2)}), CLOSE_CFG, overlay=False
        )
        rows = read_timeline(self.out / "baby_play.csv")
        self.assertEqual({r.person for r in rows}, {"baby"})
        self.assertEqual({r.gaze_class for r in rows}, {"ANYARA"})

    def test_uncalibrated_zone_is_not_codable(self) -> None:
        self.clip("baby")
        tracker = FakeHeadTracker({"baby_play.mp4": [track(5, 0.5, frames=FRAMES)]})
        run_gaze(
            self.paths,
            "7",
            tracker,
            FakeGazeEstimator({0: (0.2, 0.2)}),
            GazeConfig(),
            overlay=False,
        )
        rows = read_timeline(self.out / "baby_play.csv")
        self.assertEqual({(r.gaze_class, r.reason) for r in rows}, {("NK", "no_partner_zone")})


class TestOperation(PipelineCase):
    def test_overlays_grid_and_resume(self) -> None:
        self.clip("window")
        estimator = FakeGazeEstimator({0: (0.3, 0.4), 1: (0.8, 0.4)})
        tracker = self.window_tracker()
        run_gaze(self.paths, "7", tracker, estimator, GazeConfig())
        self.assertTrue((self.out / "overlay" / "window_play.mp4").exists())
        self.assertTrue((self.out / "overlay" / "session_play.mp4").exists())

        run_gaze(self.paths, "7", tracker, estimator, GazeConfig())
        self.assertEqual(len(tracker.calls), 1, "finished clip is skipped")
        run_gaze(self.paths, "7", tracker, estimator, GazeConfig(), overwrite=True)
        self.assertEqual(len(tracker.calls), 2)

    def test_failing_clip_is_recorded_and_others_continue(self) -> None:
        self.clip("window")
        (self.processed / "baby_play.mp4").write_bytes(b"broken")
        tracker = self.window_tracker()
        record = run_gaze(
            self.paths, "7", tracker, FakeGazeEstimator({}), GazeConfig(), overlay=False
        )
        self.assertEqual(len(record.flags["failed"]), 1)
        self.assertTrue((self.out / "window_play.csv").exists())

    def test_ambiguous_roles_flag_and_no_gaze_rows(self) -> None:
        self.clip("window")
        tracker = FakeHeadTracker({"window_play.mp4": [track(1, 0.5), track(2, 0.51)]})
        estimator = FakeGazeEstimator({})
        record = run_gaze(self.paths, "7", tracker, estimator, GazeConfig(), overlay=False)
        self.assertEqual(record.flags["roles_ambiguous"], ["window_play"])
        rows = read_timeline(self.out / "window_play.csv")
        self.assertEqual({(r.gaze_class, r.reason) for r in rows}, {("NK", "roles_ambiguous")})
        self.assertEqual(estimator.calls, [])

    def test_camera_and_phase_filters(self) -> None:
        self.clip("window"), self.clip("window", "reunion")
        tracker = self.window_tracker()
        run_gaze(self.paths, "7", tracker, FakeGazeEstimator({}), GazeConfig(),
                 cameras=["window"], phases=["play"], overlay=False)  # fmt: skip
        self.assertEqual([c.name for c, _ in tracker.calls], ["window_play.mp4"])


class TestBatching(PipelineCase):
    def test_frames_are_estimated_in_batches(self) -> None:
        self.clip("window")
        estimator = FakeGazeEstimator({})
        run_gaze(self.paths, "7", self.window_tracker(), estimator, GazeConfig(), overlay=False,
                 batch_size=8)  # fmt: skip
        self.assertEqual(sum(estimator.calls), 30)
        self.assertEqual(max(estimator.calls), 8)


class TestUndecodableFrames(PipelineCase):
    def test_frames_the_clip_cannot_deliver_are_not_codable(self) -> None:
        clip = self.clip("window")
        indices = [0, 10, 5_000, 5_002]  # the clip has 60 frames
        prepared = ClipHeads(
            indices, 60.0, RoleAssignment(2, 1),
            {"mother": [box(0.8)] * 4, "baby": [box(0.3)] * 4},
        )  # fmt: skip
        estimator = FakeGazeEstimator({0: (0.3, 0.4), 1: (0.8, 0.4)})

        decisions = _decide(prepared, "window", clip, estimator, GazeConfig(), batch_size=8)

        self.assertEqual(decisions[(0, "mother")].gaze_class, "BABARA")
        self.assertEqual(decisions[(1, "baby")].gaze_class, "ANYARA")
        for position in (2, 3):
            self.assertEqual(decisions[(position, "mother")].reason, "no_frame")
            self.assertEqual(decisions[(position, "baby")].reason, "no_frame")
        self.assertEqual(sum(estimator.calls), 2)


class TestReadFrames(PipelineCase):
    def test_yields_rgb_frames_of_the_requested_indices_in_order(self) -> None:
        clip = self.clip("window")
        frames = list(read_frames(clip, [0, 5, 30]))
        self.assertEqual(len(frames), 3)
        self.assertEqual(frames[0].shape, (120, 160, 3))
        self.assertEqual(frames[0].dtype.name, "uint8")

    def test_no_indices_yields_nothing(self) -> None:
        self.assertEqual(list(read_frames(self.clip("window"), [])), [])

    def test_index_past_the_end_ends_the_stream_early(self) -> None:
        clip = self.clip("window")
        self.assertEqual(len(list(read_frames(clip, [0, 10, 10_000, 20_000]))), 2)

    def test_unreadable_clip_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            list(read_frames(self.tmp / "missing.mp4", [0]))


if __name__ == "__main__":
    unittest.main()
