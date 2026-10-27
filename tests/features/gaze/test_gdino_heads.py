import importlib.util
import tempfile
import unittest
from pathlib import Path

from stillface.config import DatasetPaths
from tests.helpers import make_video

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(HAS_TRANSFORMERS, "transformers (features extra) is not installed")
class TestGroundingDinoHeadTracker(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from stillface.features.gaze.gdino_heads import GroundingDinoHeadTracker

        cls.tracker = GroundingDinoHeadTracker(rate_hz=1.0, batch_size=2)

    def test_clip_without_heads_gives_no_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clip = make_video(
                Path(tmp) / "c.mp4", seconds=2, fps=30, size="320x240", audio=False,
                source="color",
            )  # fmt: skip
            self.assertEqual(self.tracker.track(clip, list(range(0, 60, 3))), [])

    def test_detections_are_cached_per_clip_and_anchor_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clip = make_video(
                Path(tmp) / "s" / "processed" / "window_play.mp4", seconds=1, fps=30,
                size="320x240", audio=False,
            )  # fmt: skip
            cached, model = self.tracker, self.tracker.model
            cached.cache_dir = Path(tmp) / "cache"
            self.addCleanup(setattr, cached, "cache_dir", None)
            self.addCleanup(setattr, cached, "model", model)
            first = cached.detect_clip(clip, [0, 15])
            files = list((Path(tmp) / "cache").glob("s_window_play_*.json"))
            self.assertEqual(len(files), 1)
            cached.model = None  # a cache hit must not touch the model
            self.assertEqual(cached.detect_clip(clip, [0, 15]), first)
            cached.model = model
            cached.detect_clip(clip, [0, 10])
            self.assertEqual(len(list((Path(tmp) / "cache").glob("*.json"))), 2)

    def test_side_camera_clip_shows_two_persistent_heads(self) -> None:
        clip = DatasetPaths.default().phase_clip("605317", "window", "play")
        if not clip.exists():
            self.skipTest("development session not available")
        frames = list(range(600, 1200, 6))
        tracks = self.tracker.track(clip, frames)
        persistent = [t for t in tracks if len(t.boxes) > 0.6 * len(frames)]
        self.assertGreaterEqual(len(persistent), 2, "mother and baby")
        for t in persistent:
            interpolated = set(t.boxes) - set(t.scores)
            self.assertFalse(interpolated, "densified frames carry scores")


if __name__ == "__main__":
    unittest.main()
