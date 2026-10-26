import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stillface.config import DatasetPaths
from tests.helpers import make_video

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(HAS_TRANSFORMERS, "transformers (features extra) is not installed")
class TestOwlHeadTracker(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from stillface.features.gaze.owl_heads import OwlHeadTracker

        cls.tracker = OwlHeadTracker(device="cpu", batch_size=2, rate_hz=2.0, min_score=0.3)

    def test_clip_without_heads_gives_a_tagged_track_without_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clip = make_video(Path(tmp) / "c.mp4", seconds=2, fps=30, size="320x240", audio=False)
            tracks = self.tracker.track(clip, list(range(0, 60, 3)))
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].role, "baby")
        self.assertEqual(tracks[0].boxes, {})

    def test_dataset_clip_finds_the_baby_head_with_interpolated_frames(self) -> None:
        clip = DatasetPaths.default().phase_clip("605317", "window", "play")
        if not clip.exists():
            self.skipTest("development session not available")
        frames = list(range(600, 1000, 4))
        (track,) = self.tracker.track(clip, frames)
        self.assertGreater(len(track.boxes), 0.8 * len(frames))
        centers = np.array([b.center for b in track.boxes.values()])
        # the baby lies in the left half of the window camera image, below the middle
        self.assertTrue(((centers[:, 0] < 0.5) & (centers[:, 1] > 0.4)).all())
        interpolated = [f for f in track.boxes if f not in track.scores]
        self.assertTrue(interpolated)


if __name__ == "__main__":
    unittest.main()
