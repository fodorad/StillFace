import tempfile
import unittest
from pathlib import Path

from stillface.features.gaze.config import GazeConfig
from stillface.features.gaze.prepare import clip_properties, prepare_clip
from tests.features.gaze.fakes import FakeHeadTracker, track
from tests.helpers import make_video


class TestClipProperties(unittest.TestCase):
    def test_frame_count_and_fps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clip = make_video(Path(tmp) / "c.mp4", seconds=2, fps=30, audio=False)
            n_frames, fps = clip_properties(clip)
            self.assertEqual((n_frames, round(fps)), (60, 30))

    def test_unreadable_clip(self) -> None:
        with self.assertRaises(RuntimeError):
            clip_properties(Path("missing.mp4"))


class TestPrepareClip(unittest.TestCase):
    def test_samples_at_the_configured_rate_and_assigns_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clip = make_video(Path(tmp) / "window_play.mp4", seconds=1, fps=60, audio=False)
            frames = list(range(0, 60, 2))
            tracker = FakeHeadTracker(
                {"window_play.mp4": [track(1, 0.3, frames=frames), track(2, 0.8, frames=frames)]}
            )

            result = prepare_clip(clip, "window", "play", tracker, GazeConfig())

            self.assertEqual(result.frame_indices, frames)
            self.assertEqual(tracker.calls, [(clip, frames)])
            self.assertEqual((result.assignment.mother, result.assignment.baby), (2, 1))
            self.assertEqual(len(result.heads["mother"]), 30)
            self.assertIsNotNone(result.heads["baby"][0])

    def test_overrides_are_applied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clip = make_video(Path(tmp) / "window_play.mp4", seconds=1, fps=60, audio=False)
            tracker = FakeHeadTracker({"window_play.mp4": [track(1, 0.3), track(2, 0.8)]})
            overrides = {("window", "play"): {"mother": 1, "baby": 2}}
            result = prepare_clip(
                clip, "window", "play", tracker, GazeConfig(), overrides=overrides
            )
            self.assertEqual((result.assignment.mother, result.assignment.baby), (1, 2))


if __name__ == "__main__":
    unittest.main()
