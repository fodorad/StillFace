import importlib.util
import tempfile
import unittest
from pathlib import Path

from stillface.config import DatasetPaths
from tests.helpers import make_video

HAS_EXORDIUM = importlib.util.find_spec("exordium") is not None


@unittest.skipUnless(HAS_EXORDIUM, "exordium (features extra) is not installed")
class TestExordiumHeadTracker(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from stillface.features.gaze.exordium_heads import ExordiumHeadTracker

        cls.tracker = ExordiumHeadTracker(device_id=-1, batch_size=4)

    def test_clip_without_faces_gives_no_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clip = make_video(Path(tmp) / "c.mp4", seconds=1, fps=30, size="320x240", audio=False)
            self.assertEqual(self.tracker.track(clip, list(range(0, 30, 3))), [])

    def test_frames_beyond_the_decodable_range_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clip = make_video(Path(tmp) / "c.mp4", seconds=1, fps=30, size="320x240", audio=False)
            self.assertEqual(self.tracker.track(clip, [0, 10, 10_000]), [])

    def test_dataset_clip_yields_normalized_boxes_on_the_requested_frames(self) -> None:
        clip = DatasetPaths.default().phase_clip("605317", "mother", "play")
        if not clip.exists():
            self.skipTest("development session not available")
        frames = list(range(0, 120, 2))
        tracks = self.tracker.track(clip, frames)
        self.assertGreaterEqual(len(tracks), 1)
        for track in tracks:
            self.assertTrue(set(track.boxes) <= set(frames))
            for box in track.boxes.values():
                self.assertTrue(0.0 <= box.x0 < box.x1 <= 1.0 and 0.0 <= box.y0 < box.y1 <= 1.0)

    def test_sparse_sampling_still_links_a_face_into_one_track(self) -> None:
        clip = DatasetPaths.default().phase_clip("626021", "baby", "play")
        if not clip.exists():
            self.skipTest("development session not available")
        frames = list(range(0, 1800, 60))  # 1 fps on a 60 fps clip
        tracks = self.tracker.track(clip, frames)
        self.assertTrue(any(len(t.boxes) >= 0.8 * len(frames) for t in tracks))


if __name__ == "__main__":
    unittest.main()
