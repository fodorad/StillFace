import unittest
from pathlib import Path

from stillface.features.gaze.combined import (
    TAGGED_ID_OFFSET,
    CombinedHeadTracker,
    PerCameraTracker,
    anchor_frames,
    iou,
    overlaps,
)
from stillface.features.gaze.heads import HeadBox, HeadTrack
from tests.features.gaze.fakes import FakeHeadTracker, box, track


class TestOverlap(unittest.TestCase):
    def test_iou(self) -> None:
        a = HeadBox(0, 0, 0.2, 0.2)
        self.assertAlmostEqual(iou(a, a), 1.0)
        self.assertEqual(iou(a, HeadBox(0.5, 0.5, 0.7, 0.7)), 0.0)
        self.assertAlmostEqual(iou(a, HeadBox(0.1, 0, 0.3, 0.2)), 1 / 3)

    def test_overlaps(self) -> None:
        a, b = track(1, 0.3), track(2, 0.31)
        self.assertTrue(overlaps(a, b, 0.3, 0.5))
        self.assertFalse(overlaps(a, track(3, 0.8), 0.3, 0.5))
        self.assertFalse(overlaps(a, HeadTrack(4, {99: box(0.3)}), 0.3, 0.5), "no shared frames")


class TestCombinedTracker(unittest.TestCase):
    def test_duplicate_face_tracks_are_dropped_and_ids_do_not_collide(self) -> None:
        baby = HeadTrack(0, {f: box(0.3) for f in range(10)}, role="baby")
        tagged = FakeHeadTracker({"c.mp4": [baby]})
        faces = FakeHeadTracker(
            {"c.mp4": [track(1, 0.3, width=0.09), track(2, 0.8), track(0, 0.9)]}
        )  # the first face is the baby's, the others are not

        tracks = CombinedHeadTracker(tagged, faces).track(Path("c.mp4"), list(range(10)))

        self.assertEqual(sorted(t.track_id for t in tracks), [0, 2, TAGGED_ID_OFFSET])
        self.assertEqual([t.role for t in tracks if t.track_id == TAGGED_ID_OFFSET], ["baby"])


class TestPerCameraTracker(unittest.TestCase):
    def test_routes_by_camera_name(self) -> None:
        side = FakeHeadTracker({"window_play.mp4": [track(1, 0.3)]})
        close = FakeHeadTracker({"baby_play.mp4": [track(2, 0.5)]})
        router = PerCameraTracker(side, close)
        self.assertEqual(router.track(Path("x/window_play.mp4"), [0])[0].track_id, 1)
        self.assertEqual(router.track(Path("x/baby_play.mp4"), [0])[0].track_id, 2)
        self.assertEqual(len(side.calls), 1)
        self.assertEqual(len(close.calls), 1)


class TestAnchorFrames(unittest.TestCase):
    def test_every_20th_frame_for_3hz_on_a_60fps_clip(self) -> None:
        frames = list(range(0, 120, 2))
        self.assertEqual(anchor_frames(frames, 60.0, 3.0), [0, 20, 40, 60, 80, 100])

    def test_nearest_requested_frame_is_used(self) -> None:
        self.assertEqual(anchor_frames([0, 7, 15, 22], 30.0, 2.0), [0, 15])

    def test_empty(self) -> None:
        self.assertEqual(anchor_frames([], 60.0, 3.0), [])


if __name__ == "__main__":
    unittest.main()
