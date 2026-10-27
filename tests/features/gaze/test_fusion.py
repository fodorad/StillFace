import unittest
from pathlib import Path

from stillface.features.gaze.fusion import (
    SECONDARY_ID_OFFSET,
    FusedHeadTracker,
    fuse_tracks,
    merge_duplicates,
)
from stillface.features.gaze.heads import HeadBox, HeadTrack
from tests.features.gaze.fakes import FakeHeadTracker, box


class TestFuseTracks(unittest.TestCase):
    def test_face_inside_a_head_is_merged_and_fills_missing_frames(self) -> None:
        head = HeadTrack(
            0, {0: box(0.3, width=0.12), 1: box(0.3, width=0.12)}, scores={0: 0.5, 1: 0.4}
        )
        face = HeadTrack(
            7, {1: box(0.3, width=0.06), 2: box(0.3, width=0.06)}, scores={1: 0.9, 2: 0.8}
        )

        (fused,) = fuse_tracks([head], [face])

        self.assertEqual(fused.track_id, 0)
        self.assertEqual(sorted(fused.boxes), [0, 1, 2])
        self.assertEqual(fused.boxes[1], head.boxes[1], "the face lies inside the head box")
        self.assertEqual(fused.boxes[2], face.boxes[2], "frames only the face detector saw")
        self.assertEqual(fused.scores, {0: 0.5, 1: 0.9, 2: 0.8})

    def test_face_frames_mark_the_fused_head_as_face_visible(self) -> None:
        head = HeadTrack(0, {f: box(0.3, width=0.12) for f in range(4)})
        face = HeadTrack(
            7, {1: box(0.3, width=0.06), 2: box(0.3, width=0.06)}, scores={1: 0.9, 2: 0.8}
        )
        lonely = HeadTrack(8, {0: box(0.8, width=0.06)}, scores={0: 0.7})
        fused = {t.track_id: t for t in fuse_tracks([head], [face, lonely])}
        self.assertEqual(fused[0].faces, {1: 0.9, 2: 0.8})
        self.assertEqual(fused[SECONDARY_ID_OFFSET + 8].faces, {0: 0.7}, "a face track is a face")

    def test_union_when_boxes_only_partly_overlap(self) -> None:
        head = HeadTrack(0, {0: HeadBox(0.2, 0.2, 0.4, 0.4)})
        face = HeadTrack(1, {0: HeadBox(0.3, 0.3, 0.45, 0.45)})
        (fused,) = fuse_tracks([head], [face])
        self.assertEqual(fused.boxes[0], HeadBox(0.2, 0.2, 0.45, 0.45))

    def test_unmatched_faces_are_kept_with_offset_ids(self) -> None:
        head = HeadTrack(0, {0: box(0.3)})
        other = HeadTrack(2, {0: box(0.8)})
        fused = fuse_tracks([head], [other])
        self.assertEqual(sorted(t.track_id for t in fused), [0, SECONDARY_ID_OFFSET + 2])

    def test_a_face_merges_into_one_head_only(self) -> None:
        heads = [HeadTrack(0, {0: box(0.3)}), HeadTrack(1, {0: box(0.31)})]
        face = HeadTrack(5, {0: box(0.3, width=0.06)})
        fused = fuse_tracks(heads, [face])
        self.assertEqual(len(fused), 2)


class TestMergeDuplicates(unittest.TestCase):
    def test_nested_boxes_of_one_head_become_one_track(self) -> None:
        head = HeadTrack(
            0, {f: box(0.3, width=0.12) for f in range(10)}, scores={f: 0.6 for f in range(10)}
        )
        inner = HeadTrack(1, {f: box(0.3, width=0.06) for f in range(5, 12)},
                          scores={f: 0.9 for f in range(5, 12)})  # fmt: skip

        (merged,) = merge_duplicates([inner, head])

        self.assertEqual(merged.track_id, 0, "the longer track keeps its id")
        self.assertEqual(sorted(merged.boxes), list(range(12)))
        self.assertEqual(merged.boxes[7], inner.boxes[7], "higher score wins on shared frames")
        self.assertEqual(merged.boxes[2], head.boxes[2])

    def test_separate_heads_stay_separate(self) -> None:
        a = HeadTrack(0, {f: box(0.3) for f in range(5)})
        b = HeadTrack(1, {f: box(0.8) for f in range(5)})
        no_shared = HeadTrack(2, {f: box(0.3) for f in range(50, 55)})
        self.assertEqual(len(merge_duplicates([a, b, no_shared])), 3)


class TestFusedHeadTracker(unittest.TestCase):
    def test_runs_both_trackers_on_the_clip(self) -> None:
        heads = FakeHeadTracker({"c.mp4": [HeadTrack(0, {0: box(0.3)})]})
        faces = FakeHeadTracker({"c.mp4": [HeadTrack(0, {0: box(0.3, width=0.05)})]})
        tracks = FusedHeadTracker(heads, faces).track(Path("c.mp4"), [0])
        self.assertEqual(len(tracks), 1)
        self.assertEqual(len(heads.calls), 1)
        self.assertEqual(len(faces.calls), 1)


if __name__ == "__main__":
    unittest.main()
