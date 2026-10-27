import unittest

from stillface.features.gaze.heads import HeadTrack
from stillface.features.gaze.tracking import (
    Candidate,
    anchor_frames,
    densify,
    link_candidates,
    stitch_tracks,
)
from tests.features.gaze.fakes import box


def cand(cx: float, score: float = 0.5, width: float = 0.1) -> Candidate:
    return Candidate(box(cx, width=width), score)


class TestLinkCandidates(unittest.TestCase):
    def test_two_heads_moving_slowly_become_two_tracks(self) -> None:
        frames = [0, 20, 40, 60]
        candidates = [[cand(0.3 + 0.02 * k), cand(0.8 - 0.02 * k)] for k in range(4)]
        tracks = link_candidates(candidates, frames)
        self.assertEqual(len(tracks), 2)
        for t in tracks:
            self.assertEqual(sorted(t.boxes), frames)
        xs = sorted(round(t.boxes[0].center[0], 2) for t in tracks)
        self.assertEqual(xs, [0.3, 0.8])

    def test_order_of_candidates_does_not_swap_identities(self) -> None:
        candidates = [[cand(0.3), cand(0.8)], [cand(0.8), cand(0.3)], [cand(0.31), cand(0.79)]]
        tracks = link_candidates(candidates, [0, 1, 2])
        for t in tracks:
            centers = [b.center[0] for b in t.boxes.values()]
            self.assertLess(max(centers) - min(centers), 0.05)

    def test_jump_far_beyond_the_head_size_starts_a_new_track(self) -> None:
        tracks = link_candidates([[cand(0.1, width=0.05)], [cand(0.9, width=0.05)]], [0, 1])
        self.assertEqual(len(tracks), 2)

    def test_short_dropouts_are_bridged_long_ones_are_not(self) -> None:
        candidates = [[cand(0.3)], [], [], [cand(0.3)], [], [], [], [cand(0.3)]]
        tracks = link_candidates(candidates, list(range(8)), max_missed=2)
        self.assertEqual(sorted(len(t.boxes) for t in tracks), [1, 2])

    def test_scores_are_kept(self) -> None:
        (t,) = link_candidates([[cand(0.3, score=0.4)], [cand(0.3, score=0.6)]], [5, 7])
        self.assertEqual(t.scores, {5: 0.4, 7: 0.6})

    def test_nothing_to_link(self) -> None:
        self.assertEqual(link_candidates([[], []], [0, 1]), [])


class TestStitchTracks(unittest.TestCase):
    def piece(self, track_id: int, cx: float, frames: range) -> HeadTrack:
        return HeadTrack(track_id, {f: box(cx) for f in frames}, scores={f: 0.5 for f in frames})

    def test_pieces_of_one_head_are_joined(self) -> None:
        tracks = [self.piece(0, 0.30, range(0, 10)), self.piece(1, 0.31, range(15, 30))]
        (stitched,) = stitch_tracks(tracks, max_gap=10)
        self.assertEqual(stitched.track_id, 0)
        self.assertEqual(len(stitched.boxes), 25)
        self.assertEqual(len(stitched.scores), 25)

    def test_distant_or_late_pieces_stay_apart(self) -> None:
        far = [self.piece(0, 0.3, range(0, 10)), self.piece(1, 0.8, range(15, 30))]
        self.assertEqual(len(stitch_tracks(far, max_gap=10)), 2)
        late = [self.piece(0, 0.3, range(0, 10)), self.piece(1, 0.3, range(50, 60))]
        self.assertEqual(len(stitch_tracks(late, max_gap=10)), 2)

    def test_overlapping_tracks_are_two_heads(self) -> None:
        both = [self.piece(0, 0.3, range(0, 10)), self.piece(1, 0.31, range(5, 15))]
        self.assertEqual(len(stitch_tracks(both, max_gap=10)), 2)

    def test_closest_continuation_wins_and_chains(self) -> None:
        tracks = [
            self.piece(0, 0.30, range(0, 10)),
            self.piece(1, 0.40, range(12, 20)),
            self.piece(2, 0.31, range(12, 20)),
            self.piece(3, 0.31, range(25, 30)),
        ]
        stitched = {t.track_id: t for t in stitch_tracks(tracks, max_gap=10)}
        self.assertIn(16, stitched[0].boxes)
        self.assertIn(27, stitched[0].boxes)
        self.assertAlmostEqual(stitched[0].boxes[16].center[0], 0.31)

    def test_face_flags_are_joined_too(self) -> None:
        a = HeadTrack(0, {f: box(0.3) for f in range(5)}, faces={1: 0.9})
        b = HeadTrack(1, {f: box(0.3) for f in range(8, 12)}, faces={9: 0.8})
        (stitched,) = stitch_tracks([a, b], max_gap=10)
        self.assertEqual(stitched.faces, {1: 0.9, 9: 0.8})

    def test_empty_tracks_are_dropped(self) -> None:
        self.assertEqual(stitch_tracks([HeadTrack(0, {})], max_gap=5), [])


class TestDensify(unittest.TestCase):
    def test_interpolated_frames_get_the_weaker_neighbor_score(self) -> None:
        sparse = HeadTrack(3, {0: box(0.3), 20: box(0.5)}, scores={0: 0.4, 20: 0.8})
        dense = densify(sparse, [0, 10, 20, 30], max_gap=20)
        self.assertEqual(sorted(dense.boxes), [0, 10, 20])
        self.assertAlmostEqual(dense.boxes[10].center[0], 0.4)
        self.assertEqual(dense.scores, {0: 0.4, 10: 0.4, 20: 0.8})
        self.assertEqual(dense.track_id, 3)

    def test_faces_stay_on_detected_frames_only(self) -> None:
        sparse = HeadTrack(
            3, {0: box(0.3), 20: box(0.5)}, scores={0: 0.4, 20: 0.8}, faces={20: 0.9}
        )
        dense = densify(sparse, [0, 10, 20, 30], max_gap=20)
        self.assertEqual(dense.faces, {20: 0.9})


class TestAnchorFrames(unittest.TestCase):
    def test_every_30th_frame_for_2hz_on_a_60fps_clip(self) -> None:
        self.assertEqual(anchor_frames(list(range(0, 120, 2)), 60.0, 2.0), [0, 30, 60, 90])

    def test_nearest_requested_frame_is_used(self) -> None:
        self.assertEqual(anchor_frames([0, 7, 15, 22], 30.0, 2.0), [0, 15])

    def test_empty(self) -> None:
        self.assertEqual(anchor_frames([], 60.0, 3.0), [])


if __name__ == "__main__":
    unittest.main()
