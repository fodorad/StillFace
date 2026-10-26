import unittest

from stillface.features.gaze.heads import HeadBox, interpolate_boxes
from stillface.features.gaze.selection import Candidate, select_path
from tests.features.gaze.fakes import box


def cand(cx: float, score: float, cy: float = 0.4) -> Candidate:
    return Candidate(box(cx, cy), score)


class TestSelectPath(unittest.TestCase):
    def test_single_best_candidate_per_frame_when_nothing_competes(self) -> None:
        path = select_path([[cand(0.3, 0.5)], [cand(0.31, 0.6)]])
        self.assertEqual([c.score for c in path], [0.5, 0.6])

    def test_smooth_weaker_box_beats_a_stronger_box_elsewhere(self) -> None:
        frames = [
            [cand(0.30, 0.5)],
            [cand(0.31, 0.3), cand(0.85, 0.45)],  # the other person scores higher here
            [cand(0.30, 0.5)],
        ]
        path = select_path(frames, jump_penalty=2.0)
        self.assertEqual([round(c.box.center[0], 2) for c in path], [0.30, 0.31, 0.30])

    def test_strong_evidence_can_still_move_the_track(self) -> None:
        frames = [[cand(0.3, 0.5)]] + [[cand(0.8, 0.9)]] * 4
        path = select_path(frames, jump_penalty=0.5)
        self.assertAlmostEqual(path[-1].box.center[0], 0.8)

    def test_frames_without_candidates_stay_empty(self) -> None:
        path = select_path([[cand(0.3, 0.5)], [], [cand(0.3, 0.5)]])
        self.assertIsNone(path[1])
        self.assertIsNotNone(path[0])
        self.assertIsNotNone(path[2])

    def test_min_score_filters_candidates(self) -> None:
        path = select_path([[cand(0.3, 0.05)], [cand(0.3, 0.4)]], min_score=0.1)
        self.assertIsNone(path[0])
        self.assertIsNotNone(path[1])

    def test_no_frames(self) -> None:
        self.assertEqual(select_path([]), [])


class TestInterpolateBoxes(unittest.TestCase):
    def test_linear_between_anchors_and_exact_at_anchors(self) -> None:
        anchors = {0: HeadBox(0.0, 0.0, 0.2, 0.2), 10: HeadBox(0.5, 0.5, 0.7, 0.7)}
        boxes = interpolate_boxes(anchors, [0, 5, 10], max_gap=10)
        self.assertEqual(boxes[0], anchors[0])
        self.assertEqual(boxes[10], anchors[10])
        self.assertAlmostEqual(boxes[5].x0, 0.25)
        self.assertAlmostEqual(boxes[5].y1, 0.45)

    def test_outside_the_anchors_and_across_large_gaps_stays_empty(self) -> None:
        anchors = {10: HeadBox(0, 0, 0.1, 0.1), 40: HeadBox(0.1, 0.1, 0.2, 0.2)}
        boxes = interpolate_boxes(anchors, [0, 10, 20, 40, 50], max_gap=15)
        self.assertEqual(set(boxes), {10, 40})

    def test_no_anchors(self) -> None:
        self.assertEqual(interpolate_boxes({}, [1, 2], max_gap=5), {})


if __name__ == "__main__":
    unittest.main()
