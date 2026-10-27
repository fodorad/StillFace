import unittest

from stillface.features.gaze.heads import (
    HeadBox,
    HeadTrack,
    containment,
    interpolate_boxes,
    iou,
    overlaps,
)
from tests.features.gaze.fakes import box, track


class TestOverlap(unittest.TestCase):
    def test_iou(self) -> None:
        a = HeadBox(0, 0, 0.2, 0.2)
        self.assertAlmostEqual(iou(a, a), 1.0)
        self.assertEqual(iou(a, HeadBox(0.5, 0.5, 0.7, 0.7)), 0.0)
        self.assertAlmostEqual(iou(a, HeadBox(0.1, 0, 0.3, 0.2)), 1 / 3)

    def test_containment(self) -> None:
        outer, inner = HeadBox(0, 0, 0.4, 0.4), HeadBox(0.1, 0.1, 0.2, 0.2)
        self.assertAlmostEqual(containment(outer, inner), 1.0)
        self.assertAlmostEqual(containment(inner, outer), 1.0)
        self.assertEqual(containment(outer, HeadBox(0.5, 0.5, 0.6, 0.6)), 0.0)

    def test_overlaps(self) -> None:
        a = track(1, 0.3)
        self.assertTrue(overlaps(a, track(2, 0.31), 0.3, 0.5))
        self.assertFalse(overlaps(a, track(3, 0.8), 0.3, 0.5))
        self.assertFalse(overlaps(a, HeadTrack(4, {99: box(0.3)}), 0.3, 0.5), "no shared frames")


class TestTrackMeasures(unittest.TestCase):
    def test_central_heads_have_more_evidence_than_edge_heads(self) -> None:
        central = HeadTrack(0, {f: box(0.5, 0.5) for f in range(5)})
        edge = HeadTrack(1, {f: box(0.05, 0.5) for f in range(5)})
        self.assertAlmostEqual(central.evidence(centrality=True), 5.0)
        self.assertLess(edge.evidence(centrality=True), 1.0)
        self.assertEqual(central.evidence(), edge.evidence())

    def test_median_score(self) -> None:
        t = HeadTrack(0, {0: box(0.5), 1: box(0.5), 2: box(0.5)}, scores={0: 0.2, 1: 0.4})
        self.assertAlmostEqual(t.median_score(), 0.4, msg="the unscored frame counts as 1")
        self.assertEqual(HeadTrack(1, {}).median_score(), 0.0)

    def test_face_share(self) -> None:
        t = HeadTrack(0, {f: box(0.5) for f in range(4)}, faces={0: 0.9, 3: 0.8, 99: 0.9})
        self.assertAlmostEqual(t.face_share(), 0.5)
        self.assertEqual(HeadTrack(1, {}).face_share(), 0.0)

    def test_median_area(self) -> None:
        t = HeadTrack(0, {0: box(0.5, width=0.1), 1: box(0.5, width=0.2), 2: box(0.5, width=0.3)})
        self.assertAlmostEqual(t.median_area(), box(0.5, width=0.2).area)
        self.assertEqual(HeadTrack(1, {}).median_area(), 0.0)


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
