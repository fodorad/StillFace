import tempfile
import unittest
from pathlib import Path

import numpy as np

from stillface.features.audio.models import (
    FrameScores,
    InferenceRequest,
    InferenceResult,
    Segment,
    SegmentScores,
    load_result,
    request_from_dict,
    request_to_dict,
    save_result,
)


class TestModels(unittest.TestCase):
    def test_shapes_validated(self):
        with self.assertRaises(ValueError):
            FrameScores(np.zeros(3), ("a",), np.zeros((3, 2)))
        scores = FrameScores([0, 1], ("a", "b"), np.zeros(4))
        self.assertEqual(scores.values.shape, (2, 2))
        seg = SegmentScores((Segment(0, 1),), ("p", "q"), [0.1, 0.9], np.zeros(8))
        self.assertEqual(seg.embeddings.shape, (1, 8))

    def test_result_round_trip(self):
        rng = np.random.default_rng(0)
        segs = (Segment(0.0, 1.5, "mother", "FEM"), Segment(2.0, 2.5, "baby", "KCHI"))
        result = InferenceResult(
            "1/play",
            {"scores": FrameScores(np.arange(5) / 10, ("a", "b"), rng.uniform(size=(5, 2)))},
            {
                "emb": SegmentScores(segs, ("p",), [[0.2], [0.4]], rng.normal(size=(2, 3))),
                "plain": SegmentScores((), ("x",), np.zeros((0, 1))),
            },
            {"rttm": "SPEAKER x\n"},
            {"device": "cpu", "n": 2},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.npz"
            save_result(result, path)
            back = load_result(path)
        self.assertEqual(
            (back.key, back.meta, back.artefacts), ("1/play", result.meta, result.artefacts)
        )
        np.testing.assert_array_equal(back.frames["scores"].values, result.frames["scores"].values)
        self.assertEqual(back.frames["scores"].names, ("a", "b"))
        self.assertEqual(back.segments["emb"].segments, segs)
        np.testing.assert_array_equal(
            back.segments["emb"].embeddings, result.segments["emb"].embeddings
        )
        self.assertIsNone(back.segments["plain"].embeddings)
        self.assertEqual(back.segments["plain"].values.shape, (0, 1))

    def test_request_round_trip(self):
        for segments in (None, (Segment(0.5, 1.0, "baby", "KCHI"),)):
            request = InferenceRequest("1/play/mother", Path("a.wav"), 3.0, segments, {"x": 1})
            self.assertEqual(request_from_dict(request_to_dict(request)), request)


if __name__ == "__main__":
    unittest.main()
