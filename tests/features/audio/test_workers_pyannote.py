import unittest
from pathlib import Path

import numpy as np

from stillface.features.audio.models import FrameScores, InferenceRequest, Segment
from stillface.features.audio.workers import pyannote_worker as pw


class TestPostprocess(unittest.TestCase):
    def test_speech_is_max_over_local_speakers(self):
        activity = np.array([[0.1, 0.7, 0.0], [0.0, 0.0, 0.0], [0.9, 0.2, 1.2]])
        scores = pw.speech_scores(0.0, 0.017, 0.062, activity)
        np.testing.assert_allclose(scores.values[:, 0], [0.7, 0.0, 1.0])
        self.assertAlmostEqual(scores.times[1], 0.017 + 0.031)
        self.assertEqual(pw.speech_scores(0.0, 0.1, 0.1, np.zeros((0, 3))).values.shape, (0, 1))

    def test_embeddings_normalised_with_nan_rows(self):
        segments = (Segment(0, 2), Segment(3, 3.2), Segment(4, 6))
        scores = pw.embedding_scores(segments, [np.array([3.0, 4.0]), None, np.zeros(2)])
        np.testing.assert_allclose(scores.embeddings[0], [0.6, 0.8])
        self.assertTrue(np.isnan(scores.embeddings[1]).all())
        self.assertTrue(np.isnan(scores.embeddings[2]).all())
        np.testing.assert_allclose(scores.values[:, 0], [2.0, 0.2, 2.0])
        with self.assertRaises(ValueError):
            pw.embedding_scores(segments, [None])
        self.assertEqual(pw.embedding_scores((), []).embeddings.shape, (0, 1))


class FakeForward(pw.PyannoteWorker):
    def forward(self, request):
        if request.segments is None:
            return FrameScores([0.5], ("speech",), [0.4])
        return [np.ones(4) for _ in request.segments]


class TestInfer(unittest.TestCase):
    def test_routes_by_request_kind(self):
        worker = FakeForward()
        vad = worker.infer(InferenceRequest("1/play", Path("x.wav"), 1.0))
        self.assertEqual(list(vad.frames), ["speech"])
        emb = worker.infer(InferenceRequest("1/play", Path("x.wav"), 1.0, (Segment(0, 1),)))
        self.assertEqual(emb.segments["embedding"].embeddings.shape, (1, 4))


if __name__ == "__main__":
    unittest.main()
