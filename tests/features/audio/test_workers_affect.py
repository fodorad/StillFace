import tempfile
import unittest
from pathlib import Path

import numpy as np

from stillface.features.audio.models import InferenceRequest, Segment
from stillface.features.audio.smoke import synthetic_clip
from stillface.features.audio.workers import audeering, emotion2vec
from stillface.features.audio.workers.base import segment_windows


class TestSegmentWindows(unittest.TestCase):
    def test_windows_stay_inside_segments(self):
        segments = (Segment(0.0, 4.0, "mother", "FEM"), Segment(10.0, 11.0, "mother", "FEM"))
        windows = segment_windows(segments, 3.0, 0.5)
        self.assertEqual(
            [(w.start_s, w.end_s) for w in windows],
            [(0.0, 3.0), (0.5, 3.5), (1.0, 4.0), (10.0, 11.0)],
        )
        self.assertTrue(all(w.role == "mother" for w in windows))
        self.assertEqual(segment_windows((), 3.0, 0.5), ())


class TestAudeering(unittest.TestCase):
    def test_postprocess(self):
        windows = (Segment(0, 3), Segment(1, 4))
        scores = audeering.postprocess(
            windows, np.array([[0.2, 0.5, 1.3], [-0.1, 0.4, 0.6]]), np.ones((2, 8))
        )
        self.assertEqual(scores.names, ("arousal", "dominance", "valence"))
        np.testing.assert_allclose(scores.values, [[0.2, 0.5, 1.0], [0.0, 0.4, 0.6]])
        self.assertEqual(scores.embeddings.shape, (2, 8))
        with self.assertRaises(ValueError):
            audeering.postprocess(windows, np.zeros((1, 3)), np.ones((2, 8)))


class TestEmotion2vec(unittest.TestCase):
    def test_class_names(self):
        self.assertEqual(emotion2vec.class_name("开心/happy"), "happy")
        self.assertEqual(emotion2vec.class_name("<unk>"), "unknown")
        with self.assertRaises(ValueError):
            emotion2vec.class_name("bored")

    def test_postprocess_reorders_and_normalises(self):
        windows = (Segment(0, 3),)
        labels = [["中立/neutral", "开心/happy", "难过/sad"]]
        scores = emotion2vec.postprocess(windows, labels, [[0.5, 0.3, 0.2]], np.ones((1, 4)))
        self.assertEqual(scores.names, emotion2vec.CLASSES)
        self.assertAlmostEqual(scores.values[0, emotion2vec.CLASSES.index("neutral")], 0.5)
        self.assertAlmostEqual(scores.values[0].sum(), 1.0)
        zero = emotion2vec.postprocess(windows, [["<unk>"]], [[0.0]], np.ones((1, 4)))
        self.assertEqual(zero.values.sum(), 0.0)
        with self.assertRaises(ValueError):
            emotion2vec.postprocess(windows, [], [], np.ones((0, 4)))


class FakeAudeering(audeering.AudeeringWorker):
    def __init__(self):
        self.window_s, self.hop_s = 3.0, 0.5

    def forward(self, clips, rate):
        return np.full((len(clips), 3), 0.5), np.zeros((len(clips), 16))


class FakeE2v(emotion2vec.Emotion2vecWorker):
    def __init__(self):
        self.window_s, self.hop_s = 3.0, 0.5

    def forward(self, clips, rate):
        n = len(clips)
        return [["开心/happy"]] * n, [[1.0]] * n, np.zeros((n, 16))


class TestInfer(unittest.TestCase):
    def test_windows_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = synthetic_clip(Path(tmp) / "c.wav", seconds=5.0)
            request = InferenceRequest(
                "1/play/mother", wav, 5.0, (Segment(0.5, 4.0, "mother", "FEM"),)
            )
            a = FakeAudeering().infer(request).segments["windows"]
            e = FakeE2v().infer(request).segments["windows"]
        self.assertEqual(len(a.segments), 2)
        self.assertEqual(e.values.shape, (2, 9))


if __name__ == "__main__":
    unittest.main()
