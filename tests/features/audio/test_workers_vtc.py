import unittest

import numpy as np

from stillface.features.audio.models import Segment
from stillface.features.audio.workers import addressee, vtc

RTTM = (
    "SPEAKER audio 1 0.50 1.00 <NA> <NA> FEM <NA> <NA>\n"
    "SPEAKER audio 1 0.10 0.20 <NA> <NA> KCHI <NA> <NA>\n"
    "# comment\n"
)


class TestVtc(unittest.TestCase):
    def test_parse_rttm(self):
        segments = vtc.parse_rttm(RTTM)
        self.assertEqual(
            segments, [Segment(0.1, 0.30000000000000004, "", "KCHI"), Segment(0.5, 1.5, "", "FEM")]
        )
        self.assertEqual(vtc.parse_rttm(""), [])

    def test_postprocess_frames_and_segment_means(self):
        n = 100
        probs = {label: np.zeros(n) for label in vtc.LABELS}
        probs["FEM"][25:75] = 0.8
        probs["KCHI"][5:15] = 0.6
        frames, segments = vtc.postprocess(probs, RTTM)
        self.assertEqual(frames.names, vtc.LABELS)
        self.assertEqual(frames.values.shape, (n, 4))
        self.assertAlmostEqual(frames.times[0], 0.01)
        self.assertAlmostEqual(segments.values[0, 0], 0.6)
        self.assertAlmostEqual(segments.values[1, 0], 0.8)
        self.assertEqual([s.label for s in segments.segments], ["KCHI", "FEM"])

    def test_postprocess_errors_and_unknown_label(self):
        with self.assertRaises(ValueError):
            vtc.postprocess({"KCHI": np.zeros(3)}, "")
        bad = {label: np.zeros(3) for label in vtc.LABELS}
        bad["FEM"] = np.zeros(4)
        with self.assertRaises(ValueError):
            vtc.postprocess(bad, "")
        probs = {label: np.zeros(10) for label in vtc.LABELS}
        _, segments = vtc.postprocess(probs, "SPEAKER a 1 5.0 1.0 <NA> <NA> SPEECH <NA> <NA>\n")
        self.assertTrue(np.isnan(segments.values[0, 0]))


class TestAddressee(unittest.TestCase):
    def test_split_long_segments(self):
        parts = addressee.split_long_segments(
            (Segment(0, 70, "mother", "FEM"), Segment(80, 90, "", "MAL"))
        )
        self.assertEqual(len(parts), 4)
        self.assertTrue(all(p.end_s - p.start_s <= 30 for p in parts))
        self.assertEqual((parts[0].start_s, parts[2].end_s, parts[2].label), (0.0, 70.0, "FEM"))
        self.assertEqual(addressee.split_long_segments((Segment(0, 30),)), (Segment(0.0, 30.0),))

    def test_postprocess_reorders_and_normalises(self):
        segments = (Segment(0, 1, "", "FEM"), Segment(2, 3, "", "MAL"))
        logits = np.array([[0.0, 5.0, 0.0], [4.0, 0.0, 0.0]])  # model order ADS, KCDS, Other
        scores = addressee.postprocess(segments, logits)
        self.assertEqual(scores.names, ("KCDS", "ADS", "Other"))
        np.testing.assert_allclose(scores.values.sum(axis=1), 1.0)
        self.assertEqual(addressee.labels(scores), ["KCDS", "ADS"])
        with self.assertRaises(ValueError):
            addressee.postprocess(segments, logits[:1])
        empty = addressee.postprocess((), np.zeros((0, 3)))
        self.assertEqual((empty.values.shape, addressee.labels(empty)), ((0, 3), []))


class FakeForwardAddressee(addressee.AddresseeWorker):
    def __init__(self):
        self.seen = []

    def forward(self, request, segments):
        self.seen = segments
        return np.tile([0.0, 3.0, 0.0], (len(segments), 1))


class TestAddresseeInfer(unittest.TestCase):
    def test_only_adult_segments_are_scored(self):
        from pathlib import Path

        from stillface.features.audio.models import InferenceRequest

        worker = FakeForwardAddressee()
        segments = (Segment(0, 40, "", "FEM"), Segment(41, 42, "", "KCHI"))
        result = worker.infer(InferenceRequest("1/play", Path("x.wav"), 50.0, segments))
        scores = result.segments["addressee"]
        self.assertEqual(len(worker.seen), 2)
        self.assertEqual(addressee.labels(scores), ["KCDS", "KCDS"])
        none = worker.infer(
            InferenceRequest("1/play", Path("x.wav"), 50.0, (Segment(0, 1, "", "OCH"),))
        )
        self.assertEqual(none.segments["addressee"].values.shape, (0, 3))


if __name__ == "__main__":
    unittest.main()


class FakeForwardVtc(vtc.VtcWorker):
    def forward(self, wav):
        return {label: np.full(50, 0.3) for label in vtc.LABELS}, RTTM, {"FEM": 0.5}


class TestVtcInfer(unittest.TestCase):
    def test_result_groups(self):
        from pathlib import Path

        from stillface.features.audio.models import InferenceRequest

        result = FakeForwardVtc().infer(InferenceRequest("1/play", Path("x.wav"), 1.0))
        self.assertEqual(result.artefacts["rttm"], RTTM)
        self.assertEqual(result.frames["voice_type"].values.shape, (50, 4))
        self.assertEqual(len(result.segments["voice_type"].segments), 2)
        self.assertEqual(result.meta["applied_thresholds"], {"FEM": 0.5})
