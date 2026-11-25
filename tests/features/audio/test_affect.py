import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stillface.cli import main
from stillface.config import DatasetPaths
from stillface.features.audio.affect import (
    baby_valence,
    e2v_dimensions,
    minmax,
    run_affect,
    spearman,
    window_lane,
)
from stillface.features.audio.config import AudioConfig
from stillface.features.audio.fusion import fusion_paths
from stillface.features.audio.infer import raw_path
from stillface.features.audio.lanes import (
    Aggregation,
    Lane,
    LaneSpec,
    LaneTable,
    read_lanes,
    write_lanes,
)
from stillface.features.audio.models import InferenceResult, Segment, SegmentScores, save_result
from stillface.features.audio.workers.emotion2vec import CLASSES

TIMES = np.arange(51) / 10


def e2v_scores(happy, neutral):
    row = np.zeros(len(CLASSES))
    row[CLASSES.index("happy")], row[CLASSES.index("neutral")] = happy, neutral
    row[CLASSES.index("sad")] = 1 - happy - neutral
    return row


class TestHelpers(unittest.TestCase):
    def test_window_lane_means_overlaps_by_role(self):
        windows = (Segment(0, 3, "mother"), Segment(1, 4, "mother"), Segment(0, 5, "baby"))
        scores = SegmentScores(windows, ("arousal",), [[0.2], [0.6], [0.9]])
        lane = window_lane(scores, "arousal", "mother", TIMES)
        self.assertAlmostEqual(lane[5], 0.2)
        self.assertAlmostEqual(lane[20], 0.4)
        self.assertAlmostEqual(lane[35], 0.6)
        self.assertTrue(np.isnan(lane[45]))

    def test_e2v_dimensions(self):
        scores = SegmentScores((Segment(0, 5, "mother"),), CLASSES, [e2v_scores(0.6, 0.3)])
        valence, arousal = e2v_dimensions(scores, "mother", TIMES)
        self.assertAlmostEqual(valence[10], 0.5)
        self.assertAlmostEqual(arousal[10], 0.7)

    def test_spearman_baby_valence_minmax(self):
        a = np.arange(30, dtype=float)
        self.assertAlmostEqual(spearman(a, a**2, 20), 1.0)
        self.assertTrue(np.isnan(spearman(a[:5], a[:5], 20)))
        self.assertTrue(np.isnan(spearman(a, np.ones(30), 20)))
        p = {
            "cry": np.array([0.8, np.nan, 0.0]),
            "fuss": np.array([0.0, np.nan, 0.0]),
            "laugh": np.array([0.0, np.nan, 0.9]),
        }
        np.testing.assert_allclose(
            baby_valence(p, {"cry": -1.0, "fuss": -0.5, "laugh": 1.0}), [-0.8, np.nan, 0.9]
        )
        with self.assertRaises(ValueError):
            baby_valence({}, {"cry": -1.0})
        np.testing.assert_allclose(minmax(np.array([2.0, np.nan, 4.0])), [0.0, np.nan, 1.0])
        np.testing.assert_allclose(minmax(np.array([3.0, np.nan])), [0.5, np.nan])
        self.assertTrue(np.isnan(minmax(np.array([np.nan]))).all())


class TestRunAffect(unittest.TestCase):
    def test_lanes_and_agreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = DatasetPaths(Path(tmp) / "raw", Path(tmp) / "out")
            baby = (TIMES >= 1) & (TIMES < 3)

            def prob(name):
                return LaneSpec("fusion", f"p_{name}", "baby", "prob", (0.0, 1.0))

            lanes = [
                Lane(prob("cry"), np.where(baby, np.linspace(0, 1, 51), np.nan)),
                Lane(prob("fuss"), np.where(baby, 0.1, np.nan)),
                Lane(prob("laugh"), np.where(baby, 0.0, np.nan)),
                Lane(
                    LaneSpec(
                        "prosody", "intensity_db", "dyad", "dB", (-300, 300), Aggregation.MEAN
                    ),
                    np.linspace(40, 80, 51),
                ),
                Lane(
                    LaneSpec("prosody", "f0", "baby", "Hz", (0, 2000), Aggregation.MEAN),
                    np.where(baby, 400.0, np.nan),
                ),
            ]
            out = fusion_paths(paths, "1", "play")
            write_lanes(LaneTable(5.0, lanes), out["lanes"], out["legend"])
            windows = tuple(Segment(t, t + 1, "mother") for t in np.arange(0, 4.5, 0.5)) + (
                Segment(1, 2, "baby"),
            )
            n = len(windows)
            values = np.column_stack(
                [np.linspace(0.1, 0.9, n), np.full(n, 0.5), np.linspace(0.2, 0.8, n)]
            )
            for model, group in (
                ("audeering", SegmentScores(windows, ("arousal", "dominance", "valence"), values)),
            ):
                path = raw_path(paths, model, "1/play")
                path.parent.mkdir(parents=True, exist_ok=True)
                save_result(InferenceResult("1/play", segments={"windows": group}), path)
            e2v_rows = [e2v_scores(0.1 + 0.08 * i, 0.9 - 0.09 * i) for i in range(n)]
            path = raw_path(paths, "emotion2vec", "1/play")
            path.parent.mkdir(parents=True, exist_ok=True)
            save_result(
                InferenceResult(
                    "1/play", segments={"windows": SegmentScores(windows, CLASSES, e2v_rows)}
                ),
                path,
            )
            summary = run_affect(paths, "1", AudioConfig(), ("play", "reunion"))
            table_path = paths.audio_session_dir("1") / "affect" / "play.csv"
            table = read_lanes(table_path, table_path.with_suffix(".json"))
            stored = json.loads(
                (paths.audio_session_dir("1") / "affect" / "summary.json").read_text()
            )
            code = main(
                [
                    "--raw-dir",
                    tmp,
                    "--processed-dir",
                    str(Path(tmp) / "out"),
                    "audio",
                    "affect",
                    "--session-id",
                    "1",
                ]
            )
        self.assertEqual(code, 0)
        self.assertGreater(summary["agreement"]["arousal"], 0.9)
        self.assertEqual(stored["flags"], [])
        valence = table.lane("affect.valence.baby").values
        self.assertTrue(np.isnan(valence[0]))
        self.assertAlmostEqual(valence[10], -(0.2 + 0.05), places=5)
        arousal = table.lane("affect.arousal.baby").values
        self.assertTrue(np.isfinite(arousal[15]) and np.isnan(arousal[40]))
        self.assertIn("experimental", table.lane("audeering.valence.baby").spec.note)
        self.assertTrue(np.isfinite(table.lane("emotion2vec.valence.mother").values[10]))

    def test_no_models_flags_missing_agreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = DatasetPaths(Path(tmp) / "raw", Path(tmp) / "out")
            summary = run_affect(paths, "1", AudioConfig(), ("play",))
        self.assertEqual(summary["flags"], ["no_arousal_agreement", "no_valence_agreement"])


if __name__ == "__main__":
    unittest.main()
