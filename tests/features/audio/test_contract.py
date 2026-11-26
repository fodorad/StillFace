"""Tests of the model output contract checks (no model environments needed)."""

import unittest
from pathlib import Path

import numpy as np

from stillface.features.audio.config import load_audio_config
from stillface.features.audio.contract import (
    CONTRACTS,
    GroupContract,
    ModelContract,
    check_result,
    contract_rows,
)
from stillface.features.audio.models import (
    FrameScores,
    InferenceResult,
    Segment,
    SegmentScores,
)

SPEECH = ModelContract(
    "toy",
    (GroupContract("frames", "speech", ("speech",), (0.0, 1.0), "speech probability"),),
)
WINDOWS = ModelContract(
    "toy_windows",
    (GroupContract("segments", "windows", ("a", "b"), (0.0, 1.0), "window scores"),),
    artefacts=("log",),
)


def frames(times, values, names=("speech",)):
    return FrameScores(np.asarray(times, float), names, np.asarray(values, float))


class TestCheckResult(unittest.TestCase):
    def test_well_formed_result_passes(self):
        result = InferenceResult("k", {"speech": frames([0.5, 1.5], [[0.1], [0.9]])})
        self.assertEqual(check_result(SPEECH, result, 2.0), [])

    def test_missing_group_wrong_names_and_range(self):
        self.assertIn("missing frames/speech", check_result(SPEECH, InferenceResult("k"), 2.0))
        wrong = InferenceResult("k", {"speech": frames([0.5], [[0.1]], ("voice",))})
        self.assertTrue(any("names" in p for p in check_result(SPEECH, wrong, 2.0)))
        high = InferenceResult("k", {"speech": frames([0.5], [[1.5]])})
        self.assertTrue(any("range" in p for p in check_result(SPEECH, high, 2.0)))

    def test_time_axis_must_be_ordered_and_inside_the_input(self):
        backwards = InferenceResult("k", {"speech": frames([1.5, 0.5], [[0.1], [0.2]])})
        self.assertTrue(any("ordered" in p for p in check_result(SPEECH, backwards, 2.0)))
        late = InferenceResult("k", {"speech": frames([0.5, 9.0], [[0.1], [0.2]])})
        self.assertTrue(any("outside" in p for p in check_result(SPEECH, late, 2.0)))

    def test_all_nan_group_is_reported(self):
        empty = InferenceResult("k", {"speech": frames([0.5], [[np.nan]])})
        self.assertTrue(any("no finite" in p for p in check_result(SPEECH, empty, 2.0)))

    def test_segments_and_artefacts(self):
        scores = SegmentScores(
            (Segment(0.0, 1.0, "mother", "FEM"),), ("a", "b"), np.array([[0.2, 0.8]])
        )
        ok = InferenceResult("k", segments={"windows": scores}, artefacts={"log": "x"})
        self.assertEqual(check_result(WINDOWS, ok, 2.0), [])
        no_log = InferenceResult("k", segments={"windows": scores})
        self.assertIn("missing artefact log", check_result(WINDOWS, no_log, 2.0))
        bad = SegmentScores((Segment(1.5, 1.0, "mother", "FEM"),), ("a", "b"), [[0.2, 0.8]])
        flipped = InferenceResult("k", segments={"windows": bad}, artefacts={"log": "x"})
        self.assertTrue(any("ends before" in p for p in check_result(WINDOWS, flipped, 2.0)))

    def test_open_name_sets_check_only_the_count(self):
        contract = ModelContract(
            "toy88", (GroupContract("segments", "egemaps", None, None, "functionals", 88),)
        )
        names = tuple(f"f{i}" for i in range(88))
        scores = SegmentScores((Segment(0, 1, "baby", "KCHI"),), names, np.zeros((1, 88)))
        self.assertEqual(
            check_result(contract, InferenceResult("k", segments={"egemaps": scores}), 1.0), []
        )
        short = SegmentScores((Segment(0, 1, "baby", "KCHI"),), names[:5], np.zeros((1, 5)))
        problems = check_result(contract, InferenceResult("k", segments={"egemaps": short}), 1.0)
        self.assertTrue(any("88" in p for p in problems))


class TestContracts(unittest.TestCase):
    def test_every_configured_model_has_a_contract(self):
        self.assertEqual(set(CONTRACTS), set(load_audio_config().models))

    def test_docs_table_matches_the_contracts(self):
        page = Path(__file__).parents[3] / "docs" / "audio_outputs.md"
        text = page.read_text()
        for row in contract_rows():
            self.assertIn(row, text)


if __name__ == "__main__":
    unittest.main()
