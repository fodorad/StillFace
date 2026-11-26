import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from stillface.config import DatasetPaths
from stillface.features.audio.config import load_audio_config
from stillface.features.audio.decisions import decision_rows, to_html, to_markdown
from stillface.features.audio.fusion import (
    PhaseInputs,
    column,
    dilate,
    fuse_phase,
    fusion_paths,
    raster,
    run_fusion,
    segment_max,
)
from stillface.features.audio.infer import raw_path
from stillface.features.audio.lanes import Event, read_events, read_lanes, write_events
from stillface.features.audio.media import write_wav
from stillface.features.audio.models import (
    FrameScores,
    InferenceResult,
    Segment,
    SegmentScores,
    save_result,
)
from stillface.features.audio.prepare import wav_path
from stillface.features.audio.roles import role_events_path

_DEFAULT = load_audio_config()
CONFIG = replace(
    _DEFAULT,
    fusion=replace(_DEFAULT.fusion, gong=replace(_DEFAULT.fusion.gong, edge_s=2.0)),
    # One disabled model, to test that disabled models are neither read nor required.
    models={
        **_DEFAULT.models,
        "littlebeats": replace(_DEFAULT.models["littlebeats"], enabled=False),
    },
)
DURATION = 10.0


def frames(rate, names, fn):
    times = np.arange(0, DURATION, 1 / rate) + 0.5 / rate
    values = np.array([[fn(t, n) for n in names] for t in times])
    return FrameScores(times, tuple(names), values)


def ast_names():
    fc = CONFIG.fusion
    names = [*fc.gong.classes, *fc.context.classes]
    for rule in [*fc.baby_types.values(), *fc.mother_types.values()]:
        names += list(rule.ast)
    return list(dict.fromkeys(names))


def make_inputs(lb_label="CRY", singing=0.0):
    def vtc_fn(t, n):
        return 0.9 if (n == "KCHI" and 2 <= t < 4) or (n == "FEM" and 5 <= t < 8) else 0.05

    def ast_fn(t, n):
        if n == "Gong" and (t < 1 or 5 <= t < 5.5):
            return 0.9
        if n == "Baby cry, infant cry" and 2 <= t < 4:
            return 0.6
        if n == "Singing" and 5 <= t < 8:
            return singing
        return 0.5 if n == "Music" else 0.0

    vtc = InferenceResult(
        "1/play",
        {"voice_type": frames(50, ["KCHI", "OCH", "MAL", "FEM"], vtc_fn)},
        {
            "voice_type": SegmentScores(
                (Segment(2, 4, "", "KCHI"), Segment(5, 8, "", "FEM")),
                ("mean_prob",),
                [[0.9], [0.9]],
            )
        },
        meta={"applied_thresholds": {"KCHI": 0.5, "FEM": "x"}},
    )
    silero = InferenceResult(
        "1/play", {"speech": frames(31.25, ["speech"], lambda t, n: 0.9 if 5 <= t < 8 else 0.1)}
    )
    pyannote = InferenceResult(
        "1/play", {"speech": frames(50, ["speech"], lambda t, n: 0.9 if 2 <= t < 8 else 0.1)}
    )
    ast = InferenceResult("1/play", {"audioset": frames(4, ast_names(), ast_fn)})
    lb = InferenceResult(
        "1/play",
        {
            "infant": frames(
                1, ["CRY", "FUS", "BAB"], lambda t, n: 0.7 if n == lb_label and 2 <= t < 4 else 0.1
            )
        },
    )
    addressee = InferenceResult(
        "1/play",
        segments={
            "addressee": SegmentScores(
                (Segment(5, 8, "", "FEM"),), ("KCDS", "ADS", "Other"), [[0.1, 0.8, 0.1]]
            )
        },
    )
    prosody = InferenceResult(
        "1/play",
        {
            "f0_adult": frames(100, ["f0_hz"], lambda t, n: 220.0),
            "f0_infant": frames(100, ["f0_hz"], lambda t, n: 400.0),
            "intensity": frames(100, ["intensity_db"], lambda t, n: 60.0),
            "hnr": frames(100, ["hnr_db"], lambda t, n: 10.0),
        },
    )
    roles = [
        Event(2, 4, "baby", "baby", 1.0, "roles", None, "ROLE-BABY"),
        Event(5, 8, "mother", "FEM", 0.9, "roles", 0.6, "ROLE-FEM-MOTHER"),
    ]
    raw = {
        "vtc": vtc,
        "silero": silero,
        "pyannote": pyannote,
        "ast": ast,
        "littlebeats": lb,
        "prosody": prosody,
        "addressee": addressee,
    }
    return PhaseInputs("1", "play", DURATION, raw, roles)


def at(table, name, t):
    index = int(round(t * 10))
    for lane in table.lanes:
        if lane.spec.name == name:
            return lane.values[index]
    return bool(table.band(name).values[index])


class TestHelpers(unittest.TestCase):
    def test_raster_dilate_column_segment_max(self):
        times = np.arange(10) / 10
        np.testing.assert_array_equal(raster([(0.2, 0.4)], times).nonzero()[0], [2, 3])
        np.testing.assert_array_equal(dilate(np.array([0, 0, 1, 0, 0], bool), 1), [0, 1, 1, 1, 0])
        f = FrameScores([0.1, 0.5, 0.9], ("a", "b"), [[0.1, 0.2], [0.3, 0.9], [0.5, 0.0]])
        np.testing.assert_allclose(column(f, ["a", "b"]), [0.2, 0.9, 0.5])
        self.assertTrue(np.isnan(column(f, [])).all())
        with self.assertRaises(KeyError):
            column(f, ["zzz"])
        self.assertEqual(segment_max(f, column(f, ["a"]), 0.4, 1.0), 0.5)
        self.assertEqual(segment_max(f, column(f, ["a"]), 0.6, 0.7), 0.3)
        empty = FrameScores([], ("a",), np.zeros((0, 1)))
        self.assertTrue(np.isnan(segment_max(empty, np.zeros(0), 0, 1)))
        self.assertTrue(np.isnan(segment_max(f, np.full(3, np.nan), 0, 1)))


class TestFusePhase(unittest.TestCase):
    def test_vad_silence_gong_and_disagreement(self):
        result = fuse_phase(make_inputs(), CONFIG)
        table = result.table
        self.assertTrue(at(table, "fusion.gong.context", 0.5))
        self.assertFalse(at(table, "fusion.gong.context", 5.2))
        self.assertTrue(at(table, "fusion.vad.dyad", 3.0))
        self.assertTrue(at(table, "fusion.silence.dyad", 9.5))
        self.assertFalse(at(table, "fusion.silence.dyad", 0.5))
        self.assertTrue(at(table, "fusion.vad_disagree.dyad", 3.0))
        self.assertFalse(at(table, "fusion.vad_disagree.dyad", 6.0))
        self.assertEqual(table.band("vtc.KCHI.baby.on").spec.threshold, 0.5)
        self.assertIsNone(table.band("vtc.FEM.adult.on").spec.threshold)
        kinds = {i.item for i in result.listening}
        self.assertIn("context:Music", kinds)
        self.assertIn("ads_deviation", kinds)
        self.assertEqual(result.summary["context_shares"]["Music"] > 0.8, True)

    def test_every_priority_class_gets_a_lane_up_to_the_configured_tier(self):
        inputs = make_inputs()
        audioset = inputs.raw["ast"].frames["audioset"]
        extra = ("Cough", "Dog")  # tier 2 and tier 0
        times = audioset.times
        values = np.hstack([audioset.values, np.full((len(times), 2), 0.3)])
        frames_ = FrameScores(times, (*audioset.names, *extra), values)
        raw = {**inputs.raw, "ast": InferenceResult("1/play", {"audioset": frames_})}
        inputs = replace(inputs, raw=raw)
        names = [lane.spec.name for lane in fuse_phase(inputs, CONFIG).table.lanes]
        self.assertIn("ast.Cough.context", names)
        self.assertNotIn("ast.Dog.context", names)
        self.assertIn("ast.Gong.context", names)
        targets_only = replace(CONFIG, fusion=replace(CONFIG.fusion, ast_lane_max_tier=1))
        names = [lane.spec.name for lane in fuse_phase(inputs, targets_only).table.lanes]
        self.assertNotIn("ast.Cough.context", names)
        self.assertIn("ast.Gong.context", names)  # still needed by the gong rule

    def test_types_addressee_and_prosody(self):
        result = fuse_phase(make_inputs(), CONFIG)
        baby = [e for e in result.events if e.role == "baby"]
        self.assertEqual(
            (baby[0].label, baby[0].rule_id, baby[0].confidence), ("cry", "BABY-TYPE-VOTE", 1.0)
        )
        mother = [e for e in result.events if e.rule_id == "MOTHER-TYPE"]
        self.assertEqual(mother[0].label, "speech")
        self.assertIn("ads_deviation", [e.label for e in result.events])
        table = result.table
        self.assertGreater(at(table, "fusion.p_cry.baby", 3.0), 0.6)
        self.assertTrue(np.isnan(at(table, "fusion.p_cry.baby", 6.0)))
        self.assertEqual(at(table, "fusion.type.baby", 3.0), 0.0)
        self.assertAlmostEqual(at(table, "prosody.f0.mother", 6.0), 220.0)
        self.assertTrue(np.isnan(at(table, "prosody.f0.mother", 3.0)))
        self.assertAlmostEqual(at(table, "addressee.ADS.adult", 6.0), 0.8)
        self.assertTrue(at(table, "fusion.overlap.dyad", 3.0) is False)

    def test_conflict_none_singing_and_missing_models(self):
        conflict = fuse_phase(make_inputs(lb_label="FUS"), CONFIG)
        self.assertEqual([e.label for e in conflict.events if e.role == "baby"], ["conflict"])
        inputs = make_inputs(singing=0.7)
        inputs.raw["littlebeats"] = None
        inputs.raw["ast"] = None
        result = fuse_phase(inputs, CONFIG)
        self.assertEqual([e.label for e in result.events if e.role == "baby"], ["other_vocal"])
        self.assertEqual(result.summary["baby_type_models"], [])
        sung = fuse_phase(make_inputs(singing=0.7), CONFIG)
        self.assertEqual([e.label for e in sung.events if e.rule_id == "MOTHER-TYPE"], ["singing"])
        bare = PhaseInputs("1", "play", DURATION, {}, [])
        self.assertTrue(fuse_phase(bare, CONFIG).table.band("fusion.silence.dyad").values.all())

    def test_every_rule_id_is_in_the_decision_tables(self):
        result = fuse_phase(make_inputs(), CONFIG)
        ids = {r.rule_id for r in decision_rows(CONFIG)}
        used = {b.spec.rule_id for b in result.table.bands} | {e.rule_id for e in result.events}
        self.assertEqual(used - ids, set())
        markdown, html = to_markdown(decision_rows(CONFIG)), to_html(decision_rows(CONFIG))
        self.assertIn("| GONG |", markdown)
        self.assertIn("<td>GONG</td>", html)
        self.assertIn("0.2", markdown)
        self.assertIn("TYPE-BABY-CRY", ids)


class TestRunFusion(unittest.TestCase):
    def test_writes_outputs_from_stored_raw_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = DatasetPaths(Path(tmp) / "raw", Path(tmp) / "out")
            write_wav(wav_path(paths, "1", "play"), np.zeros(int(DURATION * 16000)), 16000)
            inputs = make_inputs()
            names = {
                "vtc": "vtc",
                "silero": "silero",
                "pyannote": "pyannote_vad",
                "ast": "ast",
                "prosody": "prosody",
                "addressee": "addressee",
            }
            for key, model in names.items():
                out = raw_path(paths, model, "1/play")
                out.parent.mkdir(parents=True, exist_ok=True)
                save_result(inputs.raw[key], out)
            write_events(inputs.roles, role_events_path(paths, "1", "play"))
            items = run_fusion(paths, "1", CONFIG, ("play", "reunion"))
            out = fusion_paths(paths, "1", "play")
            table = read_lanes(out["lanes"], out["legend"])
            events = read_events(out["events"])
            summary = json.loads(out["summary"].read_text())
        self.assertIn("fusion.vad.dyad", [b.spec.name for b in table.bands])
        self.assertEqual(table.notes["config"], CONFIG.digest)
        self.assertTrue(any(e.rule_id == "BABY-TYPE-VOTE" for e in events))
        self.assertFalse(summary["models"]["littlebeats"])
        self.assertTrue(items)

    def test_disabled_models_are_not_read(self):
        config = replace(CONFIG, models={**CONFIG.models})
        self.assertFalse(config.models["littlebeats"].enabled)


if __name__ == "__main__":
    unittest.main()
