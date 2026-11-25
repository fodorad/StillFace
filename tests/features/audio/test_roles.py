import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stillface.config import DatasetPaths
from stillface.features.audio.config import AudioConfig, ModelConfig, RolesConfig
from stillface.features.audio.infer import raw_path
from stillface.features.audio.lanes import read_events
from stillface.features.audio.media import write_wav
from stillface.features.audio.models import InferenceResult, Segment, SegmentScores, save_result
from stillface.features.audio.prepare import wav_path
from stillface.features.audio.roles import (
    RULES,
    ListeningItem,
    assign_roles,
    classify_female,
    enrol,
    merge_intervals,
    overlaps,
    role_events_path,
    role_segments,
    segment_builder,
    write_listening,
)

MOTHER, OTHER, UNSURE = np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5, 0.866])


def store(paths, model, key, segments, group, embeddings=None):
    out = raw_path(paths, model, key)
    out.parent.mkdir(parents=True, exist_ok=True)
    scores = SegmentScores(tuple(segments), ("v",), np.ones((len(segments), 1)), embeddings)
    save_result(InferenceResult(key, segments={group: scores}), out)


class TestHelpers(unittest.TestCase):
    def test_merge_and_overlap(self):
        merged = merge_intervals([Segment(2, 3), Segment(0, 1), Segment(0.5, 1.5), Segment(3, 4)])
        self.assertEqual(merged, [(0, 1.5), (2, 4)])
        self.assertTrue(overlaps(Segment(1, 2), [(1.5, 3)]))
        self.assertFalse(overlaps(Segment(1, 2), [(2, 3)]))

    def test_classify_female(self):
        rc = RolesConfig()
        self.assertEqual(classify_female(0.9, True, rc)[:3], ("mother", "FEM", "ROLE-FEM-MOTHER"))
        self.assertEqual(classify_female(0.1, True, rc)[2], "ROLE-FEM-OTHER")
        self.assertEqual(classify_female(0.5, True, rc)[2], "ROLE-FEM-UNCERTAIN")
        self.assertEqual(classify_female(None, True, rc)[2], "ROLE-FEM-UNCERTAIN")
        self.assertEqual(classify_female(0.9, False, rc)[2], "ROLE-FEM-UNRESOLVED")
        self.assertTrue(set(RULES) >= {"ROLE-BABY", "ROLE-MAL"})

    def test_enrol_cap_overlap_and_fallback(self):
        rc = RolesConfig(enrol_total_s=10.0, enrol_min_s=4.0)
        play = [
            (Segment(0, 6), MOTHER),
            (Segment(10, 16), MOTHER),
            (Segment(20, 26), MOTHER),
            (Segment(30, 35), OTHER),
        ]
        e = enrol({"play": play}, {}, rc)
        self.assertEqual((e.status, e.phase, e.seconds), ("enrolled", "play", 12.0))
        np.testing.assert_allclose(e.centroid, [1.0, 0.0])
        self.assertNotIn((30, 35), e.segments)
        baby = {"play": [(0, 40)]}
        reunion = [(Segment(0, 5), MOTHER)]
        self.assertEqual(enrol({"play": play, "reunion": reunion}, baby, rc).phase, "reunion")
        self.assertEqual(
            enrol({"play": [(Segment(0, 2), MOTHER), (Segment(3, 9), None)]}, {}, rc).status,
            "no_enrolment",
        )


class TestAssignRoles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.paths = DatasetPaths(root / "raw", root / "out")
        for phase in ("play", "stillface"):
            write_wav(wav_path(self.paths, "1", phase), np.zeros(16000 * 30), 16000)

    def tearDown(self):
        self.tmp.cleanup()

    def test_end_to_end(self):
        play = [
            Segment(0, 10, "", "FEM"),
            Segment(12, 14, "", "FEM"),
            Segment(15, 17, "", "FEM"),
            Segment(18, 19, "", "KCHI"),
            Segment(18.5, 20, "", "OCH"),
            Segment(21, 23, "", "MAL"),
            Segment(24, 25, "", "FEM"),
        ]
        store(self.paths, "vtc", "1/play", play, "voice_type")
        fem = [s for s in play if s.label == "FEM"]
        vectors = np.stack([MOTHER, OTHER, UNSURE, MOTHER * 3])
        store(self.paths, "pyannote_embedding", "1/play", fem, "embedding", vectors)
        items = assign_roles(self.paths, "1", AudioConfig(), ("play", "stillface"))
        events = read_events(role_events_path(self.paths, "1", "play"))
        rules = [(e.start_s, e.rule_id) for e in events]
        self.assertIn((0.0, "ROLE-FEM-MOTHER"), rules)
        self.assertIn((12.0, "ROLE-FEM-OTHER"), rules)
        self.assertIn((15.0, "ROLE-FEM-UNCERTAIN"), rules)
        self.assertIn((18.0, "ROLE-BABY"), rules)
        baby = [e for e in events if e.role == "baby"]
        self.assertEqual((len(baby), baby[0].end_s), (1, 20.0))
        kinds = sorted(i.item for i in items)
        self.assertEqual(kinds, ["male_adult", "uncertain_female"])
        enrolment = json.loads(
            (self.paths.audio_session_dir("1") / "roles" / "enrolment.json").read_text()
        )
        self.assertEqual(enrolment["phase"], "play")
        self.assertEqual(read_events(role_events_path(self.paths, "1", "stillface")), [])
        mothers = role_segments(self.paths, "1", "play", ("mother",))
        self.assertEqual([(s.start_s, s.end_s) for s in mothers], [(0.0, 10.0), (24.0, 25.0)])

    def test_unresolved_session(self):
        store(self.paths, "vtc", "1/play", [Segment(0, 2, "", "FEM")], "voice_type")
        items = assign_roles(self.paths, "1", AudioConfig(), ("play",))
        self.assertEqual(items[0].item, "no_enrolment")
        self.assertEqual(
            read_events(role_events_path(self.paths, "1", "play"))[0].label, "FEM_unresolved"
        )

    def test_segment_builders_and_listening_csv(self):
        store(
            self.paths,
            "vtc",
            "1/play",
            [Segment(0, 2, "", "FEM"), Segment(3, 4, "", "KCHI")],
            "voice_type",
        )
        fem = segment_builder("vtc:FEM")(self.paths, "1", "play")
        self.assertEqual([s.label for s in fem.segments], ["FEM"])
        self.assertIsNone(segment_builder("vtc:MAL")(self.paths, "1", "play"))
        self.assertIsNone(segment_builder("vtc:FEM")(self.paths, "1", "reunion"))
        self.assertIsNone(segment_builder("roles:mother")(self.paths, "1", "play"))
        assign_roles(self.paths, "1", AudioConfig(), ("play",))
        baby = segment_builder("roles:baby,mother")(self.paths, "1", "play")
        self.assertEqual([s.role for s in baby.segments], ["baby"])
        with self.assertRaises(ValueError):
            segment_builder("faces:mother")
        with self.assertRaises(ValueError):
            ModelConfig(segment_source="vtc:")
        out = Path(self.tmp.name) / "l.csv"
        write_listening(
            [ListeningItem("1", "", "no_enrolment"), ListeningItem("1", "play", "x", 1.23456, 2.0)],
            out,
        )
        with out.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual((rows[0]["start_s"], rows[1]["start_s"]), ("", "1.235"))


if __name__ == "__main__":
    unittest.main()
