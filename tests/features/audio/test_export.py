import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stillface.cli import main
from stillface.config import DatasetPaths
from stillface.features.audio.export import (
    BAND_TRACKS,
    build_annie_tree,
    export_phase,
    qc_row,
    read_listening,
    session_qc,
    unique_items,
    video_id,
    write_dataset_tables,
)
from stillface.features.audio.fusion import fuse_phase, run_fusion
from stillface.features.audio.infer import raw_path
from stillface.features.audio.lanes import read_lanes, write_events
from stillface.features.audio.media import write_wav
from stillface.features.audio.models import save_result
from stillface.features.audio.prepare import wav_path
from stillface.features.audio.provenance import write_sidecar
from stillface.features.audio.roles import ListeningItem, role_events_path, write_listening
from tests.features.audio.test_fusion import CONFIG, DURATION, make_inputs

NAMES = {
    "vtc": "vtc",
    "silero": "silero",
    "pyannote": "pyannote_vad",
    "ast": "ast",
    "prosody": "prosody",
    "addressee": "addressee",
}


def fused_session(paths):
    wav = wav_path(paths, "1", "play")
    write_wav(wav, np.zeros(int(DURATION * 16000)), 16000)
    write_sidecar(wav, {"camera": "window", "duration_s": DURATION})
    inputs = make_inputs()
    for key, model in NAMES.items():
        out = raw_path(paths, model, "1/play")
        out.parent.mkdir(parents=True, exist_ok=True)
        save_result(inputs.raw[key], out)
    write_events(inputs.roles, role_events_path(paths, "1", "play"))
    return run_fusion(paths, "1", CONFIG, ("play",))


class TestAnnieTree(unittest.TestCase):
    def test_tree_layout_and_band_events(self):
        inputs = make_inputs()
        result = fuse_phase(inputs, CONFIG)
        tree = build_annie_tree("1", "play", "baby", inputs.roles + result.events, result.table)
        node = tree[video_id("1", "baby", "play")]
        self.assertEqual(node["fps"], 60.0)
        self.assertIn("baby:fusion", node["tracks"])
        self.assertIn("dyad:vad", node["tracks"])
        cry = node["tracks"]["baby:fusion"][0]
        self.assertEqual((cry["label"], cry["start_frame"], cry["end_frame"]), ("cry", 120, 240))
        self.assertEqual(cry["attributes"]["rule_id"], "BABY-TYPE-VOTE")
        self.assertEqual([r["event_id"] for r in node["tracks"]["dyad:vad"]][:2], [1, 2])
        self.assertTrue(set(BAND_TRACKS) <= {b.spec.name for b in result.table.bands})


class TestExportAndQc(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.paths = DatasetPaths(Path(self.tmp.name) / "raw", Path(self.tmp.name) / "out")

    def tearDown(self):
        self.tmp.cleanup()

    def test_export_files(self):
        fused_session(self.paths)
        target = export_phase(self.paths, "1", "play")
        self.assertIsNone(export_phase(self.paths, "1", "reunion"))
        tree = json.loads(target.read_text())
        self.assertEqual(list(tree), ["1_window_play"])
        folder = target.parent
        lanes = read_lanes(folder / "1_play_lanes.csv", folder / "1_play_lanes.json")
        self.assertIn("fusion.p_cry.baby", [lane.spec.name for lane in lanes.lanes])

    def test_qc_record_summary_and_listening(self):
        items = fused_session(self.paths)
        write_listening(items + items[:1], self.paths.audio_dir / "listening_fusion.csv")
        write_listening(
            [ListeningItem("1", "", "no_enrolment")], self.paths.audio_dir / "listening_roles.csv"
        )
        record = session_qc(self.paths, "1", CONFIG)
        self.assertEqual(record["prepared"], ["play"])
        self.assertEqual(record["models"]["littlebeats"], "disabled")
        self.assertEqual(record["models"]["vtc"]["done"], ["play"])
        self.assertEqual(record["models"]["egemaps"]["missing"], ["play"])
        self.assertIn("enrolment:not_run", record["flags"])
        play = record["phases"]["play"]
        self.assertGreater(play["voice_share"], 0.3)
        self.assertGreater(play["context_shares"]["Music"], 0.5)
        row = qc_row(record)
        self.assertEqual(row["prepared_phases"], 1)
        self.assertGreater(row["models_incomplete"], 0)
        write_dataset_tables(self.paths, [record])
        with (self.paths.audio_dir / "qc_summary.csv").open() as handle:
            self.assertEqual(next(csv.DictReader(handle))["session"], "1")
        merged = read_listening(self.paths.audio_dir / "listening.csv")
        self.assertEqual(len(merged), len(items) + 1)
        self.assertEqual(merged[0].item, "no_enrolment")
        self.assertEqual(read_listening(self.paths.audio_dir / "nope.csv"), [])

    def test_unique_items_and_cli(self):
        a = ListeningItem("1", "play", "x", 1.0, 2.0)
        b = ListeningItem("1", "", "y")
        self.assertEqual(unique_items([a, b, a]), [b, a])
        self.assertTrue(math.isnan(b.start_s))
        fused_session(self.paths)
        base = [
            "--raw-dir",
            self.tmp.name,
            "--processed-dir",
            str(self.paths.processed_dir),
            "audio",
        ]
        self.assertEqual(main([*base, "export", "--session-id", "1"]), 0)
        self.assertEqual(main([*base, "qc", "--session-id", "1"]), 0)
        self.assertTrue((self.paths.audio_session_dir("1") / "qc.json").exists())


if __name__ == "__main__":
    unittest.main()
