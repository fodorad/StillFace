import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from stillface.config import DatasetPaths
from stillface.constants import PHASES
from stillface.features.audio.config import AudioConfig, InventoryConfig, ScopeConfig
from stillface.features.audio.inventory import (
    analysis_flags,
    build_inventory,
    is_flag_set,
    level_stats,
    read_inventory,
    session_ids,
    write_inventory,
)
from tests.helpers import make_audio, make_video


class TestLevels(unittest.TestCase):
    def test_sine_clipped_and_silence(self):
        cfg = InventoryConfig()
        t = np.arange(16000) / 16000
        sine = 0.5 * np.sin(2 * np.pi * 440 * t)
        stats = level_stats(sine, 16000, cfg)
        self.assertAlmostEqual(stats.rms_dbfs, 20 * math.log10(0.5 / math.sqrt(2)), places=2)
        self.assertAlmostEqual(stats.peak_dbfs, 20 * math.log10(0.5), places=2)
        self.assertEqual((stats.clipped_share, stats.silent_share), (0.0, 0.0))
        clipped = level_stats(np.clip(4 * sine, -1, 1), 16000, cfg)
        self.assertGreater(clipped.clipped_share, 0.3)
        half = np.concatenate([sine[:8000], np.zeros(8000)])
        self.assertAlmostEqual(level_stats(half, 16000, cfg).silent_share, 0.5, delta=0.05)
        silent = level_stats(np.zeros(100), 16000, cfg)
        self.assertEqual((silent.rms_dbfs, silent.silent_share), (-math.inf, 1.0))
        self.assertEqual(level_stats(np.zeros(0), 16000, cfg).silent_share, 0.0)


class TestFlags(unittest.TestCase):
    def test_is_flag_set(self):
        for value in (1, "y", "Igen", "x", 2.0):
            self.assertTrue(is_flag_set(value))
        for value in (0, "", "n", "No", "nem", None, float("nan"), "0"):
            self.assertFalse(is_flag_set(value))

    def test_analysis_flags(self):
        meta = pd.DataFrame({"ID": [101.0, 102], "Analysis": ["y", "n"]})
        self.assertEqual(analysis_flags(meta, "Analysis"), {"101": True, "102": False})
        self.assertEqual(analysis_flags(meta, "Missing"), {})


class TestInventory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.paths = DatasetPaths(root / "raw", root / "out")

    def tearDown(self):
        self.tmp.cleanup()

    def clip(self, sid, camera, phase, **kwargs):
        return self.paths.phase_clip(sid, camera, phase)

    def test_fallback_statuses_and_scope(self):
        p = self.paths
        # session 1: baby clip everywhere -> ok
        for phase in PHASES:
            make_audio(p.phase_clip("1", "baby", phase), seconds=0.5, video=True)
        # session 2: play only from the window camera, baby clip has no audio in stillface,
        # reunion has a broken baby clip, baseline nothing at all
        make_audio(p.phase_clip("2", "window", "play"), seconds=0.5, video=True)
        make_video(p.phase_clip("2", "baby", "stillface"), seconds=0.5, audio=False)
        broken = p.phase_clip("2", "baby", "reunion")
        broken.write_bytes(b"broken")
        (p.raw_dir / "Sessions" / ".hidden").mkdir(parents=True)
        meta = pd.DataFrame({"ID": [1, 2], "Analysis": [1, 1]})
        config = AudioConfig(scope=ScopeConfig(exclude=("2",)))
        rows = {(r.session_id, r.phase): r for r in build_inventory(p, config, meta)}
        self.assertEqual(session_ids(p), ["1", "2"])
        self.assertEqual(len(rows), 8)
        ok = rows[("1", "play")]
        self.assertEqual((ok.status, ok.camera, ok.codec, ok.channels), ("ok", "baby", "aac", 1))
        self.assertTrue(ok.in_scope)
        self.assertLess(ok.duration_diff_s, 0.2)
        self.assertEqual(ok.video_fps, 10.0)
        fb = rows[("2", "play")]
        self.assertEqual((fb.status, fb.camera), ("fallback", "window"))
        self.assertIn("baby: clip missing", fb.reason)
        self.assertFalse(fb.in_scope)
        self.assertTrue(fb.excluded)
        self.assertEqual(rows[("2", "stillface")].status, "no_audio_stream")
        self.assertEqual(rows[("2", "reunion")].status, "decode_error")
        self.assertEqual(rows[("2", "baseline")].status, "missing")

    def test_split_window_session_prefers_the_baby_facing_window_2(self):
        p = self.paths
        make_audio(p.phase_clip("3", "window_1", "play"), seconds=0.3, video=True)
        make_audio(p.phase_clip("3", "window_2", "play"), seconds=0.3, video=True)
        make_audio(p.phase_clip("3", "window_1", "reunion"), seconds=0.3, video=True)
        meta = pd.DataFrame({"ID": [3], "Analysis": ["y"]})
        rows = {(r.phase): r for r in build_inventory(p, AudioConfig(), meta)}
        play, reunion = rows["play"], rows["reunion"]
        self.assertEqual((play.status, play.camera), ("fallback", "window_2"))
        self.assertEqual(play.candidates, "window_2;window_1")
        self.assertIn("baby: clip missing", play.reason)
        self.assertEqual((reunion.status, reunion.camera), ("fallback", "window_1"))

    def test_round_trip_and_bad_header(self):
        make_audio(self.paths.phase_clip("1", "baby", "play"), seconds=0.3, video=True)
        rows = build_inventory(
            self.paths, AudioConfig(), pd.DataFrame({"ID": [1], "Analysis": [1]})
        )
        out = Path(self.tmp.name) / "inv.csv"
        write_inventory(rows, out)
        back = read_inventory(out)
        self.assertEqual(len(back), len(rows))
        for a, b in zip(rows, back, strict=True):
            for key, value in a.__dict__.items():
                other = getattr(b, key)
                if isinstance(value, float) and math.isnan(value):
                    self.assertTrue(math.isnan(other))
                else:
                    self.assertEqual(value, other, key)
        out.write_text("x\n")
        with self.assertRaises(ValueError):
            read_inventory(out)

    def test_no_sessions_folder(self):
        self.assertEqual(session_ids(self.paths), [])


if __name__ == "__main__":
    unittest.main()
