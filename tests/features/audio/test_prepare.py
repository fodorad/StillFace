import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from stillface.config import DatasetPaths
from stillface.features.audio.config import AudioConfig
from stillface.features.audio.inventory import build_inventory
from stillface.features.audio.media import MediaError, read_wav, write_wav
from stillface.features.audio.prepare import prepare_all, prepare_phase, wav_path
from stillface.features.audio.provenance import (
    file_sha256,
    is_up_to_date,
    read_sidecar,
    sidecar_path,
    write_sidecar,
)
from tests.helpers import make_audio


class TestWav(unittest.TestCase):
    def test_round_trip_and_clipping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.wav"
            x = np.array([0.0, 0.5, -0.5, 2.0])
            write_wav(path, x, 16000)
            y, rate = read_wav(path)
            self.assertEqual(rate, 16000)
            np.testing.assert_allclose(y, [0.0, 0.5, -0.5, 1.0], atol=1e-4)

    def test_rejects_stereo(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_audio(Path(tmp) / "s.wav", seconds=0.1, channels=2)
            with self.assertRaises(MediaError):
                read_wav(path)


class TestPrepare(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.paths = DatasetPaths(root / "raw", root / "out")
        self.config = AudioConfig()

    def tearDown(self):
        self.tmp.cleanup()

    def inventory(self):
        meta = pd.DataFrame({"ID": [1, 2], "Analysis": [1, 0]})
        return build_inventory(self.paths, self.config, meta)

    def test_resamples_downmixes_and_keeps_level(self):
        make_audio(
            self.paths.phase_clip("1", "baby", "play"),
            "0.05*sin(2*PI*300*t)",
            seconds=1.0,
            rate=48000,
            channels=2,
            video=True,
        )
        rows = [r for r in self.inventory() if r.phase == "play" and r.session_id == "1"]
        self.assertTrue(prepare_phase(self.paths, rows[0], self.config))
        out = wav_path(self.paths, "1", "play")
        y, rate = read_wav(out)
        self.assertEqual(rate, 16000)
        self.assertAlmostEqual(len(y) / rate, rows[0].audio_s, delta=1 / 16000 + 0.03)
        freq = np.argmax(np.abs(np.fft.rfft(y))) * rate / len(y)
        self.assertAlmostEqual(freq, 300, delta=3)
        self.assertAlmostEqual(float(np.abs(y[2000:-2000]).max()), 0.05, delta=0.01)
        side = read_sidecar(out)
        self.assertEqual(side["camera"], "baby")
        self.assertEqual(side["video_fps"], 10.0)
        self.assertEqual(side["source"], "Sessions/1/processed/baby_play.mp4")
        self.assertEqual(
            side["source_sha256"], file_sha256(self.paths.phase_clip("1", "baby", "play"))
        )
        self.assertFalse(prepare_phase(self.paths, rows[0], self.config))
        self.assertTrue(prepare_phase(self.paths, rows[0], self.config, overwrite=True))

    def test_exact_length_from_wav_source(self):
        clip = self.paths.phase_clip("1", "baby", "play")
        clip.parent.mkdir(parents=True)
        make_audio(clip.with_suffix(".wav"), seconds=0.75, rate=48000).rename(clip)
        row = [r for r in self.inventory() if r.session_id == "1" and r.phase == "play"][0]
        prepare_phase(self.paths, row, self.config)
        y, _ = read_wav(wav_path(self.paths, "1", "play"))
        self.assertLessEqual(abs(len(y) - 12000), 1)

    def test_prepare_all_scope_failures_and_unusable(self):
        make_audio(self.paths.phase_clip("1", "baby", "play"), seconds=0.3, video=True)
        make_audio(self.paths.phase_clip("1", "baby", "reunion"), seconds=0.3, video=True)
        make_audio(self.paths.phase_clip("2", "baby", "play"), seconds=0.3, video=True)
        rows = self.inventory()
        broken = [r for r in rows if r.session_id == "1" and r.phase == "reunion"][0]
        self.paths.phase_clip("1", "baby", "reunion").write_bytes(b"gone")
        report = prepare_all(self.paths, rows, self.config)
        self.assertEqual(report.written, ["1/play"])
        self.assertIn("1/reunion", report.failed)
        self.assertEqual(prepare_all(self.paths, rows, self.config, sessions={"9"}).written, [])
        self.assertEqual(prepare_all(self.paths, rows, self.config).kept, ["1/play"])
        with self.assertRaises(ValueError):
            prepare_phase(self.paths, replace(broken, status="missing"), self.config)

    def test_wav_path_tracks(self):
        self.assertEqual(wav_path(self.paths, "1", "play", "mother").name, "play_mother.wav")


class TestProvenance(unittest.TestCase):
    def test_up_to_date_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "x.npz"
            self.assertFalse(is_up_to_date(out, {}))
            out.write_text("data")
            self.assertFalse(is_up_to_date(out, {"a": 1}))
            write_sidecar(out, {"a": 1, "b": "v"})
            self.assertEqual(sidecar_path(out).name, "x.npz.json")
            self.assertTrue(is_up_to_date(out, {"a": 1}))
            self.assertFalse(is_up_to_date(out, {"a": 2}))
            self.assertFalse(is_up_to_date(out, {"a": 1}, overwrite=True))
            sidecar_path(out).write_text("{broken")
            self.assertIsNone(read_sidecar(out))


if __name__ == "__main__":
    unittest.main()
