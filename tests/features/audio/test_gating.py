import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stillface.cli import main
from stillface.config import DatasetPaths
from stillface.features.audio.config import AudioConfig
from stillface.features.audio.gating import gate, intersect, pad_intervals, run_gating
from stillface.features.audio.lanes import Event, write_events
from stillface.features.audio.media import read_wav, write_wav
from stillface.features.audio.prepare import wav_path
from stillface.features.audio.provenance import read_sidecar
from stillface.features.audio.roles import role_events_path


class TestHelpers(unittest.TestCase):
    def test_pad_and_merge(self):
        self.assertEqual(
            pad_intervals([(1.0, 2.0), (2.5, 3.0), (0.1, 0.2)], 0.3, 3.1), [(0.0, 0.5), (0.7, 3.1)]
        )

    def test_gate_keeps_inside_zeroes_outside(self):
        x = np.arange(1, 11, dtype=np.float32)
        y = gate(x, 10, [(0.2, 0.5)])
        np.testing.assert_array_equal(y, [0, 0, 3, 4, 5, 0, 0, 0, 0, 0])
        self.assertEqual(len(gate(x, 10, [(0.9, 5.0)])), 10)

    def test_intersect(self):
        self.assertEqual(intersect([(0, 2), (5, 6)], [(1, 5.5)]), [(1, 2), (5, 5.5)])
        self.assertEqual(intersect([(0, 1)], [(1, 2)]), [])


class TestRunGating(unittest.TestCase):
    def test_tracks_sidecars_and_overlaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = DatasetPaths(Path(tmp) / "raw", Path(tmp) / "out")
            rng = np.random.default_rng(0)
            signal = rng.uniform(-0.5, 0.5, 16000 * 4)
            write_wav(wav_path(paths, "1", "play"), signal, 16000)
            events = [
                Event(0.5, 1.5, "mother", "FEM", 1.0, "roles", 0.6, "ROLE-FEM-MOTHER"),
                Event(1.0, 2.0, "baby", "baby", 1.0, "roles", None, "ROLE-BABY"),
                Event(3.0, 3.5, "other", "MAL", 1.0, "roles", None, "ROLE-MAL"),
            ]
            write_events(events, role_events_path(paths, "1", "play"))
            overlaps = run_gating(paths, "1", AudioConfig(), ("play", "reunion"))
            mother, rate = read_wav(wav_path(paths, "1", "play", "mother"))
            baby, _ = read_wav(wav_path(paths, "1", "play", "baby"))
            original, _ = read_wav(wav_path(paths, "1", "play"))
            side = read_sidecar(wav_path(paths, "1", "play", "mother"))
            stored = json.loads((paths.audio_session_dir("1") / "gating" / "play.json").read_text())
            self.assertEqual(
                main(
                    [
                        "--raw-dir",
                        tmp,
                        "--processed-dir",
                        str(Path(tmp) / "out"),
                        "audio",
                        "gate",
                        "--session-id",
                        "1",
                    ]
                ),
                0,
            )
        self.assertEqual(len(mother), len(original))
        np.testing.assert_array_equal(
            mother[int(0.3 * rate) : int(1.7 * rate)], original[int(0.3 * rate) : int(1.7 * rate)]
        )
        self.assertTrue((mother[int(1.9 * rate) :] == 0).all())
        self.assertTrue((baby[: int(0.6 * rate)] == 0).all())
        self.assertEqual(overlaps, {"play": [(1.0, 1.5)]})
        self.assertEqual(stored["overlaps"], [[1.0, 1.5]])
        self.assertEqual(
            (side["track"], side["pad_s"], side["intervals"]), ("mother", 0.3, [[0.2, 1.8]])
        )


if __name__ == "__main__":
    unittest.main()
