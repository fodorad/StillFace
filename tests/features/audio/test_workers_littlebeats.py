import tempfile
import unittest
from pathlib import Path

import numpy as np

from stillface.cli import main
from stillface.features.audio.config import (
    DEFAULT_CONFIG_PATH,
    AudioConfig,
    ModelConfig,
    load_audio_config,
)
from stillface.features.audio.models import InferenceRequest
from stillface.features.audio.smoke import smoke, synthetic_clip
from stillface.features.audio.workers import littlebeats as lb
from stillface.features.audio.workers.base import fixed_windows


class TestPostprocess(unittest.TestCase):
    def test_heads_softmax_and_names(self):
        rng = np.random.default_rng(0)
        starts = [0.0, 1.0, 2.0]
        out = lb.postprocess(
            starts, 2.0, rng.normal(size=(3, 5)), rng.normal(size=(3, 3)), rng.normal(size=(3, 4))
        )
        self.assertEqual(out["speaker"].names, ("SIL", "CHN", "FAN", "MAN", "CXN"))
        self.assertEqual(out["infant"].names, ("CRY", "FUS", "BAB"))
        self.assertEqual(out["adult"].names, ("CDS", "ADS", "LAU", "SNG"))
        for scores in out.values():
            np.testing.assert_allclose(scores.values.sum(axis=1), 1.0)
            np.testing.assert_allclose(scores.times, [1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            lb.postprocess(starts, 2.0, np.zeros((3, 4)), np.zeros((3, 3)), np.zeros((3, 4)))

    def test_fixed_windows_pad_the_tail(self):
        samples = np.arange(10, dtype=np.float32)
        windows = fixed_windows(samples, 2, [0.0, 3.0], 2.0)
        self.assertEqual(windows.shape, (2, 4))
        np.testing.assert_array_equal(windows[1], [6, 7, 8, 9])
        tail = fixed_windows(samples, 2, [4.0], 2.0)
        np.testing.assert_array_equal(tail[0], [8, 9, 0, 0])


class FakeForward(lb.LittleBeatsWorker):
    def __init__(self):
        self.window_s, self.hop_s = 2.0, 1.0

    def forward(self, windows):
        n = len(windows)
        return np.zeros((n, 5)), np.zeros((n, 3)), np.zeros((n, 4))


class TestInferAndDisabled(unittest.TestCase):
    def test_infer_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = synthetic_clip(Path(tmp) / "c.wav", seconds=4.0)
            result = FakeForward().infer(InferenceRequest("1/play", wav, 4.0))
            self.assertEqual(result.frames["infant"].values.shape, (3, 3))
            self.assertEqual(
                smoke(AudioConfig(models={"lb": ModelConfig(enabled=False)}), Path(tmp), wav),
                {"lb": "disabled"},
            )

    def test_cli_refuses_a_disabled_model(self):
        self.assertTrue(load_audio_config().models["littlebeats"].enabled)
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "c.toml"
            text = DEFAULT_CONFIG_PATH.read_text().replace("enabled = true", "enabled = false")
            config.write_text(text)
            code = main(
                [
                    "--raw-dir",
                    tmp,
                    "--processed-dir",
                    tmp,
                    "audio",
                    "infer",
                    "littlebeats",
                    "--config",
                    str(config),
                ]
            )
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
