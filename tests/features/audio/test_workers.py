import tempfile
import unittest
from pathlib import Path

import numpy as np

from stillface.features.audio.config import AudioConfig, ModelConfig, load_audio_config
from stillface.features.audio.models import InferenceRequest
from stillface.features.audio.registry import IN_PROCESS
from stillface.features.audio.smoke import main, smoke, synthetic_clip
from stillface.features.audio.workers import ast_audioset, silero
from stillface.features.audio.workers.base import window_starts
from tests.features.audio.fakes import FakeAdapter


class TestWindowStarts(unittest.TestCase):
    def test_regular_short_and_invalid(self):
        self.assertEqual(window_starts(3.0, 2.0, 0.25), [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertEqual(window_starts(1.0, 2.0, 0.25), [0.0])
        self.assertEqual(window_starts(2.0, 2.0, 0.5), [0.0])
        with self.assertRaises(ValueError):
            window_starts(3.0, 0.0, 0.25)


class TestPostprocess(unittest.TestCase):
    def test_silero_chunk_centres(self):
        rng = np.random.default_rng(0)
        scores = silero.postprocess(rng.uniform(size=31), 16000)
        self.assertEqual(scores.names, ("speech",))
        self.assertEqual(scores.values.shape, (31, 1))
        self.assertAlmostEqual(scores.times[0], 0.016)
        self.assertAlmostEqual(scores.times[1] - scores.times[0], 0.032)
        self.assertEqual(
            silero.postprocess(np.array([1.2, -0.1]), 16000).values.ravel().tolist(), [1.0, 0.0]
        )

    def test_ast_window_centres_and_validation(self):
        names = tuple(f"c{i}" for i in range(ast_audioset.N_CLASSES))
        starts = window_starts(3.0, 2.0, 0.25)
        probs = np.random.default_rng(1).uniform(size=(len(starts), len(names)))
        scores = ast_audioset.postprocess(starts, 2.0, probs, names)
        self.assertEqual(scores.values.shape, (5, 527))
        np.testing.assert_allclose(scores.times, np.array(starts) + 1.0)
        with self.assertRaises(ValueError):
            ast_audioset.postprocess(starts, 2.0, probs[:2], names)
        with self.assertRaises(ValueError):
            ast_audioset.postprocess(starts, 2.0, probs + 1.0, names)


class FakeForwardAst(ast_audioset.AstWorker):
    """AST worker whose model is replaced by a constant (tests the windowing in infer)."""

    def __init__(self):
        self.window_s, self.hop_s = 2.0, 0.5
        self.names = ("Speech", "Baby cry, infant cry")
        self.seen = []

    def forward(self, windows, rate):
        self.seen = [len(w) for w in windows]
        return np.full((len(windows), len(self.names)), 0.25)


class FakeForwardSilero(silero.SileroWorker):
    def forward(self, samples, rate):
        return np.full(len(samples) // silero.CHUNK, 0.9)


class TestInferWithFakeForward(unittest.TestCase):
    def test_windows_and_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = synthetic_clip(Path(tmp) / "c.wav", seconds=3.0)
            request = InferenceRequest("1/play", wav, 3.0)
            worker = FakeForwardAst()
            result = worker.infer(request)
            self.assertEqual(worker.seen, [32000, 32000, 32000])
            np.testing.assert_allclose(result.frames["audioset"].times, [1.0, 1.5, 2.0])
            vad = FakeForwardSilero().infer(request).frames["speech"]
            self.assertEqual(vad.values.shape, (93, 1))


class TestSmoke(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.wav = synthetic_clip(self.dir / "s.wav", seconds=2.0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_statuses(self):
        IN_PROCESS["fake"] = lambda name, model: FakeAdapter(name=name)
        IN_PROCESS["broken"] = lambda name, model: FakeAdapter(name=name, fail={"smoke/play"})

        def explode(name, model):
            raise RuntimeError("cannot build")

        IN_PROCESS["explode"] = explode
        try:
            config = AudioConfig(
                models={
                    "fake": ModelConfig(),
                    "seg": ModelConfig(input="segments"),
                    "broken": ModelConfig(),
                    "explode": ModelConfig(),
                    "absent": ModelConfig(env="nowhere", module="m", weights_revision="r"),
                    "silero": ModelConfig(),
                }
            )
            IN_PROCESS["seg"] = lambda name, model: FakeAdapter(name=name)
            # A real model name whose result breaks that model's contract.
            IN_PROCESS["silero"] = lambda name, model: FakeAdapter(name=name)
            status = smoke(config, self.dir, self.wav)
        finally:
            for key in ("fake", "broken", "explode", "seg", "silero"):
                IN_PROCESS.pop(key)
        self.assertEqual(status["fake"], "ok: frames/scores")
        self.assertEqual(status["seg"], "ok: frames/scores, segments/scores")
        self.assertEqual(status["broken"], "error: fake failure")
        self.assertEqual(status["explode"], "error: cannot build")
        self.assertEqual(status["absent"], "not installed")
        self.assertEqual(status["silero"], "error: contract: missing frames/speech")

    def test_main_with_prosody_only(self):
        config = self.dir / "c.toml"
        config.write_text(
            "version = 1\n[models.prosody]\n"
            "[models.vad]\nenv = 'vad'\nmodule = 'm'\nweights_revision = 'r'\n"
        )
        self.assertEqual(main(["--config", str(config), "--envs", str(self.dir)]), 0)
        self.assertEqual(
            main(["--config", str(config), "--envs", str(self.dir), "--model", "prosody"]), 0
        )

    def test_default_config_models(self):
        config = load_audio_config()
        self.assertEqual(config.models["ast"].env, "hf")
        self.assertEqual(config.models["silero"].module, "stillface.features.audio.workers.silero")
        self.assertEqual(config.models["egemaps"].input, "segments")
        with self.assertRaises(ValueError):
            ModelConfig(input="video")


if __name__ == "__main__":
    unittest.main()
