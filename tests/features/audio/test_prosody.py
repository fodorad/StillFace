import tempfile
import unittest
from pathlib import Path

import numpy as np

from stillface.features.audio.config import AudioConfig, ModelConfig, load_audio_config
from stillface.features.audio.media import write_wav
from stillface.features.audio.models import InferenceError, InferenceRequest, Segment
from stillface.features.audio.prosody import EgemapsAdapter, ProsodyAdapter
from stillface.features.audio.registry import build_adapter

RATE = 16000


class TestProsody(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        t = np.arange(RATE) / RATE
        tone = 0.3 * np.sin(2 * np.pi * 220 * t)
        self.tone = self.dir / "tone.wav"
        write_wav(self.tone, tone, RATE)
        step = np.concatenate([0.02 * tone[: RATE // 2], 0.4 * tone[RATE // 2 :]])
        self.step = self.dir / "step.wav"
        write_wav(self.step, step, RATE)
        self.silence = self.dir / "silence.wav"
        write_wav(self.silence, np.zeros(RATE), RATE)

    def tearDown(self):
        self.tmp.cleanup()

    def test_f0_of_a_tone_and_silence(self):
        adapter = ProsodyAdapter("prosody")
        tone, silence = adapter.run(
            [InferenceRequest("1/play", self.tone, 1.0), InferenceRequest("1/x", self.silence, 1.0)]
        )
        for group in ("f0_adult", "f0_infant"):
            f0 = tone.frames[group].values[:, 0]
            self.assertLess(abs(np.nanmedian(f0) - 220), 2)
            self.assertTrue(np.isnan(silence.frames[group].values).all())
        self.assertEqual(tone.frames["hnr"].names, ("hnr_db",))
        self.assertTrue(np.isnan(silence.frames["hnr"].values).all())
        # Praat's -300 dB floor of digital silence is no measurement.
        self.assertTrue(np.isnan(silence.frames["intensity"].values).all())
        self.assertGreater(np.nanmin(tone.frames["intensity"].values), -100)
        self.assertIn("praat-parselmouth-", adapter.weights_revision)

    def test_intensity_follows_the_level_step(self):
        result = ProsodyAdapter("prosody").run([InferenceRequest("1/play", self.step, 1.0)])[0]
        group = result.frames["intensity"]
        quiet = np.nanmean(group.values[(group.times > 0.1) & (group.times < 0.4), 0])
        loud = np.nanmean(group.values[(group.times > 0.6) & (group.times < 0.9), 0])
        self.assertAlmostEqual(loud - quiet, 20 * np.log10(20), delta=1.5)

    def test_options_and_errors(self):
        model = ModelConfig(options={"adult_f0_min": "300", "adult_f0_max": "700"})
        result = ProsodyAdapter("prosody", model).run([InferenceRequest("1/play", self.tone, 1.0)])[
            0
        ]
        self.assertTrue(np.isnan(result.frames["f0_adult"].values).mean() > 0.9)
        self.assertEqual(result.meta["options"]["adult_f0_min"], "300")
        missing = ProsodyAdapter("prosody").run(
            [InferenceRequest("1/play", self.dir / "no.wav", 1.0)]
        )
        self.assertIsInstance(missing[0], InferenceError)

    def test_egemaps_per_segment(self):
        adapter = EgemapsAdapter("egemaps")
        segments = (Segment(0.0, 0.5, "mother", "FEM"), Segment(0.5, 1.0, "mother", "FEM"))
        result, bad = adapter.run(
            [
                InferenceRequest("1/play", self.tone, 1.0, segments),
                InferenceRequest("1/x", self.tone, 1.0),
            ]
        )
        scores = result.segments["egemaps"]
        self.assertEqual(scores.values.shape, (2, 88))
        self.assertEqual(len(scores.names), 88)
        self.assertIsInstance(bad, InferenceError)
        self.assertIn("segments", bad.message)

    def test_registered_in_process(self):
        config = load_audio_config()
        self.assertIsInstance(build_adapter("prosody", config), ProsodyAdapter)
        self.assertIsInstance(build_adapter("egemaps", config), EgemapsAdapter)
        self.assertEqual(config.models["prosody"].options["infant_f0_max"], "1000")
        self.assertEqual(AudioConfig().models, {})


if __name__ == "__main__":
    unittest.main()
