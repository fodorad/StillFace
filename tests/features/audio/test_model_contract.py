"""Integration tests: every configured model runs in its real environment and keeps its contract.

Opt-in and local only: they run when ``STILLFACE_AUDIO_ENVS`` points at the model
environments (``make audio-contract``) and are skipped otherwise, so CI stays torch-free.
``STILLFACE_AUDIO_TEST_WAV`` may name a real 16 kHz mono recording; without it a synthetic
clip is used (enough for the plumbing, not for the segment models' content). Segment models
get VTC's segments of the same clip, as in the chain, and fixed segments when VTC finds none.
"""

import os
import tempfile
import unittest
from pathlib import Path

from stillface.features.audio.asr import utterances
from stillface.features.audio.config import load_audio_config
from stillface.features.audio.contract import CONTRACTS, check_result
from stillface.features.audio.media import read_wav
from stillface.features.audio.models import InferenceError, InferenceRequest, Segment
from stillface.features.audio.registry import build_adapter, env_python
from stillface.features.audio.smoke import SMOKE_SEGMENTS, synthetic_clip

ENVS = os.environ.get("STILLFACE_AUDIO_ENVS", "")
ROLE_OF = {"FEM": "mother", "KCHI": "baby", "OCH": "baby", "MAL": "other"}


def segments_for(source: str, vtc: tuple[Segment, ...]) -> tuple[Segment, ...]:
    """Request segments of a segment model from VTC's segments (roles approximated by class)."""
    kind, _, wanted = source.partition(":")
    names = set(wanted.split(","))
    if kind == "vtc":
        chosen = tuple(s for s in vtc if s.label in names)
    else:
        chosen = tuple(
            Segment(s.start_s, s.end_s, ROLE_OF[s.label], s.label)
            for s in vtc
            if ROLE_OF.get(s.label) in names
        )
    return chosen or SMOKE_SEGMENTS


@unittest.skipUnless(ENVS, "STILLFACE_AUDIO_ENVS is not set (model environments needed)")
class TestModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        given = os.environ.get("STILLFACE_AUDIO_TEST_WAV")
        cls.wav = Path(given) if given else synthetic_clip(Path(cls.tmp.name) / "clip.wav")
        samples, rate = read_wav(cls.wav)
        cls.duration = len(samples) / rate
        cls.config = load_audio_config()
        cls.root = Path(ENVS)
        cls.vtc_segments: tuple[Segment, ...] = ()
        cls.words: tuple[Segment, ...] = ()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_model(self, name: str):
        model = self.config.models[name]
        if model.env and not env_python(self.root, model.env).exists():
            self.skipTest(f"environment {model.env!r} is not installed")
        segments = None
        if model.input == "segments" and model.segment_source.startswith("asr:"):
            # As in the chain: the ASR words grouped into one utterance per mother segment.
            spans = [(s.start_s, s.end_s) for s in segments_for("roles:mother", self.vtc_segments)]
            segments = tuple(utterances(self.words, spans))
            if not segments:
                self.skipTest("no transcribed words to align (synthetic clip?)")
        elif model.input == "segments":
            segments = segments_for(model.segment_source, self.vtc_segments)
        request = InferenceRequest("contract/play", self.wav, self.duration, segments)
        result = build_adapter(name, self.config, self.root).run([request])[0]
        if isinstance(result, InferenceError):
            self.fail(f"{name} failed: {result.message}")
        problems = check_result(CONTRACTS[name], result, self.duration)
        self.assertEqual(problems, [], f"{name} breaks its contract")
        if segments is not None:
            group = CONTRACTS[name].groups[0].group
            self.assertEqual(len(result.segments[group].segments) > 0, True)
        return result

    # The order matters: VTC first, its segments feed the segment models.
    def test_01_vtc(self):
        result = self.run_model("vtc")
        type(self).vtc_segments = tuple(result.segments["voice_type"].segments)

    def test_02_silero(self):
        self.run_model("silero")

    def test_03_pyannote_vad(self):
        self.run_model("pyannote_vad")

    def test_04_ast(self):
        self.run_model("ast")

    def test_05_prosody(self):
        self.run_model("prosody")

    def test_06_littlebeats(self):
        self.run_model("littlebeats")

    def test_07_addressee(self):
        self.run_model("addressee")

    def test_08_pyannote_embedding(self):
        result = self.run_model("pyannote_embedding")
        embeddings = result.segments["embedding"].embeddings
        self.assertIsNotNone(embeddings)
        self.assertEqual(embeddings.shape[0], len(result.segments["embedding"].segments))

    def test_09_audeering(self):
        self.run_model("audeering")

    def test_10_egemaps(self):
        self.run_model("egemaps")

    def test_11_emotion2vec(self):
        self.run_model("emotion2vec")

    def test_12_crisperwhisper(self):
        result = self.run_model("crisperwhisper")
        type(self).words = tuple(result.segments["verbatim_words"].segments)

    def test_13_mfa(self):
        self.run_model("mfa")


if __name__ == "__main__":
    unittest.main()
