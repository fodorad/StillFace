import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stillface.config import DatasetPaths
from stillface.features.audio.config import ModelConfig
from stillface.features.audio.infer import (
    full_wav_request,
    load_raw,
    raw_path,
    run_model,
    split_key,
    wav_duration,
)
from stillface.features.audio.media import write_wav
from stillface.features.audio.models import InferenceRequest, Segment
from stillface.features.audio.prepare import wav_path
from stillface.features.audio.provenance import read_sidecar
from tests.features.audio.fakes import FakeAdapter


class TestInfer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.paths = DatasetPaths(root / "raw", root / "out")
        rng = np.random.default_rng(0)
        for session in ("1", "2"):
            write_wav(wav_path(self.paths, session, "play"), rng.uniform(-0.1, 0.1, 16000), 16000)

    def tearDown(self):
        self.tmp.cleanup()

    def test_keys_and_paths(self):
        self.assertEqual(split_key("1/play"), ("1", "play", ""))
        self.assertEqual(split_key("1/play/mother"), ("1", "play", "mother"))
        with self.assertRaises(ValueError):
            split_key("bad")
        self.assertEqual(raw_path(self.paths, "vtc", "1/play/mother").name, "play_mother.npz")

    def test_full_wav_request_and_duration(self):
        request = full_wav_request(self.paths, "1", "play")
        self.assertEqual((request.key, request.duration_s, request.segments), ("1/play", 1.0, None))
        self.assertIsNone(full_wav_request(self.paths, "1", "reunion"))
        self.assertAlmostEqual(wav_duration(wav_path(self.paths, "1", "play")), 1.0)

    def test_batch_outputs_failures_and_staleness(self):
        fake = FakeAdapter(name="vad", fail={"2/play"})
        model = ModelConfig()
        report = run_model(self.paths, fake, model, ["1", "2"], ["play", "reunion"])
        self.assertEqual(fake.calls, [["1/play", "2/play"]])
        self.assertEqual(report.written, ["1/play"])
        self.assertEqual(report.failed, {"2/play": "fake failure"})
        self.assertEqual(report.skipped, ["1/reunion", "2/reunion"])
        result = load_raw(self.paths, "vad", "1/play")
        self.assertEqual(result.frames["scores"].values.shape, (11, 1))
        self.assertIsNone(load_raw(self.paths, "vad", "2/play"))
        side = read_sidecar(raw_path(self.paths, "vad", "1/play"))
        self.assertEqual(side["input"], "Sessions/1/features/audio/wav/play.wav")
        self.assertEqual((side["weights_revision"], side["meta"]["device"]), ("rev-1", "cpu"))
        runs = json.loads((self.paths.audio_dir / "runs" / "vad.json").read_text())
        self.assertEqual(runs["written"], ["1/play"])

        fake.fail = set()
        again = run_model(self.paths, fake, model, ["1", "2"], ["play"])
        self.assertEqual((again.kept, again.written), (["1/play"], ["2/play"]))
        changed = run_model(self.paths, fake, ModelConfig(window_s=1.0), ["1"], ["play"])
        self.assertEqual(changed.written, ["1/play"])
        fake.worker_version = "2"
        self.assertEqual(
            run_model(self.paths, fake, ModelConfig(window_s=1.0), ["1"], ["play"]).written,
            ["1/play"],
        )
        write_wav(wav_path(self.paths, "1", "play"), np.zeros(16000), 16000)
        self.assertEqual(
            run_model(self.paths, fake, ModelConfig(window_s=1.0), ["1"], ["play"]).written,
            ["1/play"],
        )
        self.assertEqual(
            run_model(
                self.paths, fake, ModelConfig(window_s=1.0), ["1"], ["play"], overwrite=True
            ).written,
            ["1/play"],
        )

    def test_segment_requests_and_artefacts(self):
        segments = (Segment(0.1, 0.5, "mother", "FEM"),)

        def builder(paths, session, phase):
            wav = wav_path(paths, session, phase)
            return InferenceRequest(f"{session}/{phase}/mother", wav, 1.0, segments)

        fake = FakeAdapter(name="addressee")
        report = run_model(self.paths, fake, ModelConfig(), ["1"], ["play"], builder)
        self.assertEqual(report.written, ["1/play/mother"])
        result = load_raw(self.paths, "addressee", "1/play/mother")
        self.assertEqual(result.segments["scores"].segments, segments)
        self.assertIsNotNone(
            read_sidecar(raw_path(self.paths, "addressee", "1/play/mother"))["segments_sha256"]
        )

        class WithArtefact(FakeAdapter):
            def run(self, requests):
                out = super().run(requests)
                for r in out:
                    r.artefacts["rttm"] = "SPEAKER\n"
                return out

        run_model(self.paths, WithArtefact(name="vtc"), ModelConfig(), ["1"], ["play"])
        self.assertEqual(
            raw_path(self.paths, "vtc", "1/play").with_suffix(".rttm").read_text(), "SPEAKER\n"
        )


if __name__ == "__main__":
    unittest.main()
