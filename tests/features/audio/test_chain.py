import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from stillface.cli import main
from stillface.config import DatasetPaths
from stillface.features.audio.chain import model_group, run_chain
from stillface.features.audio.config import AudioConfig, ModelConfig, Threshold, load_audio_config
from stillface.features.audio.fusion import fusion_paths
from stillface.features.audio.lanes import read_lanes
from stillface.features.audio.models import Segment, SegmentScores
from tests.features.audio.fakes import FakeAdapter
from tests.helpers import make_audio


class FakeVtc(FakeAdapter):
    def run(self, requests):
        out = super().run(requests)
        for result in out:
            segments = (Segment(0.2, 0.8, "", "FEM"),)
            result.segments["voice_type"] = SegmentScores(segments, ("mean_prob",), [[0.9]])
        return out


MODELS = {
    "vtc": ModelConfig(),
    "silero": ModelConfig(),
    "pyannote_embedding": ModelConfig(input="segments", segment_source="vtc:FEM"),
    "audeering": ModelConfig(input="segments", segment_source="roles:mother"),
    "crisperwhisper": ModelConfig(options={"track": "mother"}),
    "mfa": ModelConfig(input="segments", segment_source="asr:mother"),
    "littlebeats": ModelConfig(enabled=False),
}


class Factory:
    def __init__(self):
        self.adapters = {}

    def __call__(self, name, config):
        if name == "silero" and getattr(self, "break_silero", False):
            raise RuntimeError("no environment")
        if name not in self.adapters:
            self.adapters[name] = (FakeVtc if name == "vtc" else FakeAdapter)(name=name)
        return self.adapters[name]


class TestChain(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.paths = DatasetPaths(root / "raw", root / "out")
        meta = self.paths.metadata_path
        meta.parent.mkdir(parents=True)
        pd.DataFrame({"ID": [1, 2, 3], "Analysis": [1, 1, 0]}).to_excel(meta, index=False)
        for session in ("1", "2", "3"):
            make_audio(self.paths.phase_clip(session, "baby", "play"), seconds=1.0, video=True)
        self.config = replace(AudioConfig(), models=MODELS)

    def tearDown(self):
        self.tmp.cleanup()

    def test_groups(self):
        self.assertEqual(
            [
                model_group(MODELS[n])
                for n in ("vtc", "pyannote_embedding", "audeering", "crisperwhisper", "mfa")
            ],
            [1, 2, 3, 4, 5],
        )
        self.assertEqual(model_group(ModelConfig(input="segments")), 3)

    def test_full_chain_resume_and_threshold_change(self):
        factory = Factory()
        report = run_chain(self.paths, self.config, factory=factory, phases=("play",))
        self.assertEqual(report.sessions, ["1", "2"])
        self.assertEqual(report.stages[:2], ["inventory", "prepare"])
        self.assertEqual(report.stages[-1], "qc")
        self.assertEqual(report.models["littlebeats"], "disabled")
        self.assertEqual(report.models["vtc"], "ran")
        self.assertEqual(factory.adapters["pyannote_embedding"].calls, [["1/play", "2/play"]])
        self.assertEqual(
            factory.adapters["crisperwhisper"].calls, [["1/play/mother", "2/play/mother"]]
        )
        self.assertEqual(factory.adapters["audeering"].calls, [])
        self.assertTrue((self.paths.audio_dir / "qc_summary.csv").exists())
        self.assertTrue((self.paths.audio_session_dir("1") / "annie" / "1_play.json").exists())

        for adapter in factory.adapters.values():
            adapter.calls.clear()
        fusion = replace(self.config.fusion, thresholds={"silero": Threshold(0.9, "config")})
        changed = replace(self.config, fusion=fusion)
        run_chain(self.paths, changed, factory=factory, phases=("play",))
        self.assertTrue(all(not a.calls for a in factory.adapters.values()))
        out = fusion_paths(self.paths, "1", "play")
        table = read_lanes(out["lanes"], out["legend"])
        self.assertEqual(table.notes["config"], changed.digest)

    def test_failures_and_missing_environments(self):
        factory = Factory()
        factory.break_silero = True
        report = run_chain(self.paths, self.config, ["1"], factory=factory, phases=("play",))
        self.assertTrue(report.models["silero"].startswith("failed: no environment"))
        envs = Path(self.tmp.name) / "envs"
        config = replace(
            self.config, models={"ast": ModelConfig(env="hf", module="m", weights_revision="r")}
        )
        missing = run_chain(self.paths, config, ["1"], envs=envs, phases=("play",))
        self.assertEqual(missing.models["ast"], "not installed")

        class Partial(Factory):
            def __call__(self, name, config):
                return FakeAdapter(name=name, fail={"1/play"})

        partial = run_chain(
            self.paths,
            replace(self.config, models={"silero": ModelConfig()}),
            ["1"],
            factory=Partial(),
            phases=("play",),
        )
        self.assertEqual(partial.models["silero"], "partial: 1 failed")

    def test_cli_run_with_sessions_file(self):
        sessions = Path(self.tmp.name) / "sessions.txt"
        sessions.write_text("1\n\n")
        config = Path(self.tmp.name) / "audio.toml"
        config.write_text(
            "version = 1\n[models.ast]\nenv = 'hf'\nmodule = 'm'\nweights_revision = 'r'\n"
        )
        envs = Path(self.tmp.name) / "envs"
        base = [
            "--raw-dir",
            str(self.paths.raw_dir),
            "--processed-dir",
            str(self.paths.processed_dir),
            "audio",
            "run",
        ]
        code = main(
            [*base, "--sessions-file", str(sessions), "--envs", str(envs), "--config", str(config)]
        )
        self.assertEqual(code, 0)
        self.assertTrue((self.paths.audio_session_dir("1") / "qc.json").exists())
        self.assertFalse((self.paths.audio_session_dir("2") / "qc.json").exists())

    def test_default_config_groups_every_model(self):
        config = load_audio_config()
        self.assertEqual(model_group(config.models["mfa"]), 5)
        self.assertEqual(model_group(config.models["crisperwhisper"]), 4)
        self.assertEqual(model_group(config.models["egemaps"]), 3)
        self.assertEqual(model_group(config.models["addressee"]), 2)
        self.assertTrue(np.all([model_group(m) in (1, 2, 3, 4, 5) for m in config.models.values()]))


if __name__ == "__main__":
    unittest.main()
