import unittest
from pathlib import Path

from stillface.features.audio.config import AudioConfig, ModelConfig
from stillface.features.audio.registry import (
    ENVS_VAR,
    IN_PROCESS,
    build_adapter,
    env_python,
    envs_root,
    worker_options,
)
from stillface.features.audio.subprocess_adapter import SubprocessAdapter
from tests.features.audio.fakes import FakeAdapter


class TestRegistry(unittest.TestCase):
    def test_envs_root(self):
        self.assertEqual(envs_root({ENVS_VAR: "/envs"}), Path("/envs"))
        with self.assertRaises(RuntimeError):
            envs_root({})

    def test_subprocess_model(self):
        model = ModelConfig(
            env="hf", module="m.w", weights_revision="abc", window_s=2.0, options={"k": "v"}
        )
        config = AudioConfig(models={"ast": model})
        built = build_adapter("ast", config, Path("/envs"))
        self.assertIsInstance(built, SubprocessAdapter)
        self.assertEqual(built.python, env_python(Path("/envs"), "hf"))
        self.assertEqual(built.python, Path("/envs/hf/bin/python"))
        self.assertEqual(built.options, worker_options(model))
        self.assertEqual(built.options["window_s"], "2.0")
        self.assertEqual(built.options["k"], "v")

    def test_in_process_and_errors(self):
        config = AudioConfig(models={"fake": ModelConfig(), "orphan": ModelConfig()})
        IN_PROCESS["fake"] = lambda name, model: FakeAdapter(name=name)
        try:
            self.assertEqual(build_adapter("fake", config).name, "fake")
        finally:
            del IN_PROCESS["fake"]
        with self.assertRaises(ValueError):
            build_adapter("orphan", config)
        with self.assertRaises(ValueError):
            build_adapter("missing", config)

    def test_model_digest_changes(self):
        self.assertNotEqual(ModelConfig().digest, ModelConfig(window_s=1.0).digest)


if __name__ == "__main__":
    unittest.main()
