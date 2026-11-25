import tempfile
import unittest
from pathlib import Path

from stillface.features.audio.config import AudioConfig, load_audio_config
from stillface.features.config import ConfigError


class TestAudioConfig(unittest.TestCase):
    def test_default_file_matches_dataclass_defaults(self):
        from dataclasses import replace

        loaded = load_audio_config()
        self.assertEqual(replace(loaded, models={}, fusion=AudioConfig().fusion), AudioConfig())
        self.assertIn("prosody", loaded.models)
        self.assertEqual(loaded.fusion.threshold("ast_type").origin, "probe")
        self.assertIn("cry", loaded.fusion.baby_types)
        with self.assertRaises(KeyError):
            loaded.fusion.threshold("nope")

    def test_values_and_digest(self):
        config = load_audio_config()
        self.assertEqual(config.source.cameras, ("baby", "window", "door"))
        self.assertEqual(len(config.digest), 16)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.toml"
            path.write_text('version = 1\n[scope]\nexclude = ["123"]\n')
            changed = load_audio_config(path)
        self.assertEqual(changed.scope.exclude, ("123",))
        self.assertNotEqual(changed.digest, config.digest)

    def test_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.toml"
            path.write_text("version = 2\n")
            with self.assertRaises(ConfigError):
                load_audio_config(path)
            path.write_text("version = 1\n[source]\nmicrophone = 1\n")
            with self.assertRaises(ConfigError):
                load_audio_config(path)


if __name__ == "__main__":
    unittest.main()
