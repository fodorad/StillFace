import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from stillface.features.config import ConfigError, load_config


@dataclass(frozen=True)
class Inner:
    mass: float = 0.5


@dataclass(frozen=True)
class Demo:
    version: int = 1
    rate: float = 30.0
    name: str = "x"
    inner: Inner = Inner()
    zones: dict[str, tuple[float, float, float, float]] | None = None


def write(tmp: str, text: str) -> Path:
    path = Path(tmp) / "c.toml"
    path.write_text(text)
    return path


class TestLoadConfig(unittest.TestCase):
    def test_defaults_are_explicit_and_overridable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(write(tmp, "version = 1\nrate = 10\n[inner]\nmass = 0.8\n"), Demo, 1)
        self.assertEqual((cfg.rate, cfg.name, cfg.inner.mass), (10.0, "x", 0.8))

    def test_nested_tables_and_tuples(self) -> None:
        text = "version = 1\n[zones]\nbaby = [0.1, 0.2, 0.3, 0.4]\n"
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(write(tmp, text), Demo, 1)
        self.assertEqual(cfg.zones, {"baby": (0.1, 0.2, 0.3, 0.4)})

    def test_unknown_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ConfigError, "typo"):
                load_config(write(tmp, "version = 1\ntypo = 3\n"), Demo, 1)
            with self.assertRaisesRegex(ConfigError, "oops"):
                load_config(write(tmp, "version = 1\n[inner]\noops = 3\n"), Demo, 1)

    def test_missing_or_wrong_version_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ConfigError, "version"):
                load_config(write(tmp, "rate = 1\n"), Demo, 1)
            with self.assertRaisesRegex(ConfigError, "version"):
                load_config(write(tmp, "version = 2\n"), Demo, 1)

    def test_wrong_types_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ConfigError, "rate"):
                load_config(write(tmp, 'version = 1\nrate = "fast"\n'), Demo, 1)

    def test_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_config(Path("nope.toml"), Demo, 1)

    def test_structure_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ConfigError, "inner"):
                load_config(write(tmp, "version = 1\ninner = 3\n"), Demo, 1)
            with self.assertRaisesRegex(ConfigError, "zones"):
                load_config(write(tmp, "version = 1\nzones = 3\n"), Demo, 1)
            with self.assertRaisesRegex(ConfigError, "zones.baby"):
                load_config(write(tmp, "version = 1\n[zones]\nbaby = 3\n"), Demo, 1)
            with self.assertRaisesRegex(ConfigError, "4 items"):
                load_config(write(tmp, "version = 1\n[zones]\nbaby = [1.0, 2.0]\n"), Demo, 1)


if __name__ == "__main__":
    unittest.main()
