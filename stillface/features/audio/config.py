"""Configuration of the audio pipeline.

Every source choice, threshold and rule lives here, never in the code; the resolved config
is copied into each run record and its hash into every provenance sidecar.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from stillface.features.config import load_config

CONFIG_VERSION: int = 1
"""Supported version of the audio config file format."""

DEFAULT_CONFIG_PATH: Path = Path(__file__).with_name("default_config.toml")
"""Config file shipped with the package."""


@dataclass(frozen=True)
class SourceConfig:
    """Which microphone feeds the pipeline.

    Attributes:
        cameras: Camera clips tried in order; the first with a decodable audio stream wins.
            The baby-cam phone lies next to the mother's head and hears both partners.
    """

    cameras: tuple[str, ...] = ("baby", "window", "door")


@dataclass(frozen=True)
class ScopeConfig:
    """Which sessions are analysed.

    Attributes:
        analysis_column: Metadata column flagging analysed sessions.
        exclude: Session ids left out although flagged (e.g. the first visit of a dyad).
    """

    analysis_column: str = "Analysis"
    exclude: tuple[str, ...] = ()


@dataclass(frozen=True)
class InventoryConfig:
    """Level statistics of the inventory.

    Attributes:
        frame_s: Frame length for the near-silence share.
        silence_dbfs: Frame RMS below this level counts as near-silent.
        clip_level: Absolute sample value at or above which a sample counts as clipped.
    """

    frame_s: float = 0.05
    silence_dbfs: float = -60.0
    clip_level: float = 0.999


@dataclass(frozen=True)
class AudioConfig:
    """Audio pipeline configuration.

    Attributes:
        version: Config file format version.
        source: Microphone choice.
        scope: Session selection.
        inventory: Inventory statistics.
    """

    version: int = CONFIG_VERSION
    source: SourceConfig = field(default_factory=SourceConfig)
    scope: ScopeConfig = field(default_factory=ScopeConfig)
    inventory: InventoryConfig = field(default_factory=InventoryConfig)

    @property
    def digest(self) -> str:
        """Short SHA-256 of the resolved configuration, recorded in provenance sidecars."""
        text = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:16]


def load_audio_config(path: Path = DEFAULT_CONFIG_PATH) -> AudioConfig:
    """Load and validate an audio config file.

    Args:
        path: TOML file; the packaged defaults if omitted.

    Returns:
        The configuration.
    """
    return load_config(path, AudioConfig, CONFIG_VERSION)
