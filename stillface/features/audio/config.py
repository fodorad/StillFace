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
class PrepareConfig:
    """Prepared analysis audio.

    Attributes:
        sample_rate: Output sample rate; every model reads 16 kHz mono.
    """

    sample_rate: int = 16000


@dataclass(frozen=True)
class ModelConfig:
    """How one model is run.

    Attributes:
        env: Environment folder below ``$STILLFACE_AUDIO_ENVS``; empty for in-process models.
        module: Worker module run inside the environment.
        weights_revision: Exact weights revision (e.g. a Hugging Face commit hash).
        worker_version: Version of the worker code; bump it when its output changes.
        device: Preferred device (``mps``, ``cuda``, ``cpu``); workers fall back to CPU.
        timeout_s: Wall-clock limit of one batch.
        window_s: Analysis window for windowed models, 0 if not applicable.
        hop_s: Hop between windows, 0 if not applicable.
        options: Further model-specific options, passed to the worker as text.
        input: ``wav`` for whole-phase models (first pass), ``segments`` for models that
            score role segments (second pass).
        enabled: False keeps a configured model out of every run (recorded as disabled),
            e.g. until its outputs are verified.
    """

    env: str = ""
    module: str = ""
    weights_revision: str = ""
    worker_version: str = "1"
    device: str = "mps"
    timeout_s: float = 7200.0
    window_s: float = 0.0
    hop_s: float = 0.0
    options: dict[str, str] = field(default_factory=dict)
    input: str = "wav"
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate the input kind."""
        if self.input not in ("wav", "segments"):
            raise ValueError(f"model input must be 'wav' or 'segments', not {self.input!r}")

    @property
    def digest(self) -> str:
        """Short SHA-256 of these settings; a change recomputes only this model's outputs."""
        text = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class AudioConfig:
    """Audio pipeline configuration.

    Attributes:
        version: Config file format version.
        source: Microphone choice.
        scope: Session selection.
        inventory: Inventory statistics.
        prepare: Prepared audio format.
        models: Model name mapped to how it is run.
    """

    version: int = CONFIG_VERSION
    source: SourceConfig = field(default_factory=SourceConfig)
    scope: ScopeConfig = field(default_factory=ScopeConfig)
    inventory: InventoryConfig = field(default_factory=InventoryConfig)
    prepare: PrepareConfig = field(default_factory=PrepareConfig)
    models: dict[str, ModelConfig] = field(default_factory=dict)

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
