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
class RolesConfig:
    """Who is who: VTC voice types to roles, and the mother's voice enrolment.

    Attributes:
        vtc_model: Model whose segments give the voice types.
        embedding_model: Model whose segment embeddings identify the mother.
        baby_labels: VTC classes that are the baby (no siblings: key child and other child).
        enrol_phase: Phase whose female speech enrols the mother's voice.
        fallback_phase: Phase used when the enrolment phase has too little speech.
        enrol_total_s: Longest female segments are used up to this total.
        enrol_min_s: Less enrolment speech than this leaves the session unresolved.
        tau_high: Cosine similarity at or above which a female segment is the mother.
        tau_low: Cosine similarity at or below which it is another woman.
    """

    vtc_model: str = "vtc"
    embedding_model: str = "pyannote_embedding"
    baby_labels: tuple[str, ...] = ("KCHI", "OCH")
    enrol_phase: str = "play"
    fallback_phase: str = "reunion"
    enrol_total_s: float = 30.0
    enrol_min_s: float = 8.0
    tau_high: float = 0.6
    tau_low: float = 0.4


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
        segment_source: For ``segments`` models, where the segments come from:
            ``vtc:<labels>`` (VTC segments with these classes, e.g. ``vtc:FEM,MAL``) or
            ``roles:<roles>`` (role segments, e.g. ``roles:mother,baby``).
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
    segment_source: str = ""

    def __post_init__(self) -> None:
        """Validate the input kind and the segment source."""
        if self.input not in ("wav", "segments"):
            raise ValueError(f"model input must be 'wav' or 'segments', not {self.input!r}")
        if self.segment_source:
            kind, _, names = self.segment_source.partition(":")
            if kind not in ("vtc", "roles") or not names:
                raise ValueError("segment_source must be 'vtc:<labels>' or 'roles:<roles>'")

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
        roles: Role assignment settings.
    """

    version: int = CONFIG_VERSION
    source: SourceConfig = field(default_factory=SourceConfig)
    scope: ScopeConfig = field(default_factory=ScopeConfig)
    inventory: InventoryConfig = field(default_factory=InventoryConfig)
    prepare: PrepareConfig = field(default_factory=PrepareConfig)
    models: dict[str, ModelConfig] = field(default_factory=dict)
    roles: RolesConfig = field(default_factory=RolesConfig)

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
