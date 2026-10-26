"""Configuration of the gaze extraction.

Every threshold, margin and prior lives here, never in the code, and a copy is stored in
each run record. Values are calibrated on the development sessions and versioned through
the ``version`` key of the config file.
"""

from dataclasses import dataclass, field
from pathlib import Path

from stillface.features.config import load_config

CONFIG_VERSION: int = 1
"""Supported version of the gaze config file format."""

DEFAULT_CONFIG_PATH: Path = Path(__file__).with_name("default_config.toml")
"""Config file shipped with the package."""


@dataclass(frozen=True)
class SidePriors:
    """Where the mother sits in the side cameras' image.

    Attributes:
        window: ``left`` or ``right`` half of the window camera image.
        door: ``left`` or ``right`` half of the door camera image.
    """

    window: str = "right"
    door: str = "left"


@dataclass(frozen=True)
class Thresholds:
    """Decision thresholds of the class assignment.

    Attributes:
        partner_mass: Heatmap share on the partner needed for the PARTNER class.
        inout: In/out-of-frame score below which the gaze target is out of frame.
        box_margin: Relative margin added around the partner's head box.
    """

    partner_mass: float = 0.3
    inout: float = 0.5
    box_margin: float = 0.25


@dataclass(frozen=True)
class RoleRules:
    """Rules assigning head tracks to the mother and the baby.

    Attributes:
        min_presence: Share of the sampled frames a track must appear in to be a candidate.
        size_ambiguity_ratio: Two candidates whose head areas differ by less than this ratio
            (smaller over larger) cannot be told apart by size.
        min_side_separation: Smaller horizontal distance between the two candidates than this
            (normalized) makes the side prior ambiguous.
        min_identity_similarity: Cosine similarity needed to accept an identity match.
        min_relative_head_area: A head found without a text prompt must be at least this share
            of the prompted partner's head area, which rejects background people.
    """

    min_presence: float = 0.2
    size_ambiguity_ratio: float = 0.8
    min_side_separation: float = 0.05
    min_identity_similarity: float = 0.3
    min_relative_head_area: float = 0.4


@dataclass(frozen=True)
class GazeConfig:
    """Gaze extraction settings.

    Attributes:
        version: Config format version.
        sample_rate_hz: Extraction rate; 30 matches the native rate of the side cameras.
        model_variant: Gaze-LLE variant.
        side_priors: Mother position in the side cameras.
        thresholds: Class decision thresholds.
        roles: Rules assigning head tracks to the mother and the baby.
        partner_zones: Close camera name mapped to the normalized ``(x0, y0, x1, y1)`` image
            region where the partner is looked at; calibrated on the development sessions.
    """

    version: int = CONFIG_VERSION
    sample_rate_hz: float = 30.0
    model_variant: str = "vitl14_inout_childplay"
    side_priors: SidePriors = field(default_factory=SidePriors)
    thresholds: Thresholds = field(default_factory=Thresholds)
    roles: RoleRules = field(default_factory=RoleRules)
    partner_zones: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)


def load_gaze_config(path: Path = DEFAULT_CONFIG_PATH) -> GazeConfig:
    """Load a gaze config file.

    Args:
        path: TOML file, the packaged default if omitted.

    Returns:
        The validated configuration.
    """
    return load_config(path, GazeConfig, CONFIG_VERSION)
