"""From gaze heatmaps to the plan's gaze target classes.

Classes follow the manual coding scheme: for the baby (T1) ANYARA (at the mother),
TARGY_KORNYEZET (at an object or the environment), ELFORDUL (turned away); for the mother
(T6) BABARA (at the baby) and MASHOVA (elsewhere); NK if not codable.
"""

import logging
from dataclasses import dataclass

import numpy as np

from stillface.constants import CAMERAS
from stillface.features.gaze.config import GazeConfig
from stillface.features.gaze.heads import HeadBox

logger = logging.getLogger(__name__)

NK: str = "NK"
"""Class of a row that cannot be coded."""

PARTNER_CLASS: dict[str, str] = {"mother": "BABARA", "baby": "ANYARA"}
"""Class of looking at the partner, by looker."""

AWAY_CLASS: dict[str, str] = {"mother": "MASHOVA", "baby": "ELFORDUL"}
"""Class of a gaze target outside the frame, by looker."""

ENVIRONMENT_CLASS: dict[str, str] = {"mother": "MASHOVA", "baby": "TARGY_KORNYEZET"}
"""Class of looking at something else in the scene, by looker."""


@dataclass(frozen=True)
class GazeDecision:
    """Class and supporting values of one person on one frame.

    Attributes:
        gaze_class: Target class, :data:`NK` if not codable.
        reason: Why it is not codable, else empty.
        gaze_x: Gaze target x (heatmap peak), normalized, None if not codable.
        gaze_y: Gaze target y (heatmap peak), normalized, None if not codable.
        inout: In/out-of-frame score.
        partner_mass: Heatmap share on the partner (head box or zone), None if undefined.
    """

    gaze_class: str
    reason: str = ""
    gaze_x: float | None = None
    gaze_y: float | None = None
    inout: float | None = None
    partner_mass: float | None = None

    @classmethod
    def not_codable(cls, reason: str) -> "GazeDecision":
        """A row without gaze values.

        Args:
            reason: Why the person cannot be coded, e.g. ``occluded``.

        Returns:
            Decision of class :data:`NK`.
        """
        return cls(NK, reason)


def _coverage(low: float, high: float, size: int) -> np.ndarray:
    """Fraction of each of ``size`` equal cells on [0, 1] covered by ``[low, high]``."""
    edges = np.arange(size + 1) / size
    overlap = np.minimum(edges[1:], high) - np.maximum(edges[:-1], low)
    return np.clip(overlap, 0.0, None) * size


def mass_in_box(heatmap: np.ndarray, box: HeadBox) -> float:
    """Share of a heatmap's mass inside a normalized box.

    Cells that the box covers partially count proportionally, so the result is 1 for the
    whole image and never shrinks when the box grows.

    Args:
        heatmap: Gaze map of shape ``(height, width)``.
        box: Region in normalized image coordinates.

    Returns:
        Value in [0, 1]; 0 if the heatmap is empty.
    """
    total = float(heatmap.sum())
    if total <= 0.0:
        return 0.0
    height, width = heatmap.shape
    weights = _coverage(box.y0, box.y1, height)[:, None] * _coverage(box.x0, box.x1, width)[None, :]
    return float((heatmap * weights).sum() / total)


def gaze_point(heatmap: np.ndarray) -> tuple[float, float]:
    """Location of the heatmap maximum.

    Args:
        heatmap: Gaze map of shape ``(height, width)``.

    Returns:
        Normalized ``(x, y)`` of the peak cell center.
    """
    row, col = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    return (col + 0.5) / heatmap.shape[1], (row + 0.5) / heatmap.shape[0]


def classify(
    person: str,
    camera: str,
    heatmap: np.ndarray,
    inout: float | None,
    partner_box: HeadBox | None,
    config: GazeConfig,
) -> GazeDecision:
    """Turn one person's gaze heatmap into a target class.

    Side cameras see the partner, so looking at the partner means heatmap mass inside the
    partner's (enlarged) head box. A close camera does not see the partner; there the
    calibrated partner zone of that camera takes its place.

    Args:
        person: ``mother`` or ``baby``, the looker.
        camera: Camera the frame comes from.
        heatmap: Gaze map of shape ``(height, width)``.
        inout: Probability that the target is inside the frame, None if unavailable.
        partner_box: Head box of the partner on side cameras, None if not found.
        config: Thresholds, margins and partner zones.

    Returns:
        The decision with the supporting values.

    Raises:
        ValueError: If ``person`` or ``camera`` is unknown.
    """
    if person not in PARTNER_CLASS:
        raise ValueError(f"Unknown person {person!r}")
    if camera not in CAMERAS:
        raise ValueError(f"Unknown camera {camera!r}")

    thresholds = config.thresholds
    side = camera in ("window", "door")
    if side:
        if partner_box is None:
            return GazeDecision.not_codable("partner_not_found")
        region = partner_box.expand(thresholds.box_margin)
    else:
        zone = config.partner_zones.get(camera)
        if zone is None:
            return GazeDecision.not_codable("no_partner_zone")
        region = HeadBox(*zone)

    mass = mass_in_box(heatmap, region)
    x, y = gaze_point(heatmap)
    out_of_frame = inout is not None and inout < thresholds.inout
    # On a side camera the partner is in the frame, so an out-of-frame target rules it out;
    # a close camera's partner is off-frame, so the zone is checked first.
    if side and out_of_frame:
        gaze_class = AWAY_CLASS[person]
    elif mass >= thresholds.partner_mass:
        gaze_class = PARTNER_CLASS[person]
    elif out_of_frame:
        gaze_class = AWAY_CLASS[person]
    else:
        gaze_class = ENVIRONMENT_CLASS[person]
    return GazeDecision(gaze_class, "", x, y, inout, mass)
