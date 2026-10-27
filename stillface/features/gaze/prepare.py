"""Per-clip preparation shared by the head pilot and the gaze pipeline."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2

from stillface.features.gaze.config import GazeConfig
from stillface.features.gaze.heads import HeadBox, HeadTracker
from stillface.features.gaze.roles import (
    RoleAssignment,
    assign_roles,
    head_per_frame,
    missing_reason,
)
from stillface.features.timeline import sample_frame_indices

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClipHeads:
    """Sampled frames of a clip with the assigned head of each person.

    Attributes:
        frame_indices: Sampled frames of the clip.
        clip_fps: Frame rate of the clip.
        assignment: Role assignment of the head tracks.
        heads: Person mapped to one box per sampled frame, None where not found.
    """

    frame_indices: list[int]
    clip_fps: float
    assignment: RoleAssignment
    heads: dict[str, list[HeadBox | None]]

    def reason_missing(self, camera: str, person: str) -> str:
        """Why a person has no head on some frame of this clip.

        Args:
            camera: Camera name.
            person: ``mother`` or ``baby``.

        Returns:
            ``roles_ambiguous`` if the roles could not be assigned, else ``occluded`` or
            ``no_head`` (see :func:`~stillface.features.gaze.roles.missing_reason`).
        """
        if "roles_ambiguous" in self.assignment.flags:
            return "roles_ambiguous"
        return missing_reason(camera, person)


def clip_properties(clip: Path) -> tuple[int, float]:
    """Frame count and frame rate of a clip.

    Args:
        clip: Video file.

    Returns:
        ``(n_frames, fps)``.

    Raises:
        RuntimeError: If the clip cannot be read.
    """
    cap = cv2.VideoCapture(str(clip))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot read clip: {clip}")
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()


def prepare_clip(
    clip: Path,
    camera: str,
    phase: str,
    tracker: HeadTracker,
    config: GazeConfig,
    overrides: Mapping[tuple[str, str], Mapping[str, int]] | None = None,
) -> ClipHeads:
    """Track the heads of a clip and assign them to the mother and the baby.

    Args:
        clip: Phase clip of one camera.
        camera: Camera name.
        phase: Experiment phase.
        tracker: Head detector and tracker.
        config: Gaze configuration (sample rate, role rules).
        overrides: Pinned role assignments.

    Returns:
        Sampled frames, role assignment and per-frame heads.
    """
    n_frames, fps = clip_properties(clip)
    indices = sample_frame_indices(n_frames, fps, config.sample_rate_hz)
    tracks = tracker.track(clip, indices)
    assignment = assign_roles(tracks, camera, phase, len(indices), config, overrides=overrides)
    if assignment.flags:
        logger.warning("%s %s: %s", camera, phase, ", ".join(assignment.flags))
    return ClipHeads(indices, fps, assignment, head_per_frame(assignment, tracks, indices))
