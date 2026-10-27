"""Deciding which head track is the mother and which is the baby.

Close cameras show one person, so the dominant track belongs to the camera's owner. The
side cameras show both participants in profile at fixed positions (mother on the right in
the window camera, on the left in the door camera) plus occasional background people. In
the baseline phase the positions may change, so size and face identity decide there.
"""

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stillface.constants import CAMERAS, PHASES
from stillface.features.gaze.config import GazeConfig
from stillface.features.gaze.heads import HeadBox, HeadTrack

logger = logging.getLogger(__name__)

PERSONS: tuple[str, ...] = ("mother", "baby")
"""The two people whose gaze is extracted."""

SIDE_CAMERAS: tuple[str, ...] = ("window", "door")
"""Cameras seeing both participants."""

RoleOverrides = dict[tuple[str, str], dict[str, int]]
"""``(camera, phase)`` mapped to ``{person: track_id}``, from a ``roles.json`` file."""


@dataclass(frozen=True)
class RoleAssignment:
    """Outcome of the role assignment for one clip.

    Attributes:
        mother: Track id of the mother, None if not seen or not decidable.
        baby: Track id of the baby, None if not seen or not decidable.
        flags: ``roles_ambiguous`` if the participants could not be told apart,
            ``single_track`` if only one candidate head was found on a side camera.
    """

    mother: int | None
    baby: int | None
    flags: tuple[str, ...] = ()

    def track_id(self, person: str) -> int | None:
        """Track id of a person."""
        return self.mother if person == "mother" else self.baby


_AMBIGUOUS = RoleAssignment(None, None, ("roles_ambiguous",))


def assign_roles(
    tracks: Sequence[HeadTrack],
    camera: str,
    phase: str,
    n_frames: int,
    config: GazeConfig,
    overrides: Mapping[tuple[str, str], Mapping[str, int]] | None = None,
) -> RoleAssignment:
    """Assign head tracks to the mother and the baby.

    Close cameras: the owner is the track whose face is detected on enough of its frames,
    with the most evidence near the image centre. Side cameras: the two strongest persistent
    heads, excluding far-away background people, are told apart by the fixed seating (side
    prior per camera and phase).

    Args:
        tracks: Head tracks of the clip.
        camera: Camera name.
        phase: Experiment phase.
        n_frames: Number of sampled frames in the clip.
        config: Gaze configuration (side priors, role rules).
        overrides: Pinned assignments that win over the automatic decision.

    Returns:
        The assignment; people that cannot be assigned are None.
    """
    pinned = (overrides or {}).get((camera, phase))
    if pinned is not None:
        return RoleAssignment(pinned.get("mother"), pinned.get("baby"))

    rules = config.roles
    if camera in PERSONS:
        # the camera frames its owner: a head whose face is seen, near the image centre; the
        # partner's head entering from above or a drawing on the wall shows no face
        owners = [t for t in tracks if t.face_share() >= rules.min_owner_face_share]
        owner = max(owners, key=lambda t: t.evidence(centrality=True), default=None)
        track_id = None if owner is None else owner.track_id
        return RoleAssignment(*((track_id, None) if camera == "mother" else (None, track_id)))

    present = [t for t in tracks if len(t.boxes) >= rules.min_presence * n_frames]
    largest = max((t.median_area() for t in present), default=0.0)
    candidates = sorted(
        (t for t in present if t.median_area() >= rules.min_relative_head_area * largest),
        key=_weight,
        reverse=True,
    )[:2]
    if not candidates:
        return RoleAssignment(None, None)
    if len(candidates) == 1:
        # one visible participant cannot be told from the other by position alone
        return RoleAssignment(None, None, ("single_track", "roles_ambiguous"))
    return _by_side(candidates, config.side_priors.side(camera, phase), config)


def _weight(track: HeadTrack) -> float:
    return track.weight()


def _mean_x(track: HeadTrack) -> float:
    return float(np.mean([box.center[0] for box in track.boxes.values()]))


def _by_side(candidates: list[HeadTrack], prior: str, config: GazeConfig) -> RoleAssignment:
    first, second = sorted(candidates, key=_mean_x)
    if _mean_x(second) - _mean_x(first) < config.roles.min_side_separation:
        return _AMBIGUOUS
    mother, baby = (second, first) if prior == "right" else (first, second)
    return RoleAssignment(mother.track_id, baby.track_id)


def head_per_frame(
    assignment: RoleAssignment, tracks: Sequence[HeadTrack], frame_indices: Sequence[int]
) -> dict[str, list[HeadBox | None]]:
    """Look up each person's head box on every sampled frame.

    Args:
        assignment: Role assignment of the clip.
        tracks: Head tracks of the clip.
        frame_indices: Sampled frames.

    Returns:
        Person mapped to one box per frame, None where the head was not found.
    """
    by_id = {t.track_id: t for t in tracks}
    heads: dict[str, list[HeadBox | None]] = {}
    for person in PERSONS:
        track_id = assignment.track_id(person)
        boxes = by_id[track_id].boxes if track_id in by_id else {}
        heads[person] = [boxes.get(frame) for frame in frame_indices]
    return heads


def face_per_frame(
    assignment: RoleAssignment, tracks: Sequence[HeadTrack], frame_indices: Sequence[int]
) -> dict[str, list[bool]]:
    """Whether each person's face was detected on every sampled frame.

    Args:
        assignment: Role assignment of the clip.
        tracks: Head tracks of the clip.
        frame_indices: Sampled frames.

    Returns:
        Person mapped to one flag per frame.
    """
    by_id = {t.track_id: t for t in tracks}
    faces: dict[str, list[bool]] = {}
    for person in PERSONS:
        track_id = assignment.track_id(person)
        seen = by_id[track_id].faces if track_id in by_id else {}
        faces[person] = [frame in seen for frame in frame_indices]
    return faces


def camera_persons(camera: str) -> tuple[str, ...]:
    """People whose gaze a camera can show.

    Args:
        camera: Camera name.

    Returns:
        Both people for the side cameras, only the camera's owner for the close cameras.
    """
    return PERSONS if camera in SIDE_CAMERAS else (camera,)


def missing_reason(camera: str, person: str) -> str:
    """Why a person has no head on a frame.

    The mother's face is regularly hidden by the baby camera in the door view; elsewhere a
    missing head is just not found.

    Args:
        camera: Camera name.
        person: ``mother`` or ``baby``.

    Returns:
        ``occluded`` or ``no_head``.
    """
    return "occluded" if camera == "door" and person == "mother" else "no_head"


def load_role_overrides(path: Path) -> RoleOverrides:
    """Read a per-session ``roles.json``.

    The file maps camera, phase and person to a track id, e.g.
    ``{"window": {"play": {"mother": 3, "baby": 1}}}``.

    Args:
        path: File to read; a missing file means no overrides.

    Returns:
        The overrides, empty if the file does not exist.

    Raises:
        ValueError: If the file names an unknown camera, phase or person, or a non-integer
            track id.
    """
    if not path.exists():
        return {}
    overrides: RoleOverrides = {}
    for camera, phases in json.loads(path.read_text()).items():
        if camera not in CAMERAS:
            raise ValueError(f"{path}: unknown camera {camera!r}")
        for phase, people in phases.items():
            if phase not in PHASES:
                raise ValueError(f"{path}: unknown phase {phase!r}")
            for person, track_id in people.items():
                if person not in PERSONS:
                    raise ValueError(f"{path}: unknown person {person!r}")
                if type(track_id) is not int:
                    raise ValueError(f"{path}: track id of {person} must be an integer")
            overrides[(camera, phase)] = dict(people)
    return overrides
