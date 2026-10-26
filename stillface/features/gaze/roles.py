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
class RoleReferences:
    """Face-identity embeddings of the two participants of a session.

    Attributes:
        mother: Mean embedding from the mother camera.
        baby: Mean embedding from the baby camera.
    """

    mother: tuple[float, ...]
    baby: tuple[float, ...]


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
    references: RoleReferences | None = None,
    overrides: Mapping[tuple[str, str], Mapping[str, int]] | None = None,
) -> RoleAssignment:
    """Assign head tracks to the mother and the baby.

    Args:
        tracks: Head tracks of the clip.
        camera: Camera name.
        phase: Experiment phase.
        n_frames: Number of sampled frames in the clip.
        config: Gaze configuration (side priors, role rules).
        references: Identity embeddings of the participants, if available.
        overrides: Pinned assignments that win over the automatic decision.

    Returns:
        The assignment; people that cannot be assigned are None.
    """
    pinned = (overrides or {}).get((camera, phase))
    if pinned is not None:
        return RoleAssignment(pinned.get("mother"), pinned.get("baby"))

    if camera in PERSONS:
        # the camera's owner is the most confidently present face, not the biggest box: the
        # partner's face partly enters the close-up views as a large, weak detection
        dominant = max(tracks, key=lambda t: t.evidence(), default=None)
        track_id = None if dominant is None else dominant.track_id
        return RoleAssignment(*((track_id, None) if camera == "mother" else (None, track_id)))

    rules = config.roles
    tagged = {t.role: t for t in tracks if t.role is not None and t.role in PERSONS}
    if tagged:
        return _with_tagged(tracks, tagged, n_frames, config)
    candidates = sorted(
        (t for t in tracks if len(t.boxes) >= rules.min_presence * n_frames),
        key=_weight,
        reverse=True,
    )[:2]
    if not candidates:
        return RoleAssignment(None, None)
    prior = getattr(config.side_priors, camera)
    if len(candidates) == 1:
        # one visible participant cannot be told from the other by position alone: the mother
        # leans over the baby, and the baby's face is often missed
        return RoleAssignment(None, None, ("single_track", "roles_ambiguous"))
    if phase == "baseline":
        return _by_identity_or_size(candidates, config, references)
    return _by_side(candidates, prior, config)


def _with_tagged(
    tracks: Sequence[HeadTrack],
    tagged: dict[str, HeadTrack],
    n_frames: int,
    config: GazeConfig,
) -> RoleAssignment:
    """Assignment when a text-prompted detector already named some of the heads.

    The missing role takes the most confident anonymous head that is not tiny compared with
    the named partner's (background people are small).
    """
    ids = {person: tagged[person].track_id if person in tagged else None for person in PERSONS}
    rules = config.roles
    for person in PERSONS:
        if ids[person] is not None:
            continue
        partner = tagged.get("baby" if person == "mother" else "mother")
        floor = 0.0
        if partner is not None and partner.boxes:
            floor = rules.min_relative_head_area * float(
                np.median([box.area for box in partner.boxes.values()])
            )
        used = {i for i in ids.values() if i is not None}
        candidates = [
            t for t in tracks
            if t.role is None and t.track_id not in used
            and len(t.boxes) >= rules.min_presence * n_frames
            and float(np.mean([b.area for b in t.boxes.values()])) >= floor
        ]  # fmt: skip
        if candidates:
            ids[person] = max(candidates, key=lambda t: t.weight()).track_id
    return RoleAssignment(ids["mother"], ids["baby"])


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


def _by_identity_or_size(
    candidates: list[HeadTrack], config: GazeConfig, references: RoleReferences | None
) -> RoleAssignment:
    if references is not None:
        matched = _by_identity(candidates, config.roles.min_identity_similarity, references)
        if matched is not None:
            return matched
    larger, smaller = sorted(candidates, key=_weight, reverse=True)
    if _weight(smaller) / _weight(larger) >= config.roles.size_ambiguity_ratio:
        return _AMBIGUOUS
    return RoleAssignment(larger.track_id, smaller.track_id)


def _by_identity(
    candidates: list[HeadTrack], min_similarity: float, references: RoleReferences
) -> RoleAssignment | None:
    embeddings = [t.embedding for t in candidates if t.embedding is not None]
    if len(embeddings) != len(candidates):
        return None
    similarity = np.array(
        [[_cosine(e, references.mother), _cosine(e, references.baby)] for e in embeddings]
    )
    as_mother, as_baby = similarity.argmax(axis=0)
    if as_mother == as_baby:
        return None
    if similarity[as_mother, 0] < min_similarity or similarity[as_baby, 1] < min_similarity:
        return None
    return RoleAssignment(candidates[as_mother].track_id, candidates[as_baby].track_id)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denominator = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(va @ vb / denominator) if denominator else 0.0


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
