"""Combining head trackers: text-prompted heads plus face detections, per camera."""

import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from stillface.features.gaze.heads import HeadBox, HeadTrack, HeadTracker

logger = logging.getLogger(__name__)

TAGGED_ID_OFFSET: int = 1000
"""Track ids of role-tagged tracks start here so they never collide with anonymous ones."""


def iou(a: HeadBox, b: HeadBox) -> float:
    """Intersection over union of two boxes."""
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / (a.area + b.area - inter)


def overlaps(track: HeadTrack, other: HeadTrack, min_iou: float, min_share: float) -> bool:
    """Whether two tracks mostly follow the same head.

    Args:
        track: First track.
        other: Second track.
        min_iou: Boxes of the same frame with at least this IoU count as the same head.
        min_share: Share of the shared frames that must agree.

    Returns:
        True if the tracks share frames and agree on at least ``min_share`` of them.
    """
    shared = set(track.boxes) & set(other.boxes)
    if not shared:
        return False
    same = sum(iou(track.boxes[f], other.boxes[f]) >= min_iou for f in shared)
    return same / len(shared) >= min_share


class CombinedHeadTracker:
    """Tracks of a role-tagged tracker plus the face tracks that are not the same head.

    Args:
        tagged: Tracker returning role-tagged tracks (text-prompted detector).
        faces: Tracker returning anonymous face tracks.
        min_iou: Overlap above which a face track duplicates a tagged track.
        min_share: Share of shared frames that must overlap to drop a face track.
    """

    def __init__(
        self, tagged: HeadTracker, faces: HeadTracker, min_iou: float = 0.3, min_share: float = 0.5
    ) -> None:
        self.tagged = tagged
        self.faces = faces
        self.min_iou = min_iou
        self.min_share = min_share

    def track(self, clip: Path, frame_indices: Sequence[int]) -> list[HeadTrack]:
        """Track heads with both trackers and merge the results.

        Args:
            clip: Video clip.
            frame_indices: Frames to analyze, ascending.

        Returns:
            Tagged tracks (ids offset by :data:`TAGGED_ID_OFFSET`) and the face tracks that
            do not duplicate one of them.
        """
        tagged = [
            replace(t, track_id=TAGGED_ID_OFFSET + t.track_id)
            for t in self.tagged.track(clip, frame_indices)
        ]
        faces = [
            t
            for t in self.faces.track(clip, frame_indices)
            if not any(overlaps(t, other, self.min_iou, self.min_share) for other in tagged)
        ]
        return [*tagged, *faces]


class PerCameraTracker:
    """Routes a clip to the tracker of its camera, read from the ``<camera>_<phase>`` name.

    Args:
        side: Tracker for the side cameras, which see both participants.
        close: Tracker for the close cameras, which see one participant.
    """

    SIDE: tuple[str, ...] = ("window", "door")

    def __init__(self, side: HeadTracker, close: HeadTracker) -> None:
        self.side = side
        self.close = close

    def track(self, clip: Path, frame_indices: Sequence[int]) -> list[HeadTrack]:
        """Track heads with the tracker of the clip's camera.

        Args:
            clip: Clip named ``<camera>_<phase>.mp4``.
            frame_indices: Frames to analyze, ascending.

        Returns:
            Head tracks.
        """
        camera = clip.stem.split("_")[0]
        return (self.side if camera in self.SIDE else self.close).track(clip, frame_indices)


def anchor_frames(frame_indices: Sequence[int], clip_fps: float, rate_hz: float) -> list[int]:
    """Choose the requested frames closest to a regular, slower time grid.

    Args:
        frame_indices: Ascending frames that will be asked for.
        clip_fps: Frame rate of the clip.
        rate_hz: Rate of the anchor grid, at most the clip rate.

    Returns:
        Unique anchor frames out of ``frame_indices``; empty if there are none.
    """
    if not len(frame_indices):
        return []
    frames = np.asarray(frame_indices)
    targets = np.arange(frames[0], frames[-1] + 1, clip_fps / rate_hz)
    nearest = np.abs(frames[None, :] - targets[:, None]).argmin(axis=1)
    return frames[np.unique(nearest)].tolist()
