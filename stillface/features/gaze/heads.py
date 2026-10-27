"""Head boxes and tracks: what the gaze pipeline needs from a head detector."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class HeadBox:
    """Axis-aligned head box in normalized image coordinates.

    Attributes:
        x0: Left edge.
        y0: Top edge.
        x1: Right edge.
        y1: Bottom edge.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def area(self) -> float:
        """Box area as a share of the image."""
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    @property
    def center(self) -> tuple[float, float]:
        """Box center ``(x, y)``."""
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2

    def expand(self, margin: float) -> "HeadBox":
        """Grow the box on every side by a share of its size, clipped to the image.

        Args:
            margin: Relative margin, 0.25 adds a quarter of the width and height per side.

        Returns:
            The enlarged box.
        """
        dx, dy = (self.x1 - self.x0) * margin, (self.y1 - self.y0) * margin
        return HeadBox(
            max(0.0, self.x0 - dx),
            max(0.0, self.y0 - dy),
            min(1.0, self.x1 + dx),
            min(1.0, self.y1 + dy),
        )


@dataclass(frozen=True)
class HeadTrack:
    """One head followed across the sampled frames of a clip.

    Attributes:
        track_id: Identifier of the track within the clip.
        boxes: Frame index mapped to the head box, for the frames the head was found in.
        embedding: Mean face-identity embedding of the track, None if not computed.
        scores: Frame index mapped to the detection confidence; frames without a score count
            as fully confident.
    """

    track_id: int
    boxes: dict[int, HeadBox]
    embedding: tuple[float, ...] | None = None
    scores: dict[int, float] = field(default_factory=dict)

    def evidence(self, centrality: bool = False) -> float:
        """Confidence-weighted presence: the sum of the detection scores over the frames.

        Args:
            centrality: Also weight each frame by how close the box center is to the image
                center (1 at the center, 0 at the edge midpoints and beyond).

        Returns:
            The weighted sum.
        """
        total = 0.0
        for frame, box in self.boxes.items():
            weight = self.scores.get(frame, 1.0)
            if centrality:
                cx, cy = box.center
                weight *= max(0.0, 1.0 - 2.0 * float(np.hypot(cx - 0.5, cy - 0.5)))
            total += weight
        return total

    def median_score(self) -> float:
        """Median detection score of the track (frames without a score count as 1)."""
        if not self.boxes:
            return 0.0
        return float(np.median([self.scores.get(frame, 1.0) for frame in self.boxes]))

    def median_area(self) -> float:
        """Median box area of the track, 0 for an empty track."""
        return float(np.median([b.area for b in self.boxes.values()])) if self.boxes else 0.0

    def weight(self) -> float:
        """Confidence-weighted head size: the sum of box area times score over the frames."""
        return sum(box.area * self.scores.get(frame, 1.0) for frame, box in self.boxes.items())


class HeadTracker(Protocol):
    """Finds and tracks the heads in a clip."""

    def track(self, clip: Path, frame_indices: Sequence[int]) -> list[HeadTrack]:
        """Track all heads in the given frames of a clip.

        Args:
            clip: Video clip.
            frame_indices: Frames to analyze, ascending.

        Returns:
            One track per head, with boxes only on the requested frames.
        """
        ...


def interpolate_boxes(
    anchors: dict[int, HeadBox], frame_indices: Sequence[int], max_gap: int
) -> dict[int, HeadBox]:
    """Fill the frames between detected anchor frames by linear interpolation.

    Frames before the first or after the last anchor, and frames whose surrounding anchors
    are further apart than ``max_gap`` frames, stay empty.

    Args:
        anchors: Frame index mapped to the detected box.
        frame_indices: Frames to produce boxes for.
        max_gap: Largest distance in frames between two anchors that is still bridged.

    Returns:
        Frame index mapped to a box, for every requested frame that could be filled.
    """
    if not anchors:
        return {}
    frames = np.array(sorted(anchors))
    coords = np.array([[b.x0, b.y0, b.x1, b.y1] for _, b in sorted(anchors.items())])
    boxes: dict[int, HeadBox] = {}
    for frame in frame_indices:
        if frame in anchors:
            boxes[frame] = anchors[frame]
            continue
        right = int(np.searchsorted(frames, frame))
        if right == 0 or right == len(frames) or frames[right] - frames[right - 1] > max_gap:
            continue
        weight = (frame - frames[right - 1]) / (frames[right] - frames[right - 1])
        x0, y0, x1, y1 = (1 - weight) * coords[right - 1] + weight * coords[right]
        boxes[frame] = HeadBox(float(x0), float(y0), float(x1), float(y1))
    return boxes


def iou(a: HeadBox, b: HeadBox) -> float:
    """Intersection over union of two boxes."""
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / (a.area + b.area - inter)


def containment(a: HeadBox, b: HeadBox) -> float:
    """Share of the smaller box covered by the other box."""
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / min(a.area, b.area)


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
