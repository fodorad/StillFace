"""Head boxes and tracks: what the gaze pipeline needs from a head detector."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


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
    """

    track_id: int
    boxes: dict[int, HeadBox]
    embedding: tuple[float, ...] | None = None


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
