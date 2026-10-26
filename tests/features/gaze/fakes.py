"""Hand-written fakes and builders for the gaze tests."""

from collections.abc import Sequence
from pathlib import Path

from stillface.features.gaze.heads import HeadBox, HeadTrack


def box(cx: float, cy: float = 0.4, width: float = 0.1, height: float | None = None) -> HeadBox:
    """Normalized head box centered on ``(cx, cy)``."""
    height = width * 1.2 if height is None else height
    return HeadBox(cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)


def track(
    track_id: int,
    cx: float,
    width: float = 0.1,
    frames: Sequence[int] = tuple(range(10)),
    embedding: tuple[float, ...] | None = None,
) -> HeadTrack:
    """Track with a constant box over ``frames``."""
    return HeadTrack(track_id, {f: box(cx, width=width) for f in frames}, embedding)


class FakeHeadTracker:
    """HeadTracker returning scripted tracks per clip name."""

    def __init__(self, tracks_by_clip: dict[str, list[HeadTrack]]) -> None:
        self.tracks_by_clip = tracks_by_clip
        self.calls: list[tuple[Path, list[int]]] = []

    def track(self, clip: Path, frame_indices: Sequence[int]) -> list[HeadTrack]:
        self.calls.append((clip, list(frame_indices)))
        return self.tracks_by_clip.get(clip.name, [])
