"""Hand-written fakes and builders for the gaze tests."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from stillface.features.gaze.estimator import GazeEstimate
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
    score: float | None = None,
) -> HeadTrack:
    """Track with a constant box over ``frames`` and, optionally, a constant score."""
    scores = {} if score is None else {f: score for f in frames}
    return HeadTrack(track_id, {f: box(cx, width=width) for f in frames}, embedding, scores)


class FakeHeadTracker:
    """HeadTracker returning scripted tracks per clip name."""

    def __init__(self, tracks_by_clip: dict[str, list[HeadTrack]]) -> None:
        self.tracks_by_clip = tracks_by_clip
        self.calls: list[tuple[Path, list[int]]] = []

    def track(self, clip: Path, frame_indices: Sequence[int]) -> list[HeadTrack]:
        self.calls.append((clip, list(frame_indices)))
        return self.tracks_by_clip.get(clip.name, [])


def peak_heatmap(x: float, y: float, size: int = 64, sigma: float = 0.02) -> np.ndarray:
    """Gaussian heatmap with its peak at normalized ``(x, y)``, values in [0, 1]."""
    coords = (np.arange(size) + 0.5) / size
    gx = np.exp(-((coords - x) ** 2) / (2 * sigma**2))
    gy = np.exp(-((coords - y) ** 2) / (2 * sigma**2))
    return (gy[:, None] * gx[None, :]).astype(np.float32)


class FakeGazeEstimator:
    """GazeEstimator placing each person's heatmap peak at a scripted point.

    Args:
        targets: Person index (order of the boxes of a frame) mapped to the gaze point.
        inout: In/out score returned for everyone.
    """

    variant = "fake"

    def __init__(self, targets: dict[int, tuple[float, float]], inout: float = 0.9) -> None:
        self.targets = targets
        self.inout = inout
        self.calls: list[int] = []

    def estimate(self, frames: np.ndarray, boxes: Sequence[Sequence[HeadBox | None]]):
        self.calls.append(len(frames))
        n_people = max(len(b) for b in boxes)
        heatmaps = np.zeros((len(frames), n_people, 64, 64), dtype=np.float32)
        for person, (x, y) in self.targets.items():
            heatmaps[:, person] = peak_heatmap(x, y)
        return GazeEstimate(heatmaps, np.full((len(frames), n_people), self.inout, np.float32))
