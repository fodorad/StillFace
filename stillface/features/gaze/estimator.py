"""What the gaze pipeline needs from a gaze target model."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from stillface.features.gaze.heads import HeadBox


@dataclass(frozen=True)
class GazeEstimate:
    """Gaze target prediction for a batch of frames.

    Attributes:
        heatmap: Gaze target probability map per person, shape ``(frames, people, 64, 64)``,
            values in [0, 1]. People are padded to the largest count in the batch.
        inout: Probability that the target is inside the frame, shape ``(frames, people)``,
            None for models without that output.
    """

    heatmap: np.ndarray
    inout: np.ndarray | None


class GazeEstimator(Protocol):
    """Predicts where the given heads look."""

    def estimate(
        self, frames: np.ndarray, boxes: Sequence[Sequence[HeadBox | None]]
    ) -> GazeEstimate:
        """Estimate the gaze target of every head.

        Args:
            frames: RGB frames, ``uint8`` array of shape ``(frames, height, width, 3)``.
            boxes: Per frame the head box of each person, in the same person order for all
                frames. A None box yields an empty heatmap.

        Returns:
            Heatmaps and in/out scores.
        """
        ...
