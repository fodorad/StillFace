"""Choosing one head box per frame from noisy candidates."""

from dataclasses import dataclass

import numpy as np

from stillface.features.gaze.heads import HeadBox


@dataclass(frozen=True)
class Candidate:
    """One detected box on one frame.

    Attributes:
        box: Head box in normalized coordinates.
        score: Detection confidence.
    """

    box: HeadBox
    score: float


def select_path(
    frames: list[list[Candidate]], jump_penalty: float = 2.0, min_score: float = 0.0
) -> list[Candidate | None]:
    """Pick the candidate sequence with the best scores and the smoothest motion.

    A head moves little between neighboring samples, so a weaker box next to the previous
    one beats a strong box elsewhere (another person, an object). Dynamic programming over
    the candidates of all frames maximizes the sum of scores minus ``jump_penalty`` times
    the distance the box center travels (normalized image units); a frame may also be left
    empty at no gain.

    Args:
        frames: Candidates of each frame, possibly empty.
        jump_penalty: Cost per unit of center distance between consecutive chosen boxes.
        min_score: Candidates below this score are ignored.

    Returns:
        One chosen candidate per frame, None where the frame stays empty.
    """
    kept = [[c for c in cands if c.score >= min_score] for cands in frames]
    n = len(kept)
    # state per frame: candidates plus the "empty" state (index len(cands))
    best: list[np.ndarray] = []
    back: list[np.ndarray] = []
    previous_centers = np.zeros((0, 2))
    for t, cands in enumerate(kept):
        scores = np.array([c.score for c in cands] + [0.0])
        centers = np.array([c.box.center for c in cands]).reshape(-1, 2)
        if t == 0:
            best.append(scores)
            back.append(np.zeros(len(scores), dtype=int))
        else:
            previous = best[-1]
            move = np.zeros((len(scores), len(previous)))
            if len(cands) and len(previous_centers):
                dist = np.linalg.norm(centers[:, None, :] - previous_centers[None, :, :], axis=-1)
                move[: len(cands), : len(previous_centers)] = jump_penalty * dist
            total = previous[None, :] - move
            choice = total.argmax(axis=1)
            best.append(scores + total[np.arange(len(scores)), choice])
            back.append(choice)
        previous_centers = centers
    if n == 0:
        return []
    state = int(best[-1].argmax())
    path: list[Candidate | None] = [None] * n
    for t in range(n - 1, -1, -1):
        path[t] = kept[t][state] if state < len(kept[t]) else None
        state = int(back[t][state])
    return path
