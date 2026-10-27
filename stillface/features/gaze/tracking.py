"""Linking per-frame head candidates into tracks."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from stillface.features.gaze.heads import HeadBox, HeadTrack, interpolate_boxes


@dataclass(frozen=True)
class Candidate:
    """One detected box on one frame.

    Attributes:
        box: Head box in normalized coordinates.
        score: Detection confidence.
    """

    box: HeadBox
    score: float


def _size(box: HeadBox) -> float:
    return max(box.x1 - box.x0, box.y1 - box.y0)


def link_candidates(
    candidates: list[list[Candidate]],
    frames: list[int],
    max_distance: float = 1.5,
    max_missed: int = 2,
) -> list[HeadTrack]:
    """Link the candidates of consecutive frames into tracks.

    Each frame's candidates are matched to the open tracks by the Hungarian algorithm on the
    distance between box centers, measured in head sizes (a head moves less than about one
    and a half of its own size between neighboring samples). Unmatched candidates start new
    tracks; a track is closed after ``max_missed`` consecutive frames without a match.

    Args:
        candidates: Candidates of each frame.
        frames: Frame index of each entry of ``candidates``.
        max_distance: Largest accepted center distance, in head sizes.
        max_missed: Frames a track may stay unmatched before it is closed.

    Returns:
        One track per head, with boxes and scores on the frames it was matched.
    """
    boxes: list[dict[int, HeadBox]] = []
    scores: list[dict[int, float]] = []
    missed: list[int] = []
    open_ids: list[int] = []
    for frame, cands in zip(frames, candidates, strict=True):
        matched: set[int] = set()
        matched_rows: set[int] = set()
        if open_ids and cands:
            last = [boxes[i][max(boxes[i])] for i in open_ids]
            cost = np.array(
                [
                    [
                        np.linalg.norm(np.subtract(c.box.center, b.center))
                        / max(_size(b), _size(c.box), 1e-6)
                        for c in cands
                    ]
                    for b in last
                ]
            )
            for row, col in zip(*linear_sum_assignment(cost), strict=True):
                if cost[row, col] <= max_distance:
                    i = open_ids[row]
                    boxes[i][frame] = cands[col].box
                    scores[i][frame] = cands[col].score
                    missed[i] = 0
                    matched.add(col)
                    matched_rows.add(row)
        for row, i in enumerate(open_ids):
            if row not in matched_rows:
                missed[i] += 1
        open_ids = [i for i in open_ids if missed[i] <= max_missed]
        for col, cand in enumerate(cands):
            if col not in matched:
                boxes.append({frame: cand.box})
                scores.append({frame: cand.score})
                missed.append(0)
                open_ids.append(len(boxes) - 1)
    return [HeadTrack(i, b, scores=s) for i, (b, s) in enumerate(zip(boxes, scores, strict=True))]


def stitch_tracks(
    tracks: list[HeadTrack], max_gap: int, max_distance: float = 1.5
) -> list[HeadTrack]:
    """Join track pieces of the same head that a detection dropout split.

    A track that starts after another one ended, within ``max_gap`` frames and close to
    where the other ended (in head sizes), continues it. Pieces are joined greedily, the
    closest pair first.

    Args:
        tracks: Tracks from :func:`link_candidates`.
        max_gap: Largest gap in frames between the end of one piece and the start of the next.
        max_distance: Largest center distance between the two ends, in head sizes.

    Returns:
        The stitched tracks; a stitched track keeps the id of its first piece.
    """
    pieces = [
        HeadTrack(t.track_id, dict(t.boxes), t.embedding, dict(t.scores), dict(t.faces))
        for t in tracks
        if t.boxes
    ]
    while True:
        best: tuple[float, int, int] | None = None
        for i, a in enumerate(pieces):
            end = max(a.boxes)
            for j, b in enumerate(pieces):
                start = min(b.boxes)
                if i == j or not 0 < start - end <= max_gap:
                    continue
                last, first = a.boxes[end], b.boxes[start]
                distance = float(np.hypot(*np.subtract(last.center, first.center))) / max(
                    _size(last), _size(first), 1e-6
                )
                if distance <= max_distance and (best is None or distance < best[0]):
                    best = (distance, i, j)
        if best is None:
            return pieces
        _, i, j = best
        pieces[i].boxes.update(pieces[j].boxes)
        pieces[i].scores.update(pieces[j].scores)
        pieces[i].faces.update(pieces[j].faces)
        del pieces[j]


def densify(track: HeadTrack, frame_indices: list[int], max_gap: int) -> HeadTrack:
    """Fill a track between its detected frames by interpolation.

    Args:
        track: Track detected on a sparse set of frames.
        frame_indices: Frames to produce boxes for.
        max_gap: Longest gap in frames that is bridged.

    Returns:
        The track on the requested frames; interpolated frames carry the lower score of
        the two neighboring detections.
    """
    boxes = interpolate_boxes(track.boxes, frame_indices, max_gap)
    detected = sorted(track.scores)
    scores = {}
    for frame in boxes:
        if frame in track.scores:
            scores[frame] = track.scores[frame]
        else:
            right = int(np.searchsorted(detected, frame))
            scores[frame] = min(track.scores[detected[right - 1]], track.scores[detected[right]])
    faces = {frame: score for frame, score in track.faces.items() if frame in boxes}
    return HeadTrack(track.track_id, boxes, track.embedding, scores, faces)


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
