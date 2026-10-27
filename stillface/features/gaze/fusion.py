"""Fusing the head tracks of two detectors into one track per head."""

from collections.abc import Sequence
from pathlib import Path

from stillface.features.gaze.heads import HeadBox, HeadTrack, HeadTracker, containment, overlaps

SECONDARY_ID_OFFSET: int = 1000
"""Ids of secondary tracks without a primary partner start here."""


def _union(a: HeadBox, b: HeadBox) -> HeadBox:
    return HeadBox(min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1))


def _same_head(a: HeadTrack, b: HeadTrack, min_containment: float, min_share: float) -> bool:
    shared = set(a.boxes) & set(b.boxes)
    if not shared:
        return False
    agree = sum(containment(a.boxes[f], b.boxes[f]) >= min_containment for f in shared)
    return agree >= min_share * len(shared)


def merge_duplicates(
    tracks: Sequence[HeadTrack], min_containment: float = 0.6, min_share: float = 0.5
) -> list[HeadTrack]:
    """Merge tracks that follow the same head (e.g. a head box and a box of its face part).

    Two tracks are duplicates if on at least ``min_share`` of their shared frames the
    smaller box lies mostly (``min_containment``) inside the other. A merged track keeps,
    per frame, the box with the higher score.

    Args:
        tracks: Head tracks of one detector.
        min_containment: Share of the smaller box inside the other.
        min_share: Share of shared frames that must agree.

    Returns:
        Tracks without duplicates; a merged track keeps the id of the longer one.
    """
    merged = [HeadTrack(t.track_id, dict(t.boxes), t.embedding, dict(t.scores)) for t in tracks]
    merged.sort(key=lambda t: -len(t.boxes))
    result: list[HeadTrack] = []
    for track in merged:
        target = next(
            (kept for kept in result if _same_head(kept, track, min_containment, min_share)), None
        )
        if target is None:
            result.append(track)
            continue
        for frame, box in track.boxes.items():
            score = track.scores.get(frame, 1.0)
            if frame not in target.boxes or score > target.scores.get(frame, 1.0):
                target.boxes[frame], target.scores[frame] = box, score
    return result


def fuse_tracks(
    primary: Sequence[HeadTrack],
    secondary: Sequence[HeadTrack],
    min_iou: float = 0.1,
    min_share: float = 0.5,
) -> list[HeadTrack]:
    """Merge the tracks of a head detector (primary) and a face detector (secondary).

    A secondary track that follows the same person as a primary track (overlapping boxes on
    most shared frames) is merged into it: on frames where both exist the box is their union
    (a face box lies inside the head box, so the head box mostly wins) and the score is the
    larger one; frames only one detector saw are taken from that detector. Secondary tracks
    without a partner are kept as additional heads.

    Args:
        primary: Head tracks.
        secondary: Face tracks.
        min_iou: Boxes overlapping at least this much belong to the same person.
        min_share: Share of shared frames that must overlap.

    Returns:
        Fused tracks; ids of unmatched secondary tracks are offset by
        :data:`SECONDARY_ID_OFFSET`.
    """
    merged: list[HeadTrack] = []
    used: set[int] = set()
    for head in primary:
        boxes, scores = dict(head.boxes), dict(head.scores)
        for face in secondary:
            if face.track_id in used or not overlaps(head, face, min_iou, min_share):
                continue
            used.add(face.track_id)
            for frame, box in face.boxes.items():
                score = face.scores.get(frame, 1.0)
                if frame in boxes:
                    boxes[frame] = _union(boxes[frame], box)
                    scores[frame] = max(scores.get(frame, 1.0), score)
                else:
                    boxes[frame], scores[frame] = box, score
        merged.append(HeadTrack(head.track_id, boxes, head.embedding, scores))
    for face in secondary:
        if face.track_id not in used:
            merged.append(
                HeadTrack(SECONDARY_ID_OFFSET + face.track_id, dict(face.boxes),
                          face.embedding, dict(face.scores))
            )  # fmt: skip
    return merged


class FusedHeadTracker:
    """Runs a head detector and a face detector on a clip and fuses their tracks.

    Args:
        primary: Head tracker (text-prompted head detector).
        secondary: Face tracker.
    """

    def __init__(self, primary: HeadTracker, secondary: HeadTracker) -> None:
        self.primary = primary
        self.secondary = secondary

    def track(self, clip: Path, frame_indices: Sequence[int]) -> list[HeadTrack]:
        """Track heads with both detectors and fuse the tracks.

        Args:
            clip: Video clip.
            frame_indices: Frames to analyze, ascending.

        Returns:
            Fused head tracks; duplicate head tracks are merged first.
        """
        return fuse_tracks(
            merge_duplicates(self.primary.track(clip, frame_indices)),
            self.secondary.track(clip, frame_indices),
        )
