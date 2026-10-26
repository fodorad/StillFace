"""Session thumbnails taken from the stillface overview video."""

import logging
from pathlib import Path

import cv2
from tqdm import tqdm

from stillface.config import DatasetPaths

logger = logging.getLogger(__name__)

OVERVIEW_VIDEO: str = "session_stillface.mp4"
"""Overview video the thumbnail is taken from."""


def middle_frame(video: Path) -> "cv2.typing.MatLike | None":
    """Read the frame in the middle of a video.

    Args:
        video: Video file.

    Returns:
        BGR image, or None if the video cannot be read.
    """
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def generate_thumbnails(paths: DatasetPaths, session_ids: list[str] | None = None) -> int:
    """Save a thumbnail for each session, per session and in a flat overview folder.

    Args:
        paths: Dataset locations.
        session_ids: Sessions to process; all sessions with processed data if None.

    Returns:
        Number of thumbnails written.
    """
    if session_ids is None:
        sessions_root = paths.processed_dir / "Sessions"
        session_ids = (
            sorted(p.name for p in sessions_root.glob("*")) if sessions_root.exists() else []
        )
    paths.thumbnails_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for session_id in tqdm(session_ids, desc="Generating thumbnails"):
        video = paths.visualize_dir(session_id) / OVERVIEW_VIDEO
        if not video.exists():
            logger.error("Video not found for session %s", session_id)
            continue
        frame = middle_frame(video)
        if frame is None:
            logger.error("Failed to read a frame for session %s", session_id)
            continue
        cv2.imwrite(str(paths.thumbnail_path(session_id)), frame)
        cv2.imwrite(str(paths.thumbnails_dir / f"{session_id}.png"), frame)
        written += 1
    return written
