"""Synchronize the camera videos of a session.

The mother and baby cameras are aligned to each other with :mod:`avst`; the window and
door cameras are aligned to the resulting reference. Originals are never modified, every
product is written below the processed dataset root.
"""

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from avst.io import get_video_fps
from avst.sync import sync_to_reference, sync_videos
from tqdm import tqdm

from stillface.config import DatasetPaths
from stillface.constants import CAMERAS, TARGET_FPS
from stillface.data_preparation.metadata import load_metadata, sessions_to_sync
from stillface.data_preparation.video import convert_to_fps, run_ffmpeg

logger = logging.getLogger(__name__)

SOURCE_SUFFIXES: dict[str, tuple[str, ...]] = {
    "mother": (".mp4",),
    "baby": (".mp4",),
    "window": (".MTS", ".mp4"),
    "door": (".MTS", ".mp4"),
}
"""Accepted file extensions of each camera's original recording, by preference."""

REFERENCE_PRIORITY: tuple[str, ...] = ("baby", "mother", "window")
"""Camera order used as the alignment reference of the auxiliary cameras."""

AUXILIARY_CAMERAS: tuple[str, ...] = ("window", "door")
"""Cameras aligned to the mother-baby reference instead of to each other."""


@dataclass
class SyncResult:
    """Outcome of synchronizing one session.

    Attributes:
        videos: Camera name mapped to the synced video path.
        offset_ms: Mother-baby offset in milliseconds, None if not computed in this run.
    """

    videos: dict[str, Path] = field(default_factory=dict)
    offset_ms: int | None = None


def find_cameras(original_dir: Path) -> dict[str, Path]:
    """Detect the available original recordings of a session.

    Args:
        original_dir: Folder with the original recordings.

    Returns:
        Camera name mapped to its source path, for the cameras that exist.
    """
    found: dict[str, Path] = {}
    for camera in CAMERAS:
        for suffix in SOURCE_SUFFIXES[camera]:
            candidate = original_dir / f"{camera}{suffix}"
            if candidate.exists():
                found[camera] = candidate
                break
    return found


def normalize(cameras: dict[str, Path], normalized_dir: Path) -> dict[str, Path]:
    """Bring the window and door recordings to ``.mp4`` at the target frame rate.

    Args:
        cameras: Output of :func:`find_cameras`.
        normalized_dir: Folder receiving the converted files.

    Returns:
        The camera mapping, with auxiliary cameras pointing to the converted files.

    Raises:
        RuntimeError: If a conversion fails.
    """
    prepared = dict(cameras)
    for camera in AUXILIARY_CAMERAS:
        if camera not in cameras:
            continue
        src, dst = cameras[camera], normalized_dir / f"{camera}.mp4"
        if abs(get_video_fps(src) - TARGET_FPS) > 0.01:
            prepared[camera] = convert_to_fps(src, dst, TARGET_FPS)
        elif dst.exists() or _remux(src, dst):
            prepared[camera] = dst
        else:
            raise RuntimeError(f"Failed to remux {src}")
    return prepared


def _remux(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    return run_ffmpeg(["-i", str(src), "-c", "copy", str(dst)])


def _copy(src: Path, dst: Path) -> Path:
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
    return dst


def sync_mother_baby(
    mother: Path | None, baby: Path | None, synced_dir: Path, visualize: bool = False
) -> SyncResult:
    """Align the mother and baby videos, or pass a single available one through.

    Args:
        mother: Mother camera source, if recorded.
        baby: Baby camera source, if recorded.
        synced_dir: Output folder.
        visualize: Also write a stacked check video of the aligned pair.

    Returns:
        Synced videos and the offset when both cameras exist and were aligned.
    """
    result = SyncResult()
    out_mother, out_baby = synced_dir / "mother.mp4", synced_dir / "baby.mp4"
    if mother is not None and baby is not None:
        logger.info("Syncing mother and baby videos")
        synced_dir.mkdir(parents=True, exist_ok=True)
        if not (out_mother.exists() and out_baby.exists()):
            result.offset_ms = sync_videos(
                video1_path=mother,
                video2_path=baby,
                output1_path=out_mother,
                output2_path=out_baby,
                synced_session_path=synced_dir / "session_mb.mp4" if visualize else None,
            )
        result.videos = {"mother": out_mother, "baby": out_baby}
    elif baby is not None:
        logger.info("Copying baby video, no mother video available")
        result.videos["baby"] = _copy(baby, out_baby)
    elif mother is not None:
        logger.info("Copying mother video, no baby video available")
        result.videos["mother"] = _copy(mother, out_mother)
    return result


def sync_auxiliary(
    camera: str,
    source: Path,
    reference: Path | None,
    synced_dir: Path,
    visualize: bool = False,
) -> Path:
    """Align an auxiliary camera to the reference video.

    Args:
        camera: Camera name, ``window`` or ``door``.
        source: Normalized camera source.
        reference: Reference video, or None if the camera is the only one.
        synced_dir: Output folder.
        visualize: Also write a stacked check video.

    Returns:
        Path of the synced video.
    """
    dst = synced_dir / f"{camera}.mp4"
    if reference is None or source == reference:
        logger.info("Copying %s video, no reference available", camera)
        return _copy(source, dst)
    logger.info("Syncing %s to the reference video", camera)
    if not dst.exists():
        synced_dir.mkdir(parents=True, exist_ok=True)
        sync_to_reference(
            video1_path=reference,
            video2_path=source,
            output2_path=dst,
            synced_session_path=synced_dir / f"session_{reference.stem}_{camera}.mp4"
            if visualize
            else None,
        )
    return dst


def pick_reference(synced: dict[str, Path], cameras: dict[str, Path]) -> Path | None:
    """Choose the video the auxiliary cameras are aligned to.

    Args:
        synced: Cameras synced so far.
        cameras: Normalized sources of all available cameras.

    Returns:
        The synced baby, mother or window video by :data:`REFERENCE_PRIORITY`; the
        unsynced window source if it is the only camera so far; otherwise None.
    """
    for name in REFERENCE_PRIORITY:
        if name in synced:
            return synced[name]
    return cameras.get("window")


def sync(paths: DatasetPaths, session_id: str, visualize: bool = False) -> SyncResult:
    """Synchronize all available cameras of a session.

    Args:
        paths: Dataset locations.
        session_id: Session identifier.
        visualize: Also write stacked check videos.

    Returns:
        Synced videos and the mother-baby offset. Empty if no camera was recorded.
    """
    logger.info("Syncing session %s", session_id)
    cameras = find_cameras(paths.original_dir(session_id))
    if not cameras:
        logger.warning("No cameras available for session %s", session_id)
        return SyncResult()
    logger.info("Available cameras: %s", ", ".join(cameras))

    cameras = normalize(cameras, paths.normalized_dir(session_id))
    synced_dir = paths.synced_dir(session_id)
    result = sync_mother_baby(cameras.get("mother"), cameras.get("baby"), synced_dir, visualize)

    for camera in AUXILIARY_CAMERAS:
        if camera not in cameras:
            continue
        reference = pick_reference(result.videos, cameras)
        result.videos[camera] = sync_auxiliary(
            camera, cameras[camera], reference, synced_dir, visualize
        )

    logger.info("Synced cameras: %s", ", ".join(result.videos))
    return result


def sync_all(paths: DatasetPaths, visualize: bool = False) -> None:
    """Synchronize every pending, automatically processable session.

    Failures are logged per session and recorded in ``logs/failed_sessions.txt``; offsets
    of successful sessions go to ``logs/synced_sessions.txt`` for review.

    Args:
        paths: Dataset locations.
        visualize: Also write stacked check videos.
    """
    metadata = load_metadata(paths.metadata_path)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    for session_id in tqdm(sessions_to_sync(metadata), desc="Syncing sessions"):
        try:
            result = sync(paths, session_id, visualize)
        except Exception:
            logger.exception("Sync failed for session %s", session_id)
            _append(paths.logs_dir / "failed_sessions.txt", session_id)
        else:
            _append(paths.logs_dir / "synced_sessions.txt", f"{session_id},{result.offset_ms}")


def _append(path: Path, line: str) -> None:
    with path.open("a") as handle:
        handle.write(f"{line}\n")
