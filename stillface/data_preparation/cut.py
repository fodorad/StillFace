"""Cut the synchronized camera videos into experiment phases."""

import logging
from pathlib import Path

from tqdm import tqdm

from stillface.config import DatasetPaths
from stillface.constants import CAMERAS, PHASES
from stillface.data_preparation.metadata import (
    PhaseWindows,
    load_metadata,
    phase_windows,
    sessions_to_cut,
)
from stillface.data_preparation.video import run_ffmpeg, to_seconds, video_duration
from stillface.data_preparation.visualize import create_stacked_videos

logger = logging.getLogger(__name__)


def cut_clip(src: Path, dst: Path, start: str, end: str) -> bool:
    """Cut ``[start, end]`` out of a video without re-encoding.

    Args:
        src: Input video.
        dst: Output video. An existing file is kept.
        start: Start time in ``MM:SS`` format.
        end: End time in ``MM:SS`` format.

    Returns:
        True if the clip exists afterwards.
    """
    if not src.exists():
        logger.warning("Input video not found: %s", src)
        return False
    if dst.exists():
        logger.info("Clip already exists: %s", dst)
        return True
    try:
        start_s, end_s = to_seconds(start), to_seconds(end)
    except ValueError:
        logger.warning("Start or end time is missing: %s to %s", start, end)
        return False
    if start_s >= video_duration(src):
        logger.warning("Start time %ss is outside of %s", start_s, src.name)
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Cutting %s -> %s (%s to %s)", src.name, dst.name, start, end)
    return run_ffmpeg(
        ["-ss", start, "-i", str(src), "-t", str(end_s - start_s), "-c", "copy", str(dst)]
    )


def cut_all_phases(
    synced_dir: Path, windows: PhaseWindows, output_dir: Path
) -> dict[str, dict[str, Path]]:
    """Cut every synced camera into the experiment phases.

    Args:
        synced_dir: Folder with ``<camera>.mp4`` files.
        windows: Phase boundaries.
        output_dir: Folder receiving ``<camera>_<phase>.mp4`` clips.

    Returns:
        Camera name mapped to phase mapped to clip path, for the clips that exist.
    """
    clips: dict[str, dict[str, Path]] = {}
    for camera in CAMERAS:
        src = synced_dir / f"{camera}.mp4"
        if not src.exists():
            logger.info("Skipping %s, video not found: %s", camera, src)
            continue
        clips[camera] = {}
        for phase in PHASES:
            start, end = windows.windows[phase]
            dst = output_dir / f"{camera}_{phase}.mp4"
            if cut_clip(src, dst, start, end):
                clips[camera][phase] = dst
    return clips


def cut(
    paths: DatasetPaths, session_id: str, windows: PhaseWindows, visualize: bool = False
) -> dict[str, dict[str, Path]]:
    """Cut one session into phases and optionally render the stacked overviews.

    Args:
        paths: Dataset locations.
        session_id: Session identifier.
        windows: Phase boundaries of the session.
        visualize: Also create the stacked overview videos.

    Returns:
        Camera name mapped to phase mapped to clip path.
    """
    logger.info("Cutting session %s", session_id)
    clips = cut_all_phases(paths.synced_dir(session_id), windows, paths.cut_dir(session_id))
    if visualize:
        create_stacked_videos(clips, paths.visualize_dir(session_id))
    return clips


def cut_all(paths: DatasetPaths, visualize: bool = False) -> None:
    """Cut every automatically processable session listed in the metadata.

    Failures are logged per session and recorded in ``logs/failed_cut_sessions.txt``.

    Args:
        paths: Dataset locations.
        visualize: Also create the stacked overview videos.
    """
    metadata = load_metadata(paths.metadata_path)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    for session_id in tqdm(sessions_to_cut(metadata), desc="Cutting videos"):
        try:
            cut(paths, session_id, phase_windows(metadata, session_id), visualize)
        except Exception:
            logger.exception("Cutting failed for session %s", session_id)
            _append(paths.logs_dir / "failed_cut_sessions.txt", session_id)
        else:
            _append(paths.logs_dir / "cut_sessions.txt", session_id)


def _append(path: Path, line: str) -> None:
    with path.open("a") as handle:
        handle.write(f"{line}\n")
