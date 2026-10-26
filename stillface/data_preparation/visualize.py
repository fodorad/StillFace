"""Stacked overview videos of the camera views."""

import logging
from pathlib import Path

from avst.io import get_video_fps, get_video_frame_size

from stillface.constants import CAMERAS, PHASES
from stillface.data_preparation.video import run_ffmpeg, video_duration

logger = logging.getLogger(__name__)

GRID_ORDER: tuple[str, ...] = ("mother", "window", "baby", "door")
"""Camera layout of the 2x2 grid: top-left, top-right, bottom-left, bottom-right."""

AUDIO_PRIORITY: tuple[str, ...] = ("baby", "mother", "window", "door")
"""Fallback order when the requested audio source is unavailable."""


def placeholder(reference: Path) -> str:
    """Build the lavfi source standing in for a missing camera.

    Args:
        reference: Present video whose size, frame rate and length are matched.

    Returns:
        ffmpeg ``lavfi`` input description of a black video.
    """
    width, height = get_video_frame_size(reference)
    return (
        f"color=black:size={width}x{height}:rate={get_video_fps(reference)}"
        f":duration={video_duration(reference)}"
    )


def stack_vertical(top: Path, bottom: Path, dst: Path, audio_source: int = 1) -> bool:
    """Stack two videos on top of each other.

    Args:
        top: Video placed on top.
        bottom: Video placed below.
        dst: Output video. An existing file is kept.
        audio_source: 0 to keep the audio of ``top``, 1 for ``bottom``.

    Returns:
        True if the output exists afterwards.
    """
    if not top.exists() or not bottom.exists():
        logger.warning("Cannot stack, missing video(s): %s, %s", top.name, bottom.name)
        return False
    if dst.exists():
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Stacking %s + %s -> %s", top.name, bottom.name, dst.name)
    return run_ffmpeg([
        "-i", str(top), "-i", str(bottom),
        "-filter_complex", "[0:v][1:v]vstack=inputs=2[v]",
        "-map", "[v]", "-map", f"{audio_source}:a",
        str(dst),
    ])  # fmt: skip


def stack_grid(videos: dict[str, Path | None], dst: Path, audio_source: str = "baby") -> bool:
    """Arrange the cameras in a 2x2 grid, replacing missing ones with black.

    Args:
        videos: Camera name mapped to video path, or None if unavailable.
        dst: Output video. An existing file is kept.
        audio_source: Camera providing the audio; falls back by :data:`AUDIO_PRIORITY`.

    Returns:
        True if the output exists afterwards.
    """
    present = {name: path for name, path in videos.items() if path is not None and path.exists()}
    if not present:
        logger.warning("No videos available for the 2x2 grid")
        return False
    if dst.exists():
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)

    if audio_source not in present:
        audio_source = next(name for name in AUDIO_PRIORITY if name in present)
    black = placeholder(present[audio_source])
    inputs: list[str] = []
    for name in GRID_ORDER:
        inputs += ["-i", str(present[name])] if name in present else ["-f", "lavfi", "-i", black]

    filter_complex = (
        "[0:v][1:v]hstack=inputs=2[top];[2:v][3:v]hstack=inputs=2[bottom];"
        "[top][bottom]vstack=inputs=2[v]"
    )
    logger.info("Creating 2x2 grid -> %s (audio from %s)", dst.name, audio_source)
    return run_ffmpeg([
        *inputs, "-filter_complex", filter_complex,
        "-map", "[v]", "-map", f"{GRID_ORDER.index(audio_source)}:a?",
        "-shortest", str(dst),
    ])  # fmt: skip


def create_stacked_videos(clips: dict[str, dict[str, Path]], output_dir: Path) -> None:
    """Create the mother-baby stack and the 2x2 grid of every phase.

    Args:
        clips: Camera name mapped to phase mapped to clip path.
        output_dir: Folder receiving ``mother-baby_<phase>.mp4`` and ``session_<phase>.mp4``.
    """
    for phase in PHASES:
        mother = clips.get("mother", {}).get(phase)
        baby = clips.get("baby", {}).get(phase)
        if mother is not None and baby is not None:
            stack_vertical(mother, baby, output_dir / f"mother-baby_{phase}.mp4")
        else:
            logger.info("Skipping mother-baby for %s, video(s) not available", phase)
        grid = {camera: clips.get(camera, {}).get(phase) for camera in CAMERAS}
        stack_grid(grid, output_dir / f"session_{phase}.mp4")


def find_cut_videos(cut_dir: Path) -> dict[str, dict[str, Path]]:
    """Collect the existing phase clips of a session.

    Args:
        cut_dir: Folder with ``<camera>_<phase>.mp4`` clips.

    Returns:
        Camera name mapped to phase mapped to clip path.
    """
    clips: dict[str, dict[str, Path]] = {camera: {} for camera in CAMERAS}
    for camera in CAMERAS:
        for phase in PHASES:
            clip = cut_dir / f"{camera}_{phase}.mp4"
            if clip.exists():
                clips[camera][phase] = clip
            else:
                logger.warning("Cut video not found: %s", clip)
    return clips
