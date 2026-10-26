"""Thin ffmpeg/ffprobe helpers."""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def run_ffmpeg(args: list[str]) -> bool:
    """Run ffmpeg quietly, overwriting outputs.

    Args:
        args: Arguments following the ``ffmpeg`` executable.

    Returns:
        True if ffmpeg exited successfully.
    """
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("ffmpeg failed: %s", result.stderr.strip())
    return result.returncode == 0


def video_duration(path: Path) -> float:
    """Return the container duration of a media file.

    Args:
        path: Media file.

    Returns:
        Duration in seconds, 0.0 if it cannot be determined.
    """
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True,
        text=True,
    )  # fmt: skip
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def to_seconds(timestamp: str) -> int:
    """Convert ``MM:SS`` or ``HH:MM:SS`` to seconds.

    Args:
        timestamp: Time string.

    Returns:
        Seconds.

    Raises:
        ValueError: If the string is not a valid timestamp.
    """
    parts = [int(part) for part in timestamp.split(":")]
    if len(parts) not in (2, 3):
        raise ValueError(f"Invalid timestamp: {timestamp!r}")
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def convert_to_fps(src: Path, dst: Path, fps: int) -> Path:
    """Re-encode a video to a fixed frame rate, skipping existing output.

    Args:
        src: Input video.
        dst: Output ``.mp4`` path.
        fps: Target frame rate.

    Returns:
        ``dst``.

    Raises:
        RuntimeError: If the conversion fails.
    """
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not run_ffmpeg(["-i", str(src), "-r", str(fps), str(dst)]):
        raise RuntimeError(f"Failed to convert {src} to {fps} fps")
    return dst
