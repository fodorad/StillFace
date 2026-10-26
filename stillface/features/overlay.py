"""Overlay videos: extracted features drawn onto the camera clips.

An overlay clip has the resolution, frame rate and frame count of its source clip, drops
the audio, and shows the annotations of the last sampled frame until the next one
(sample and hold). The overlays of the four cameras can be stacked into the same 2x2 grid
as the overview videos.
"""

import logging
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from stillface.data_preparation.visualize import stack_grid
from stillface.features.gaze.heads import HeadBox
from stillface.features.io import is_finished, write_atomic

logger = logging.getLogger(__name__)

PERSON_COLORS: dict[str, tuple[int, int, int]] = {
    "mother": (191, 212, 45),
    "baby": (60, 140, 245),
}
"""BGR drawing color of each person."""

NOT_CODABLE_COLOR: tuple[int, int, int] = (160, 160, 160)
"""BGR color of rows that cannot be coded."""

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_ROW_HEIGHT = 18


@dataclass(frozen=True)
class OverlayItem:
    """What is drawn for one person on one frame.

    Attributes:
        person: ``mother`` or ``baby``.
        box: Head box in normalized coordinates, None if no head was found.
        gaze: Gaze target in normalized coordinates, None if there is none (pilot mode or
            not codable).
        label: Text after the role, e.g. the gaze class; labels starting with ``NK`` are
            drawn as not codable.
    """

    person: str
    box: HeadBox | None
    gaze: tuple[float, float] | None
    label: str


def draw_items(frame: np.ndarray, items: Sequence[OverlayItem]) -> None:
    """Draw annotations onto a BGR frame, in place.

    Args:
        frame: Image of shape ``(height, width, 3)``.
        items: One item per person to draw.
    """
    height, width = frame.shape[:2]
    for row, item in enumerate(items):
        not_codable = item.label.startswith("NK")
        color = NOT_CODABLE_COLOR if not_codable else PERSON_COLORS[item.person]
        text = f"{item.person} {item.label}".strip()
        if item.box is None:
            origin = (4, _ROW_HEIGHT * (row + 1))
        else:
            x0, y0 = round(item.box.x0 * width), round(item.box.y0 * height)
            x1, y1 = round(item.box.x1 * width), round(item.box.y1 * height)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            origin = (x0, max(y0 - 6, _ROW_HEIGHT // 2))
            if item.gaze is not None:
                gx, gy = round(item.gaze[0] * width), round(item.gaze[1] * height)
                center = (round((x0 + x1) / 2), round((y0 + y1) / 2))
                cv2.line(frame, center, (gx, gy), color, 1)
                cv2.circle(frame, (gx, gy), 5, color, -1)
        cv2.putText(frame, text, origin, _FONT, _FONT_SCALE, color, 1, cv2.LINE_8)


def render_overlay(
    clip: Path,
    items_by_frame: Mapping[int, Sequence[OverlayItem]],
    dst: Path,
    overwrite: bool = False,
) -> None:
    """Render an overlay clip.

    Args:
        clip: Source clip.
        items_by_frame: Sampled frame index mapped to the annotations from that frame on.
        dst: Output ``.mp4``. An existing file is kept unless ``overwrite``.
        overwrite: Render even if ``dst`` exists.

    Raises:
        RuntimeError: If the clip cannot be read or ffmpeg fails.
    """
    if is_finished(dst, overwrite):
        logger.info("Overlay already exists: %s", dst)
        return
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot read clip: {clip}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def write(tmp: Path) -> None:
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(tmp),
        ]  # fmt: skip
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert process.stdin is not None and process.stderr is not None
        current: Sequence[OverlayItem] = ()
        index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                current = items_by_frame.get(index, current)
                draw_items(frame, current)
                process.stdin.write(frame.tobytes())
                index += 1
        finally:
            process.stdin.close()
            error = process.stderr.read().decode()
            process.stderr.close()
            process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {error.strip()}")

    try:
        write_atomic(dst, write)
    finally:
        cap.release()


class _NoGrid(Exception):
    """No overlay was available for the grid."""


def render_overlay_grid(
    overlays: Mapping[str, Path | None], dst: Path, overwrite: bool = False
) -> bool:
    """Stack the camera overlays into the 2x2 overview layout.

    Mother is top-left, window top-right, baby bottom-left, door bottom-right; missing
    cameras are black.

    Args:
        overlays: Camera name mapped to its overlay clip, or None.
        dst: Output ``.mp4``. An existing file is kept unless ``overwrite``.
        overwrite: Render even if ``dst`` exists.

    Returns:
        True if the grid exists afterwards.
    """
    if is_finished(dst, overwrite):
        return True

    def build(tmp: Path) -> None:
        if not stack_grid(dict(overlays), tmp):
            raise _NoGrid

    try:
        write_atomic(dst, build)
    except _NoGrid:
        return False
    return True
