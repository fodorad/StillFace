"""Frame-wise feature timelines and their CSV format.

All feature extractors write the same long format: one row per sampled frame and person.
Times are phase-local, i.e. seconds from the start of the phase clip, so they stay
consistent with the cut and stacked overview videos.
"""

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from stillface.features.io import write_atomic

COLUMNS: tuple[str, ...] = (
    "session_id",
    "camera",
    "phase",
    "frame_index",
    "time_s",
    "person",
    "gaze_x",
    "gaze_y",
    "inout",
    "partner_mass",
    "gaze_class",
    "reason",
)
"""Timeline CSV columns, in file order."""


@dataclass(frozen=True)
class TimelineRow:
    """One person at one sampled frame.

    Attributes:
        session_id: Session identifier.
        camera: Camera name.
        phase: Experiment phase.
        frame_index: Index of the frame in the phase clip.
        time_s: Phase-local time, ``frame_index / clip_fps``.
        person: ``mother`` or ``baby``.
        gaze_x: Gaze target x in normalized image coordinates, None if not codable.
        gaze_y: Gaze target y in normalized image coordinates, None if not codable.
        inout: Probability that the gaze target is inside the frame, None if not codable.
        partner_mass: Share of the gaze heatmap on the partner (head box or zone).
        gaze_class: Target class from the plan codes, ``NK`` if not codable.
        reason: Why the row is not codable (``occluded``, ``no_head``, ...), else empty.
    """

    session_id: str
    camera: str
    phase: str
    frame_index: int
    time_s: float
    person: str
    gaze_x: float | None
    gaze_y: float | None
    inout: float | None
    partner_mass: float | None
    gaze_class: str
    reason: str


def _optional_float(text: str) -> float | None:
    return float(text) if text != "" else None


def write_timeline(rows: list[TimelineRow], path: Path) -> None:
    """Write a timeline CSV atomically.

    Args:
        rows: Timeline rows; may be empty.
        path: Output file, parent folders are created.
    """

    def write(tmp: Path) -> None:
        with tmp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            for row in rows:
                record = asdict(row)
                writer.writerow({k: "" if v is None else v for k, v in record.items()})

    write_atomic(path, write)


def read_timeline(path: Path) -> list[TimelineRow]:
    """Read a timeline CSV.

    Args:
        path: File written by :func:`write_timeline`.

    Returns:
        Timeline rows.

    Raises:
        ValueError: If the header does not match :data:`COLUMNS`.
    """
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"Unexpected timeline columns in {path}: {reader.fieldnames}")
        rows = [
            TimelineRow(
                session_id=record["session_id"],
                camera=record["camera"],
                phase=record["phase"],
                frame_index=int(record["frame_index"]),
                time_s=float(record["time_s"]),
                person=record["person"],
                gaze_x=_optional_float(record["gaze_x"]),
                gaze_y=_optional_float(record["gaze_y"]),
                inout=_optional_float(record["inout"]),
                partner_mass=_optional_float(record["partner_mass"]),
                gaze_class=record["gaze_class"],
                reason=record["reason"],
            )
            for record in reader
        ]
    return rows


def sample_frame_indices(n_frames: int, clip_fps: float, sample_rate_hz: float) -> list[int]:
    """Choose the frames of a clip that lie on a regular time grid.

    Args:
        n_frames: Number of frames in the clip.
        clip_fps: Frame rate of the clip.
        sample_rate_hz: Requested sampling rate, at most ``clip_fps``.

    Returns:
        Ascending, unique frame indices. At 30 Hz on a 60 fps clip every second frame.

    Raises:
        ValueError: If ``sample_rate_hz`` exceeds ``clip_fps``.
    """
    if sample_rate_hz > clip_fps:
        raise ValueError(f"Sample rate {sample_rate_hz} Hz exceeds the clip rate {clip_fps} fps")
    grid = np.round(np.arange(0, n_frames, clip_fps / sample_rate_hz)).astype(int)
    return np.unique(grid[grid < n_frames]).tolist()
