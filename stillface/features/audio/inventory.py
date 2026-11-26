"""Session inventory: which microphone each session and phase has, and in what state.

The inventory is the first audio stage. It reads nothing but the phase clips and the
metadata sheet, picks one source clip per session and phase by the configured fallback
order, and records the audio format and level problems, so the later stages and the user
know what to expect before any model runs.
"""

import csv
import logging
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stillface.config import DatasetPaths
from stillface.constants import PHASES
from stillface.features.audio.config import AudioConfig, InventoryConfig
from stillface.features.audio.media import AudioStreamInfo, MediaError, decode_audio, probe_audio
from stillface.features.io import write_atomic

logger = logging.getLogger(__name__)

STATUSES: tuple[str, ...] = ("ok", "fallback", "missing", "no_audio_stream", "decode_error")
"""Inventory status of a session phase, from usable to unusable."""

FALSY_FLAGS: frozenset[str] = frozenset({"", "0", "0.0", "no", "n", "nem", "false", "nan", "none"})
"""Metadata flag values read as False (case-insensitive); anything else is True."""


@dataclass(frozen=True)
class LevelStats:
    """Level statistics of a decoded signal.

    Attributes:
        rms_dbfs: RMS level in dBFS (``-inf`` for digital silence).
        peak_dbfs: Peak level in dBFS.
        clipped_share: Share of samples at or above the clip level.
        silent_share: Share of frames whose RMS is below the silence level.
    """

    rms_dbfs: float
    peak_dbfs: float
    clipped_share: float
    silent_share: float


@dataclass(frozen=True)
class InventoryRow:
    """One session phase in the inventory.

    Attributes:
        session_id: Session identifier.
        phase: Experiment phase.
        analysis: Metadata analysis flag.
        excluded: Listed in the config exclusions.
        in_scope: ``analysis and not excluded``.
        camera: Chosen source camera, empty if none.
        candidates: Cameras whose clip exists, ``;``-separated in fallback order.
        status: One of :data:`STATUSES`.
        reason: Why the status was given (skipped candidates, errors).
        codec: Audio codec of the chosen clip.
        sample_rate: Native sample rate.
        channels: Native channel count.
        audio_s: Audio stream duration.
        video_s: Video stream duration, NaN if none.
        duration_diff_s: ``|audio_s - video_s|``, NaN if no video.
        video_fps: Video frame rate of the chosen clip, NaN if none (Annie event frames).
        rms_dbfs: See :class:`LevelStats`.
        peak_dbfs: See :class:`LevelStats`.
        clipped_share: See :class:`LevelStats`.
        silent_share: See :class:`LevelStats`.
    """

    session_id: str
    phase: str
    analysis: bool
    excluded: bool
    in_scope: bool
    camera: str = ""
    candidates: str = ""
    status: str = "missing"
    reason: str = ""
    codec: str = ""
    sample_rate: int = 0
    channels: int = 0
    audio_s: float = math.nan
    video_s: float = math.nan
    duration_diff_s: float = math.nan
    video_fps: float = math.nan
    rms_dbfs: float = math.nan
    peak_dbfs: float = math.nan
    clipped_share: float = math.nan
    silent_share: float = math.nan


INVENTORY_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(InventoryRow))
"""Inventory CSV columns, in file order."""


def _dbfs(value: float) -> float:
    return 20.0 * math.log10(value) if value > 0 else -math.inf


def level_stats(samples: np.ndarray, rate: int, config: InventoryConfig) -> LevelStats:
    """Compute level statistics of a signal.

    Args:
        samples: Float samples, shape ``(channels, n)`` or ``(n,)``.
        rate: Sample rate.
        config: Frame length and levels.

    Returns:
        The statistics; shares are 0 for an empty signal.
    """
    x = np.atleast_2d(np.asarray(samples, dtype=np.float64)).mean(axis=0)
    if x.size == 0:
        return LevelStats(-math.inf, -math.inf, 0.0, 0.0)
    frame = max(1, round(config.frame_s * rate))
    n_frames = max(1, x.size // frame)
    framed = x[: n_frames * frame].reshape(n_frames, -1) if x.size >= frame else x[None, :]
    frame_rms = np.sqrt((framed**2).mean(axis=1))
    threshold = 10 ** (config.silence_dbfs / 20.0)
    channels = np.atleast_2d(np.asarray(samples, dtype=np.float64))
    return LevelStats(
        rms_dbfs=_dbfs(float(np.sqrt((x**2).mean()))),
        peak_dbfs=_dbfs(float(np.abs(channels).max())),
        clipped_share=float((np.abs(channels) >= config.clip_level).mean()),
        silent_share=float((frame_rms < threshold).mean()),
    )


def is_flag_set(value: object) -> bool:
    """Read a metadata yes/no cell.

    Args:
        value: Cell value (number, text, NaN or None).

    Returns:
        False for empty, NaN, zero and the words in :data:`FALSY_FLAGS`, else True.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() not in FALSY_FLAGS


def analysis_flags(metadata: pd.DataFrame, column: str) -> dict[str, bool]:
    """Analysis flag per session id.

    Args:
        metadata: Metadata table with an ``ID`` column.
        column: Flag column.

    Returns:
        Session id (as text, without a trailing ``.0``) mapped to the flag. Distinct raw
        values are logged so the reading can be checked.
    """
    if column not in metadata.columns:
        logger.warning("Metadata has no %r column: every session is out of scope", column)
        return {}
    logger.info("Distinct %r values: %s", column, sorted(map(str, metadata[column].unique())))
    flags = {}
    for sid, value in zip(metadata["ID"], metadata[column], strict=True):
        text = str(sid)
        flags[text.removesuffix(".0")] = is_flag_set(value)
    return flags


def inspect_phase(
    paths: DatasetPaths, session_id: str, phase: str, row: InventoryRow, config: AudioConfig
) -> InventoryRow:
    """Choose the source clip of a session phase and measure it.

    Args:
        paths: Dataset locations.
        session_id: Session identifier.
        phase: Experiment phase.
        row: Row with the scope fields filled.
        config: Source order and level settings.

    Returns:
        The completed row.
    """
    order = config.source.cameras
    present = [c for c in order if paths.phase_clip(session_id, c, phase).exists()]
    notes, problems = [], set()
    for camera in present:
        clip = paths.phase_clip(session_id, camera, phase)
        try:
            info = probe_audio(clip)
            if info is None:
                problems.add("no_audio_stream")
                notes.append(f"{camera}: no audio stream")
                continue
            samples, rate = decode_audio(clip)
        except MediaError as error:
            problems.add("decode_error")
            notes.append(f"{camera}: {error}")
            continue
        return _measured(row, camera, present, notes, info, samples, rate, config, order)
    status = (
        "decode_error"
        if "decode_error" in problems
        else ("no_audio_stream" if problems else "missing")
    )
    return replace(row, candidates=";".join(present), status=status, reason="; ".join(notes))


def _measured(
    row: InventoryRow,
    camera: str,
    present: list[str],
    notes: list[str],
    info: AudioStreamInfo,
    samples: np.ndarray,
    rate: int,
    config: AudioConfig,
    order: tuple[str, ...],
) -> InventoryRow:
    stats = level_stats(samples, rate, config.inventory)
    fallback = camera != order[0]
    if fallback:
        notes = [f"{order[0]}: clip missing", *notes] if order[0] not in present else notes
    video = math.nan if info.video_duration_s is None else info.video_duration_s
    return replace(
        row,
        camera=camera,
        candidates=";".join(present),
        status="fallback" if fallback else "ok",
        reason="; ".join(notes),
        codec=info.codec,
        sample_rate=info.sample_rate,
        channels=info.channels,
        audio_s=info.duration_s,
        video_s=video,
        duration_diff_s=abs(info.duration_s - video),
        video_fps=math.nan if info.video_fps is None else info.video_fps,
        rms_dbfs=stats.rms_dbfs,
        peak_dbfs=stats.peak_dbfs,
        clipped_share=stats.clipped_share,
        silent_share=stats.silent_share,
    )


def session_ids(paths: DatasetPaths) -> list[str]:
    """Session folders of the raw dataset.

    Args:
        paths: Dataset locations.

    Returns:
        Sorted session ids; empty if the sessions folder is missing.
    """
    root = paths.raw_dir / "Sessions"
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


def build_inventory(
    paths: DatasetPaths, config: AudioConfig, metadata: pd.DataFrame
) -> list[InventoryRow]:
    """Inventory every session and phase.

    Args:
        paths: Dataset locations.
        config: Audio configuration.
        metadata: Session metadata table.

    Returns:
        One row per session and phase, sessions sorted, phases in protocol order.
    """
    flags = analysis_flags(metadata, config.scope.analysis_column)
    excluded = set(config.scope.exclude)
    rows = []
    for sid in session_ids(paths):
        analysis = flags.get(sid, False)
        out = sid in excluded
        for phase in PHASES:
            base = InventoryRow(sid, phase, analysis, out, analysis and not out)
            rows.append(inspect_phase(paths, sid, phase, base, config))
    by_status = {s: sum(r.status == s for r in rows) for s in STATUSES}
    logger.info("Inventory of %d session phases: %s", len(rows), by_status)
    return rows


def _cell(value: object) -> object:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def write_inventory(rows: list[InventoryRow], path: Path) -> None:
    """Write the inventory CSV atomically.

    Args:
        rows: Inventory rows.
        path: Output file.
    """

    def write(tmp: Path) -> None:
        with tmp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=INVENTORY_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: _cell(v) for k, v in asdict(row).items()})

    write_atomic(path, write)


def read_inventory(path: Path) -> list[InventoryRow]:
    """Read an inventory CSV.

    Args:
        path: File written by :func:`write_inventory`.

    Returns:
        The rows.

    Raises:
        ValueError: If the header does not match :data:`INVENTORY_COLUMNS`.
    """
    kinds = {f.name: f.type for f in fields(InventoryRow)}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != INVENTORY_COLUMNS:
            raise ValueError(f"Unexpected inventory columns in {path}: {reader.fieldnames}")
        rows = []
        for record in reader:
            values: dict[str, Any] = {}
            for key, text in record.items():
                kind = kinds[key]
                if kind is bool:
                    values[key] = text == "1"
                elif kind is int:
                    values[key] = int(text)
                elif kind is float:
                    values[key] = math.nan if text == "" else float(text)
                else:
                    values[key] = text
            rows.append(InventoryRow(**values))
        return rows
