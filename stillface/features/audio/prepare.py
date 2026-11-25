"""Prepared analysis audio: one 16 kHz mono WAV per session and phase.

The source clip is the one chosen by the inventory. The audio is downmixed by the channel
mean and resampled, without gain or normalisation, so intensity stays meaningful. Each WAV
has a provenance sidecar naming its source and the source hash; outputs whose source and
settings did not change are kept.
"""

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from stillface import __version__
from stillface.config import DatasetPaths
from stillface.features.audio.config import AudioConfig
from stillface.features.audio.inventory import InventoryRow
from stillface.features.audio.media import decode_audio, write_wav
from stillface.features.audio.provenance import file_sha256, is_up_to_date, write_sidecar

logger = logging.getLogger(__name__)

USABLE_STATUSES: tuple[str, ...] = ("ok", "fallback")
"""Inventory statuses with a usable source clip."""


def wav_path(paths: DatasetPaths, session_id: str, phase: str, track: str = "") -> Path:
    """Prepared WAV of a session phase.

    Args:
        paths: Dataset locations.
        session_id: Session identifier.
        phase: Experiment phase.
        track: Empty for the full phase audio, else a gated track name (``mother``, ``baby``).

    Returns:
        ``<audio session dir>/wav/<phase>[_<track>].wav``.
    """
    name = f"{phase}_{track}" if track else phase
    return paths.audio_session_dir(session_id) / "wav" / f"{name}.wav"


@dataclass
class PrepareReport:
    """Outcome of a preparation run.

    Attributes:
        written: ``<session>/<phase>`` items written.
        kept: Items already up to date.
        failed: Item mapped to the error message.
    """

    written: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


def prepare_phase(
    paths: DatasetPaths, row: InventoryRow, config: AudioConfig, overwrite: bool = False
) -> bool:
    """Write the prepared WAV of one inventory row.

    Args:
        paths: Dataset locations.
        row: Inventory row with a usable source.
        config: Audio configuration.
        overwrite: Recompute even if up to date.

    Returns:
        True if written, False if the existing output was kept.

    Raises:
        ValueError: If the row has no usable source.
        MediaError: If decoding fails.
    """
    if row.status not in USABLE_STATUSES:
        raise ValueError(f"{row.session_id}/{row.phase}: no usable source ({row.status})")
    source = paths.phase_clip(row.session_id, row.camera, row.phase)
    out = wav_path(paths, row.session_id, row.phase)
    expected = {
        "source_sha256": file_sha256(source),
        "sample_rate": config.prepare.sample_rate,
        "stillface_version": __version__,
    }
    if is_up_to_date(out, expected, overwrite):
        return False
    samples, rate = decode_audio(source, config.prepare.sample_rate, mono=True)
    write_wav(out, samples[0], rate)
    write_sidecar(
        out,
        {
            **expected,
            "source": source.relative_to(paths.raw_dir).as_posix(),
            "camera": row.camera,
            "inventory_status": row.status,
            "source_sample_rate": row.sample_rate,
            "source_channels": row.channels,
            "video_fps": None if math.isnan(row.video_fps) else row.video_fps,
            "source_rms_dbfs": row.rms_dbfs,
            "source_peak_dbfs": row.peak_dbfs,
            "samples": int(samples.shape[1]),
            "duration_s": samples.shape[1] / rate,
        },
    )
    return True


def prepare_all(
    paths: DatasetPaths,
    rows: Iterable[InventoryRow],
    config: AudioConfig,
    sessions: set[str] | None = None,
    overwrite: bool = False,
) -> PrepareReport:
    """Prepare every in-scope session phase with a usable source.

    A failing item is logged and recorded; the others continue.

    Args:
        paths: Dataset locations.
        rows: Inventory rows.
        config: Audio configuration.
        sessions: Restrict to these sessions (still only in-scope ones); all if None.
        overwrite: Recompute existing outputs.

    Returns:
        What was written, kept and failed.
    """
    report = PrepareReport()
    for row in rows:
        if not row.in_scope or row.status not in USABLE_STATUSES:
            continue
        if sessions is not None and row.session_id not in sessions:
            continue
        item = f"{row.session_id}/{row.phase}"
        try:
            (
                report.written if prepare_phase(paths, row, config, overwrite) else report.kept
            ).append(item)
        except Exception as error:
            logger.exception("Preparing %s failed", item)
            report.failed[item] = str(error)
    logger.info(
        "Prepared %d, kept %d, failed %d",
        len(report.written),
        len(report.kept),
        len(report.failed),
    )
    return report
