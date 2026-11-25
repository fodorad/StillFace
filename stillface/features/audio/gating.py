"""Mother-only and baby-only tracks by time-gating.

Most of the time only one partner vocalises, so keeping the original samples inside one
role's segments (widened by a small pad) and silencing the rest already gives a clean track
for ASR and forced alignment. Where mother and baby vocalise at the same time the gated
tracks still contain both; those overlap intervals are listed so later stages can exclude
them or hand them to a source-separation model (a later phase).
"""

import json
import logging
from collections.abc import Sequence

import numpy as np

from stillface.config import DatasetPaths
from stillface.constants import PHASES
from stillface.features.audio.config import AudioConfig
from stillface.features.audio.media import read_wav, write_wav
from stillface.features.audio.prepare import wav_path
from stillface.features.audio.provenance import file_sha256, write_sidecar
from stillface.features.audio.roles import merge_intervals, role_segments
from stillface.features.io import write_atomic

logger = logging.getLogger(__name__)

TRACKS: tuple[str, ...] = ("mother", "baby")
"""Gated tracks written per phase."""


def pad_intervals(
    intervals: Sequence[tuple[float, float]], pad_s: float, duration_s: float
) -> list[tuple[float, float]]:
    """Widen intervals, clip them to the phase and merge the ones that now touch.

    Args:
        intervals: ``(start, end)`` pairs.
        pad_s: Seconds added on both sides.
        duration_s: Phase duration.

    Returns:
        Disjoint padded intervals in time order.
    """
    merged: list[list[float]] = []
    for a, b in sorted((max(0.0, a - pad_s), min(duration_s, b + pad_s)) for a, b in intervals):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def gate(samples: np.ndarray, rate: int, intervals: Sequence[tuple[float, float]]) -> np.ndarray:
    """Keep samples inside intervals, zero elsewhere.

    Args:
        samples: Mono samples.
        rate: Sample rate.
        intervals: ``(start, end)`` pairs in seconds.

    Returns:
        Gated copy of the same length.
    """
    out = np.zeros_like(samples)
    for a, b in intervals:
        start, stop = max(0, round(a * rate)), min(len(samples), round(b * rate))
        out[start:stop] = samples[start:stop]
    return out


def intersect(
    first: Sequence[tuple[float, float]], second: Sequence[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Intervals where both disjoint interval lists are active.

    Args:
        first: Disjoint intervals in time order.
        second: Disjoint intervals in time order.

    Returns:
        Positive-length intersections in time order.
    """
    out = []
    for a1, b1 in first:
        for a2, b2 in second:
            a, b = max(a1, a2), min(b1, b2)
            if b > a:
                out.append((a, b))
    return sorted(out)


def run_gating(
    paths: DatasetPaths, session: str, config: AudioConfig, phases: Sequence[str] = PHASES
) -> dict[str, list[tuple[float, float]]]:
    """Write the gated tracks and the overlap list of every prepared phase.

    Args:
        paths: Dataset locations.
        session: Session identifier.
        config: Audio configuration.
        phases: Phases to process.

    Returns:
        Phase mapped to its mother-baby overlap intervals.
    """
    overlaps = {}
    for phase in phases:
        source = wav_path(paths, session, phase)
        if not source.exists():
            continue
        samples, rate = read_wav(source)
        duration = len(samples) / rate
        spans = {}
        for track in TRACKS:
            segments = role_segments(paths, session, phase, (track,))
            spans[track] = merge_intervals(segments)
            padded = pad_intervals(spans[track], config.gating.pad_s, duration)
            out = wav_path(paths, session, phase, track)
            write_wav(out, gate(samples, rate, padded), rate)
            write_sidecar(
                out,
                {
                    "source": source.relative_to(paths.processed_dir).as_posix(),
                    "source_sha256": file_sha256(source),
                    "track": track,
                    "pad_s": config.gating.pad_s,
                    "intervals": [[round(a, 3), round(b, 3)] for a, b in padded],
                    "duration_s": duration,
                },
            )
        overlaps[phase] = intersect(spans["mother"], spans["baby"])
        text = json.dumps({"overlaps": overlaps[phase]}, indent=2) + "\n"
        target = paths.audio_session_dir(session) / "gating" / f"{phase}.json"
        write_atomic(target, lambda tmp, t=text: tmp.write_text(t))
    logger.info("%s: gated %d phases", session, len(overlaps))
    return overlaps
