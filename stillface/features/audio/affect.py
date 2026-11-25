"""Affect lanes: arousal and valence of the mother and the baby.

Mother (adult speech models, acceptable for her):

- audEERING MSP-dim arousal (``[0, 1]``) and valence (rescaled to ``[-1, 1]``);
- the emotion2vec+ cross-check, derived from its categories:
  ``valence = p(happy) - (p(sad) + p(angry) + p(fearful) + p(disgusted))`` and
  ``arousal = 1 - p(neutral)``;
- per session, the Spearman agreement of the two models on each dimension. Low agreement
  means the session's maternal affect estimates should not be trusted.

Baby (no released infant valence / arousal model exists):

- derived valence ``= sum(weight * p(type))`` with the configured weights
  (default ``-1 p(cry) - 0.5 p(fuss) + 1 p(laugh)``), clipped to ``[-1, 1]``;
- derived arousal = mean of the within-session min-max scaled loudness, F0 and p(cry);
- audEERING on baby windows, written as a separate lane marked *experimental*.

Every lane is NaN outside the segments of its role. Window scores are spread over the grid
bins their window covers (the mean of overlapping windows).
"""

import json
import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from stillface.config import DatasetPaths
from stillface.constants import PHASES
from stillface.features.audio.config import AudioConfig
from stillface.features.audio.fusion import fusion_paths, nanmean_rows
from stillface.features.audio.infer import load_raw
from stillface.features.audio.lanes import (
    Aggregation,
    Lane,
    LaneSpec,
    LaneTable,
    read_lanes,
    write_lanes,
)
from stillface.features.audio.models import SegmentScores
from stillface.features.io import write_atomic

logger = logging.getLogger(__name__)

NEGATIVE_E2V: tuple[str, ...] = ("sad", "angry", "fearful", "disgusted")
"""emotion2vec+ categories counted as negative valence."""


def window_lane(scores: SegmentScores, name: str, role: str, times: np.ndarray) -> np.ndarray:
    """Mean of the windows of one role covering each grid bin.

    Args:
        scores: Window scores (segments = windows with their role).
        name: Score column.
        role: Role whose windows are used.
        times: Grid times.

    Returns:
        Values on the grid, NaN where no window of the role covers a bin.
    """
    column = scores.names.index(name)
    sums, counts = np.zeros(len(times)), np.zeros(len(times))
    for window, row in zip(scores.segments, scores.values, strict=True):
        if window.role != role or not np.isfinite(row[column]):
            continue
        inside = (times >= window.start_s) & (times < window.end_s)
        sums[inside] += row[column]
        counts[inside] += 1
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def e2v_dimensions(
    scores: SegmentScores, role: str, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Valence and arousal derived from emotion2vec+ categories, on the grid.

    Args:
        scores: emotion2vec+ window scores.
        role: Role whose windows are used.
        times: Grid times.

    Returns:
        ``(valence in [-1, 1], arousal in [0, 1])``.
    """
    lanes = {name: window_lane(scores, name, role, times) for name in scores.names}
    negative = np.sum([lanes[n] for n in NEGATIVE_E2V], axis=0)
    valence = np.clip(lanes["happy"] - negative, -1.0, 1.0)
    arousal = np.clip(1.0 - lanes["neutral"], 0.0, 1.0)
    return valence, arousal


def spearman(a: np.ndarray, b: np.ndarray, min_frames: int) -> float:
    """Spearman correlation over the frames where both values exist.

    Args:
        a: First lane values.
        b: Second lane values.
        min_frames: Fewest shared frames for a value.

    Returns:
        The correlation, NaN with too few frames or a constant lane.
    """
    both = np.isfinite(a) & np.isfinite(b)
    if both.sum() < min_frames or np.ptp(a[both]) == 0 or np.ptp(b[both]) == 0:
        return float("nan")
    return float(pd.Series(a[both]).corr(pd.Series(b[both]), method="spearman"))


def baby_valence(p: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Derived baby valence.

    Args:
        p: Type name mapped to its probability lane (NaN outside baby segments).
        weights: Type name mapped to its weight.

    Returns:
        ``sum(weight * p)`` clipped to ``[-1, 1]``; NaN where no type probability exists.

    Raises:
        ValueError: If none of the weighted types has a lane.
    """
    used = [t for t in weights if t in p]
    if not used:
        raise ValueError("no type probability for the baby valence")
    stack = np.stack([p[t] for t in used])
    w = np.array([weights[t] for t in used])[:, None]
    total = np.where(np.isfinite(stack), stack * w, 0.0).sum(axis=0)
    return np.where(np.isfinite(stack).any(axis=0), np.clip(total, -1.0, 1.0), np.nan)


def minmax(values: np.ndarray) -> np.ndarray:
    """Scale to ``[0, 1]`` by the finite minimum and maximum.

    Args:
        values: Values (the whole session, all phases).

    Returns:
        Scaled values; NaN stays NaN, a constant becomes 0.5.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values.copy()
    low, high = finite.min(), finite.max()
    if high == low:
        return np.where(np.isfinite(values), 0.5, np.nan)
    return (values - low) / (high - low)


def _spec(
    source: str, quantity: str, role: str, value_range: tuple[float, float], note: str = ""
) -> LaneSpec:
    return LaneSpec(source, quantity, role, "score", value_range, Aggregation.MEAN, note=note)


def _lane(table: LaneTable, name: str) -> np.ndarray | None:
    try:
        return table.lane(name).values
    except KeyError:
        return None


def run_affect(
    paths: DatasetPaths, session: str, config: AudioConfig, phases: Sequence[str] = PHASES
) -> dict:
    """Write ``affect/<phase>.csv`` (+ legend) and ``affect/summary.json`` of a session.

    Args:
        paths: Dataset locations.
        session: Session identifier.
        config: Audio configuration.
        phases: Phases to process (those with fusion lanes).

    Returns:
        The session summary: per-dimension agreement of the mother models and flags.
    """
    ac = config.affect
    fused: dict[str, LaneTable] = {}
    for phase in phases:
        out = fusion_paths(paths, session, phase)
        if out["lanes"].exists():
            fused[phase] = read_lanes(out["lanes"], out["legend"])
    # Session-wide scaling of the baby arousal features.
    sources = {
        "intensity": "prosody.intensity_db.dyad",
        "f0": "prosody.f0.baby",
        "cry": "fusion.p_cry.baby",
    }
    features: dict[str, dict[str, np.ndarray]] = {}
    for feature in ac.baby_arousal_features:
        per_phase = {}
        for phase, table in fused.items():
            values = _lane(table, sources[feature])
            baby = _lane(table, "fusion.p_cry.baby")
            if values is not None:
                inside = np.isfinite(baby) if baby is not None else np.zeros(len(values), bool)
                per_phase[phase] = np.where(inside, values, np.nan)
        if per_phase:
            joined = minmax(np.concatenate(list(per_phase.values())))
            offsets = np.cumsum([0] + [len(v) for v in per_phase.values()])
            features[feature] = {
                phase: joined[offsets[i] : offsets[i + 1]] for i, phase in enumerate(per_phase)
            }
    pairs: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {"arousal": [], "valence": []}
    for phase, table in fused.items():
        times = table.times
        lanes: list[Lane] = []
        key = f"{session}/{phase}"
        audeering = load_raw(paths, "audeering", key)
        mother: dict[str, np.ndarray] = {}
        if audeering is not None and "windows" in audeering.segments:
            windows = audeering.segments["windows"]
            for role, note in (
                ("mother", ""),
                ("baby", "experimental: adult model on infant voice"),
            ):
                arousal = window_lane(windows, "arousal", role, times)
                valence = 2.0 * window_lane(windows, "valence", role, times) - 1.0
                lanes.append(Lane(_spec("audeering", "arousal", role, (0.0, 1.0), note), arousal))
                lanes.append(Lane(_spec("audeering", "valence", role, (-1.0, 1.0), note), valence))
                if role == "mother":
                    mother = {"arousal": arousal, "valence": valence}
        e2v = load_raw(paths, "emotion2vec", key)
        if e2v is not None and "windows" in e2v.segments:
            valence, arousal = e2v_dimensions(e2v.segments["windows"], "mother", times)
            lanes.append(Lane(_spec("emotion2vec", "arousal", "mother", (0.0, 1.0)), arousal))
            lanes.append(Lane(_spec("emotion2vec", "valence", "mother", (-1.0, 1.0)), valence))
            if mother:
                pairs["arousal"].append((mother["arousal"], arousal))
                pairs["valence"].append((mother["valence"], valence))
        p = {
            t: v
            for t in ac.baby_valence_weights
            if (v := _lane(table, f"fusion.p_{t}.baby")) is not None
        }
        if p:
            lanes.append(
                Lane(
                    _spec("affect", "valence", "baby", (-1.0, 1.0), "derived"),
                    baby_valence(p, ac.baby_valence_weights),
                )
            )
        scaled = [features[f][phase] for f in features if phase in features[f]]
        if scaled:
            lanes.append(
                Lane(
                    _spec("affect", "arousal", "baby", (0.0, 1.0), "derived"),
                    nanmean_rows(np.stack(scaled)),
                )
            )
        out = paths.audio_session_dir(session) / "affect" / f"{phase}.csv"
        notes = {"session": session, "phase": phase, "config": config.digest}
        write_lanes(LaneTable(table.duration_s, lanes, notes=notes), out, out.with_suffix(".json"))
    summary: dict = {"agreement": {}, "flags": []}
    for dimension in ("arousal", "valence"):
        if pairs[dimension]:
            a = np.concatenate([x for x, _ in pairs[dimension]])
            b = np.concatenate([y for _, y in pairs[dimension]])
            rho = spearman(a, b, ac.min_agreement_frames)
        else:
            rho = float("nan")
        summary["agreement"][dimension] = None if np.isnan(rho) else rho
        if np.isnan(rho):
            summary["flags"].append(f"no_{dimension}_agreement")
    text = json.dumps(summary, indent=2) + "\n"
    write_atomic(
        paths.audio_session_dir(session) / "affect" / "summary.json",
        lambda tmp: tmp.write_text(text),
    )
    logger.info("%s: affect lanes for %d phases", session, len(fused))
    return summary
