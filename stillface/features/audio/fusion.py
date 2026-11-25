"""Rule-based fusion: every model output on one grid, plus the fused decisions.

For one session phase the fusion stage reads the raw outputs of the first- and second-pass
models and the role events, and writes:

- ``lanes/<phase>.csv`` + ``lanes/<phase>.json``: every raw score as a lane (10 Hz) and every
  presence decision as a band, each band with its threshold, origin and rule id;
- ``events/<phase>.csv``: fused vocalisation events (baby type per baby segment, mother type
  per mother segment, addressee per adult segment);
- ``fusion/<phase>.json``: phase summary (context shares, disagreement share, which models
  were available).

All rules are listed by id in :data:`RULES` and their thresholds come from the config, so
:mod:`~stillface.features.audio.decisions` can render the exact decision tables that
produced the data. No rule is learned; learned fusion needs annotations (later phase).
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from stillface.config import DatasetPaths
from stillface.constants import PHASES
from stillface.features.audio.config import AudioConfig, TypeRule
from stillface.features.audio.infer import load_raw, wav_duration
from stillface.features.audio.lanes import (
    Aggregation,
    Band,
    BandSpec,
    Event,
    Lane,
    LaneSpec,
    LaneTable,
    grid_times,
    read_events,
    runs,
    to_grid,
    write_events,
    write_lanes,
)
from stillface.features.audio.models import FrameScores, InferenceResult
from stillface.features.audio.prepare import wav_path
from stillface.features.audio.roles import ListeningItem, role_events_path
from stillface.features.io import write_atomic

logger = logging.getLogger(__name__)

MODELS: dict[str, str] = {
    "silero": "silero",
    "pyannote": "pyannote_vad",
    "ast": "ast",
    "littlebeats": "littlebeats",
    "prosody": "prosody",
    "addressee": "addressee",
}
"""Model names whose raw outputs the fusion reads (VTC comes from the roles config)."""

VTC_ROLES: dict[str, str] = {"KCHI": "baby", "OCH": "baby", "FEM": "adult", "MAL": "adult"}
"""Lane role of each VTC class (adults are split into mother / other by the roles stage)."""

EXTRA_TYPES: tuple[str, ...] = ("other_vocal", "conflict")
"""Baby type labels besides the configured types: no vote, and a tie."""

RULES: dict[str, str] = {
    "VAD-VTC": "VTC segment of one class (thresholded by VTC's own per-class set)",
    "VAD-SILERO": "Silero speech probability >= threshold 'silero'",
    "VAD-PYANNOTE": "pyannote speech probability >= threshold 'pyannote'",
    "VAD-FUSED": "Voice activity = union of the four VTC classes, outside the gong mask",
    "VAD-DISAGREE": "Fused VAD differs from Silero or from pyannote",
    "SILENCE": "No voice source (VTC, Silero, pyannote) active and no gong",
    "GONG": "Max of the gong classes >= threshold 'gong' within the phase edges, dilated",
    "CONTEXT": "AudioSet context class >= threshold 'context'",
    "ROLE-OTHER": "Other adult: MAL, another woman, uncertain or unresolved female voice",
    "OVERLAP": "Baby and mother segments at the same time",
    "BABY-TYPE-VOTE": "Baby segment type = the type most models vote for",
    "BABY-TYPE-CONFLICT": "Baby segment where two or more types get the same top vote count",
    "BABY-TYPE-NONE": "Baby segment where no model votes for any type: other_vocal",
    "BABY-P-TYPE": "Probability of a baby type = mean over available models, baby segments only",
    "MOTHER-TYPE": "Mother segment type = highest AST type >= threshold 'ast_type', else speech",
    "ADDRESSEE": "Adult segment addressee = most probable of KCDS / ADS / Other",
    "ADS-DEVIATION": "Mother segment classified ADS: speech to another adult (flag)",
    "PROSODY-ROLE": "F0 of the adult range inside mother segments, infant range inside baby ones",
}
"""Fusion rules by id (the role rules are in :data:`stillface.features.audio.roles.RULES`)."""


@dataclass
class PhaseInputs:
    """Everything the fusion reads for one phase.

    Attributes:
        session: Session identifier.
        phase: Experiment phase.
        duration_s: Phase duration.
        raw: Model key (``vtc``, ``silero`` ...) mapped to its raw result, None if missing.
        roles: Role events of the phase.
    """

    session: str
    phase: str
    duration_s: float
    raw: dict[str, InferenceResult | None]
    roles: list[Event] = field(default_factory=list)


@dataclass
class FusionResult:
    """Output of the fusion of one phase.

    Attributes:
        table: Lanes and bands.
        events: Fused events.
        summary: Phase summary.
        listening: Listening items found by the fusion.
    """

    table: LaneTable
    events: list[Event]
    summary: dict
    listening: list[ListeningItem]


def hold_for(times: np.ndarray) -> float:
    """Hold distance for a source: half its median sample spacing plus half a grid bin.

    Args:
        times: Sample times.

    Returns:
        Seconds; 0 for fewer than two samples.
    """
    if len(times) < 2:
        return 0.0
    return float(np.median(np.diff(np.sort(times)))) / 2 + 0.5 / 10


def nanmean_rows(stack: np.ndarray) -> np.ndarray:
    """Column-wise mean ignoring NaN, NaN where a column has no finite value.

    Args:
        stack: Shape ``(k, n)``.

    Returns:
        Shape ``(n,)``.
    """
    finite = np.isfinite(stack)
    counts = finite.sum(axis=0)
    sums = np.where(finite, stack, 0.0).sum(axis=0)
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def raster(intervals: Sequence[tuple[float, float]], times: np.ndarray) -> np.ndarray:
    """Grid presence of intervals: ``start <= t < end``.

    Args:
        intervals: ``(start, end)`` pairs.
        times: Grid times.

    Returns:
        Boolean mask.
    """
    mask = np.zeros(len(times), dtype=bool)
    for a, b in intervals:
        mask |= (times >= a) & (times < b)
    return mask


def dilate(mask: np.ndarray, bins: int) -> np.ndarray:
    """Widen every run of a mask by ``bins`` on both sides.

    Args:
        mask: Boolean mask.
        bins: Bins added per side.

    Returns:
        The widened mask.
    """
    out = mask.copy()
    for a, b in runs(mask):
        out[max(0, a - bins) : b + bins] = True
    return out


def column(frames: FrameScores, names: Sequence[str]) -> np.ndarray:
    """Maximum over named columns of frame scores.

    Args:
        frames: Frame scores.
        names: Column names; all must exist.

    Returns:
        Per-frame maximum, NaN if ``names`` is empty.

    Raises:
        KeyError: If a name is unknown.
    """
    if not names:
        return np.full(len(frames.times), np.nan)
    missing = [n for n in names if n not in frames.names]
    if missing:
        raise KeyError(f"Unknown classes {missing}")
    index = [frames.names.index(n) for n in names]
    return frames.values[:, index].max(axis=1)


def segment_max(frames: FrameScores, values: np.ndarray, start: float, end: float) -> float:
    """Largest value of the frames whose time lies in a segment (nearest frame if none).

    Args:
        frames: Frame scores giving the times.
        values: One value per frame.
        start: Segment start.
        end: Segment end.

    Returns:
        The maximum, NaN without frames.
    """
    if len(frames.times) == 0:
        return float("nan")
    inside = (frames.times >= start) & (frames.times < end)
    if not inside.any():
        inside = np.zeros(len(frames.times), dtype=bool)
        inside[int(np.argmin(np.abs(frames.times - (start + end) / 2)))] = True
    return float(np.nanmax(values[inside])) if np.isfinite(values[inside]).any() else float("nan")


def _prob_spec(source: str, quantity: str, role: str, note: str = "") -> LaneSpec:
    return LaneSpec(source, quantity, role, "prob", (0.0, 1.0), Aggregation.MAX, note=note)


class _Builder:
    """Collects lanes, bands, events and listening items of one phase."""

    def __init__(self, inputs: PhaseInputs, config: AudioConfig) -> None:
        self.inputs, self.config = inputs, config
        self.times = grid_times(inputs.duration_s)
        self.lanes: list[Lane] = []
        self.bands: list[Band] = []
        self.events: list[Event] = []
        self.listening: list[ListeningItem] = []
        self.summary: dict = {"models": {k: v is not None for k, v in inputs.raw.items()}}

    def lane(self, spec: LaneSpec, times: np.ndarray, values: np.ndarray) -> Lane:
        lane = to_grid(
            times,
            np.clip(values, *spec.value_range),
            self.inputs.duration_s,
            spec,
            hold_s=hold_for(times),
        )
        self.lanes.append(lane)
        return lane

    def grid_lane(self, spec: LaneSpec, values: np.ndarray) -> Lane:
        lane = Lane(spec, values)
        self.lanes.append(lane)
        return lane

    def band(
        self,
        name: str,
        lane: str,
        threshold: float | None,
        origin: str,
        rule: str,
        mask: np.ndarray,
    ) -> np.ndarray:
        band = Band(BandSpec(name, lane, threshold, origin, rule), mask)
        self.bands.append(band)
        return band.values

    def listen(
        self, item: str, start: float = float("nan"), end: float = float("nan"), detail: str = ""
    ) -> None:
        i = self.inputs
        self.listening.append(ListeningItem(i.session, i.phase, item, start, end, detail))

    def threshold_band(self, lane: Lane, name: str, rule: str) -> np.ndarray:
        t = self.config.fusion.threshold(name)
        mask = np.nan_to_num(lane.values, nan=-np.inf) >= t.value
        return self.band(f"{lane.spec.name}.on", lane.spec.name, t.value, t.origin, rule, mask)


def fuse_phase(inputs: PhaseInputs, config: AudioConfig) -> FusionResult:
    """Fuse the model outputs of one phase.

    Args:
        inputs: Raw outputs and role events.
        config: Audio configuration (fusion rules, thresholds).

    Returns:
        Lanes, bands, events, summary and listening items.
    """
    b = _Builder(inputs, config)
    fc = config.fusion
    raw = inputs.raw
    n = len(b.times)
    off = np.zeros(n, dtype=bool)

    # Voice activity: VTC classes, the fused union, and the two second opinions.
    vtc_any = off.copy()
    vtc = raw.get("vtc")
    if vtc is not None and "voice_type" in vtc.frames:
        frames = vtc.frames["voice_type"]
        applied = vtc.meta.get("applied_thresholds", {})
        segments = vtc.segments.get("voice_type")
        for label in frames.names:
            role = VTC_ROLES.get(label, "adult")
            lane = b.lane(_prob_spec("vtc", label, role), frames.times, column(frames, [label]))
            spans = (
                [(s.start_s, s.end_s) for s in segments.segments if s.label == label]
                if segments
                else []
            )
            value = applied.get(label)
            threshold = float(value) if isinstance(value, int | float) else None
            mask = raster(spans, b.times)
            vtc_any |= b.band(
                f"{lane.spec.name}.on", lane.spec.name, threshold, "vendor", "VAD-VTC", mask
            )
    opinions = {}
    for key, rule in (("silero", "VAD-SILERO"), ("pyannote", "VAD-PYANNOTE")):
        result = raw.get(key)
        if result is not None and "speech" in result.frames:
            frames = result.frames["speech"]
            lane = b.lane(
                _prob_spec(key, "speech", "dyad"), frames.times, column(frames, ["speech"])
            )
            opinions[key] = b.threshold_band(lane, key, rule)

    # AST lanes of every class a rule uses, then the gong mask.
    ast = raw.get("ast")
    ast_frames = ast.frames.get("audioset") if ast is not None else None
    ast_lanes: dict[str, Lane] = {}
    if ast_frames is not None:
        used = [*fc.gong.classes, *fc.context.classes]
        for rule in [*fc.baby_types.values(), *fc.mother_types.values()]:
            used += list(rule.ast)
        for name in dict.fromkeys(used):
            spec = _prob_spec("ast", name, "context")
            ast_lanes[name] = b.lane(spec, ast_frames.times, column(ast_frames, [name]))
    gong = off.copy()
    if ast_frames is not None:
        t = fc.threshold("gong")
        score = np.nanmax(np.stack([ast_lanes[c].values for c in fc.gong.classes]), axis=0)
        edges = (b.times < fc.gong.edge_s) | (b.times > inputs.duration_s - fc.gong.edge_s)
        hit = (np.nan_to_num(score, nan=-np.inf) >= t.value) & edges
        mask = dilate(hit, round(fc.gong.dilate_s * 10)) & edges
        gong = b.band("fusion.gong.context", "", t.value, t.origin, "GONG", mask)
    fused = b.band("fusion.vad.dyad", "", None, "config", "VAD-FUSED", vtc_any & ~gong)
    disagree = off.copy()
    for mask in opinions.values():
        disagree |= fused != (mask & ~gong)
    b.band("fusion.vad_disagree.dyad", "", None, "config", "VAD-DISAGREE", disagree)
    any_voice = vtc_any.copy()
    for mask in opinions.values():
        any_voice |= mask
    b.band("fusion.silence.dyad", "", None, "config", "SILENCE", ~any_voice & ~gong)
    share = float(disagree.mean()) if n else 0.0
    b.summary["vad_disagreement_share"] = share
    if opinions and share >= fc.disagreement_listen_share:
        b.listen("vad_disagreement", detail=f"share {share:.2f}")

    # Context flags.
    shares = {}
    for name in fc.context.classes:
        if name in ast_lanes:
            mask = b.threshold_band(ast_lanes[name], "context", "CONTEXT") & ~gong
            shares[name] = float(mask.mean()) if n else 0.0
            if shares[name] >= fc.context.listen_share:
                b.listen(f"context:{name}", detail=f"share {shares[name]:.2f}")
    b.summary["context_shares"] = shares

    # Roles on the grid.
    role_spans = {
        role: [(e.start_s, e.end_s) for e in inputs.roles if e.role == role]
        for role in ("baby", "mother", "other")
    }
    rules = {"baby": "ROLE-BABY", "mother": "ROLE-FEM-MOTHER", "other": "ROLE-OTHER"}
    masks = {}
    for role, spans in role_spans.items():
        masks[role] = b.band(
            f"roles.active.{role}", "", None, "config", rules[role], raster(spans, b.times)
        )
    b.band("fusion.overlap.dyad", "", None, "config", "OVERLAP", masks["baby"] & masks["mother"])

    _baby_types(b, inputs, ast_frames, masks["baby"])
    _mother_types(b, inputs, ast_frames)
    _addressee(b, raw.get("addressee"), masks["mother"])
    _prosody(b, raw.get("prosody"), masks)
    table = LaneTable(
        inputs.duration_s,
        b.lanes,
        b.bands,
        notes={"session": inputs.session, "phase": inputs.phase, "config": config.digest},
    )
    b.events.sort(key=lambda e: (e.start_s, e.role, e.label))
    return FusionResult(table, b.events, b.summary, b.listening)


def _type_scores(
    rule: TypeRule, ast_frames: FrameScores | None, lb_frames: FrameScores | None
) -> dict[str, tuple[FrameScores, np.ndarray]]:
    scores = {}
    if ast_frames is not None and rule.ast:
        scores["ast"] = (ast_frames, column(ast_frames, rule.ast))
    if lb_frames is not None and rule.littlebeats:
        scores["littlebeats"] = (lb_frames, column(lb_frames, rule.littlebeats))
    return scores


def _baby_types(
    b: _Builder, inputs: PhaseInputs, ast_frames: FrameScores | None, baby: np.ndarray
) -> None:
    fc = b.config.fusion
    lb = inputs.raw.get("littlebeats")
    lb_frames = lb.frames.get("infant") if lb is not None else None
    thresholds = {
        "ast": fc.threshold("ast_type").value,
        "littlebeats": fc.threshold("littlebeats").value,
    }
    available = [m for m, f in (("ast", ast_frames), ("littlebeats", lb_frames)) if f is not None]
    b.summary["baby_type_models"] = available
    types = list(fc.baby_types)
    categories = (*types, *EXTRA_TYPES)
    per_type = {
        name: _type_scores(rule, ast_frames, lb_frames) for name, rule in fc.baby_types.items()
    }
    for name, scores in per_type.items():
        grids = [
            to_grid(
                f.times, v, inputs.duration_s, _prob_spec(m, name, "baby"), hold_s=hold_for(f.times)
            ).values
            for m, (f, v) in scores.items()
        ]
        values = nanmean_rows(np.stack(grids)) if grids else np.full(len(b.times), np.nan)
        values = np.where(baby, values, np.nan)
        b.grid_lane(_prob_spec("fusion", f"p_{name}", "baby", "BABY-P-TYPE"), values)
    labels = np.full(len(b.times), np.nan)
    for event in inputs.roles:
        if event.role != "baby":
            continue
        votes = dict.fromkeys(types, 0)
        for name, scores in per_type.items():
            for model, (frames, values) in scores.items():
                if segment_max(frames, values, event.start_s, event.end_s) >= thresholds[model]:
                    votes[name] += 1
        top = max(votes.values()) if votes else 0
        winners = [t for t, v in votes.items() if v == top and v > 0]
        if not winners:
            label, rule, confidence = "other_vocal", "BABY-TYPE-NONE", 1.0
        elif len(winners) > 1:
            label, rule, confidence = "conflict", "BABY-TYPE-CONFLICT", top / max(1, len(available))
        else:
            label, rule, confidence = winners[0], "BABY-TYPE-VOTE", top / max(1, len(available))
        threshold = thresholds["ast"] if ast_frames is not None else None
        b.events.append(
            Event(
                event.start_s,
                event.end_s,
                "baby",
                label,
                min(1.0, confidence),
                "fusion",
                threshold,
                rule,
            )
        )
        labels[(b.times >= event.start_s) & (b.times < event.end_s)] = categories.index(label)
    spec = LaneSpec(
        "fusion",
        "type",
        "baby",
        "category",
        (0.0, len(categories) - 1.0),
        Aggregation.MAJORITY,
        categories,
    )
    b.grid_lane(spec, labels)


def _mother_types(b: _Builder, inputs: PhaseInputs, ast_frames: FrameScores | None) -> None:
    fc = b.config.fusion
    t = fc.threshold("ast_type")
    for event in inputs.roles:
        if event.role != "mother":
            continue
        best, score = "speech", -np.inf
        if ast_frames is not None:
            for name, rule in fc.mother_types.items():
                value = segment_max(
                    ast_frames, column(ast_frames, rule.ast), event.start_s, event.end_s
                )
                if value >= t.value and value > score:
                    best, score = name, value
        confidence = float(np.clip(score, 0, 1)) if np.isfinite(score) else 1.0
        b.events.append(
            Event(
                event.start_s,
                event.end_s,
                "mother",
                best,
                confidence,
                "fusion",
                t.value,
                "MOTHER-TYPE",
            )
        )


def _addressee(b: _Builder, result: InferenceResult | None, mother: np.ndarray) -> None:
    if result is None or "addressee" not in result.segments:
        return
    scores = result.segments["addressee"]
    grids = {name: np.full(len(b.times), np.nan) for name in scores.names}
    for segment, row in zip(scores.segments, scores.values, strict=True):
        inside = (b.times >= segment.start_s) & (b.times < segment.end_s)
        for name, value in zip(scores.names, row, strict=True):
            grids[name][inside] = value
        label = scores.names[int(np.argmax(row))]
        is_mother = bool(inside.any() and mother[inside].mean() >= 0.5)
        role = "mother" if is_mother else "other"
        b.events.append(
            Event(
                segment.start_s,
                segment.end_s,
                role,
                label,
                float(row.max()),
                "addressee",
                None,
                "ADDRESSEE",
            )
        )
        if is_mother and label == "ADS":
            b.events.append(
                Event(
                    segment.start_s,
                    segment.end_s,
                    "mother",
                    "ads_deviation",
                    float(row.max()),
                    "fusion",
                    None,
                    "ADS-DEVIATION",
                )
            )
            b.listen("ads_deviation", segment.start_s, segment.end_s, f"p(ADS) {row.max():.2f}")
    for name, values in grids.items():
        b.grid_lane(_prob_spec("addressee", name, "adult"), values)


def _prosody(b: _Builder, result: InferenceResult | None, masks: dict[str, np.ndarray]) -> None:
    if result is None:
        return
    duration = b.inputs.duration_s
    for group, role in (("f0_adult", "mother"), ("f0_infant", "baby")):
        if group in result.frames:
            frames = result.frames[group]
            spec = LaneSpec(
                "prosody", "f0", role, "Hz", (0.0, 2000.0), Aggregation.MEAN, note="PROSODY-ROLE"
            )
            grid = to_grid(frames.times, column(frames, ["f0_hz"]), duration, spec)
            b.grid_lane(spec, np.where(masks[role], grid.values, np.nan))
    for group, quantity, unit, value_range in (
        ("intensity", "intensity_db", "dB", (-300.0, 300.0)),
        ("hnr", "hnr_db", "dB", (-300.0, 300.0)),
    ):
        if group in result.frames:
            frames = result.frames[group]
            spec = LaneSpec("prosody", quantity, "dyad", unit, value_range, Aggregation.MEAN)
            b.lane(spec, frames.times, column(frames, [quantity]))


def load_inputs(
    paths: DatasetPaths, session: str, phase: str, config: AudioConfig
) -> PhaseInputs | None:
    """Collect the raw outputs and role events of one phase.

    Args:
        paths: Dataset locations.
        session: Session identifier.
        phase: Experiment phase.
        config: Audio configuration (model names).

    Returns:
        The inputs, None if the phase WAV was not prepared.
    """
    wav = wav_path(paths, session, phase)
    if not wav.exists():
        return None
    key = f"{session}/{phase}"
    raw = {"vtc": load_raw(paths, config.roles.vtc_model, key)}
    for name, model in MODELS.items():
        enabled = config.models.get(model)
        raw[name] = load_raw(paths, model, key) if enabled is None or enabled.enabled else None
    roles_file = role_events_path(paths, session, phase)
    roles = read_events(roles_file) if roles_file.exists() else []
    return PhaseInputs(session, phase, wav_duration(wav), raw, roles)


def fusion_paths(paths: DatasetPaths, session: str, phase: str) -> dict[str, Path]:
    """Output files of the fusion of one phase.

    Args:
        paths: Dataset locations.
        session: Session identifier.
        phase: Experiment phase.

    Returns:
        ``lanes`` (CSV), ``legend`` (JSON), ``events`` (CSV) and ``summary`` (JSON) paths.
    """
    root = paths.audio_session_dir(session)
    return {
        "lanes": root / "lanes" / f"{phase}.csv",
        "legend": root / "lanes" / f"{phase}.json",
        "events": root / "events" / f"{phase}.csv",
        "summary": root / "fusion" / f"{phase}.json",
    }


def run_fusion(
    paths: DatasetPaths, session: str, config: AudioConfig, phases: Sequence[str] = PHASES
) -> list[ListeningItem]:
    """Fuse every prepared phase of a session and write the outputs.

    Args:
        paths: Dataset locations.
        session: Session identifier.
        config: Audio configuration.
        phases: Phases to fuse.

    Returns:
        The listening items of the session.
    """
    items: list[ListeningItem] = []
    for phase in phases:
        inputs = load_inputs(paths, session, phase, config)
        if inputs is None:
            continue
        result = fuse_phase(inputs, config)
        out = fusion_paths(paths, session, phase)
        write_lanes(result.table, out["lanes"], out["legend"])
        write_events(result.events, out["events"])
        text = json.dumps(result.summary, indent=2, sort_keys=True) + "\n"
        write_atomic(out["summary"], lambda tmp, t=text: tmp.write_text(t))
        items += result.listening
    logger.info("%s: fused, %d listening items", session, len(items))
    return items
