"""Roles: turn VTC voice types into mother, baby and other speakers.

VTC tells voice *types* apart (key child, other child, adult female, adult male), not
people. Here:

- the baby is every ``KCHI`` or ``OCH`` segment (no siblings take part), merged;
- an adult male (``MAL``) is another adult (father, experimenter);
- an adult female (``FEM``) segment is the mother only if its speaker embedding is close to
  the mother's enrolled voice. Enrolment takes the dominant female voice of the play phase
  (segments without baby overlap), see :func:`enrol`; with too little speech the reunion
  phase is tried, and failing that the session stays unresolved.

Every decision carries a rule id; uncertain female segments, male segments and unresolved
sessions go to the listening list for the user to annotate.
"""

import csv
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from stillface.config import DatasetPaths
from stillface.constants import PHASES
from stillface.features.audio.config import AudioConfig, RolesConfig
from stillface.features.audio.infer import RequestBuilder, load_raw, wav_duration
from stillface.features.audio.lanes import Event, read_events, write_events
from stillface.features.audio.models import InferenceRequest, Segment
from stillface.features.audio.prepare import wav_path
from stillface.features.io import write_atomic

logger = logging.getLogger(__name__)

RULES: dict[str, str] = {
    "ROLE-BABY": "VTC KCHI or OCH segment, overlapping ones merged: the baby",
    "ROLE-MAL": "VTC MAL segment: another adult (father, experimenter)",
    "ROLE-FEM-MOTHER": "VTC FEM segment with embedding similarity >= tau_high: the mother",
    "ROLE-FEM-OTHER": "VTC FEM segment with similarity <= tau_low: another woman",
    "ROLE-FEM-UNCERTAIN": "VTC FEM segment between tau_low and tau_high, or without embedding",
    "ROLE-FEM-UNRESOLVED": "VTC FEM segment of a session without mother enrolment",
}
"""Role rules, by id (rendered in the decision tables)."""


def merge_intervals(segments: Sequence[Segment]) -> list[tuple[float, float]]:
    """Union of overlapping or touching intervals.

    Args:
        segments: Segments in any order.

    Returns:
        Disjoint ``(start, end)`` intervals in time order.
    """
    merged: list[list[float]] = []
    for s in sorted(segments, key=lambda s: s.start_s):
        if merged and s.start_s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], s.end_s)
        else:
            merged.append([s.start_s, s.end_s])
    return [(a, b) for a, b in merged]


def overlaps(segment: Segment, intervals: Sequence[tuple[float, float]]) -> bool:
    """Whether a segment overlaps any interval.

    Args:
        segment: Segment.
        intervals: ``(start, end)`` intervals.

    Returns:
        True on a positive-length overlap.
    """
    return any(min(segment.end_s, b) > max(segment.start_s, a) for a, b in intervals)


def vtc_segments(paths: DatasetPaths, session: str, phase: str, model: str) -> list[Segment]:
    """VTC segments of a phase, empty if VTC has not run.

    Args:
        paths: Dataset locations.
        session: Session identifier.
        phase: Experiment phase.
        model: VTC model name.

    Returns:
        Segments with the VTC class as ``label``.
    """
    result = load_raw(paths, model, f"{session}/{phase}")
    if result is None or "voice_type" not in result.segments:
        return []
    return list(result.segments["voice_type"].segments)


@dataclass
class Enrolment:
    """The mother's enrolled voice.

    Attributes:
        status: ``enrolled`` or ``no_enrolment``.
        phase: Phase the enrolment speech came from, empty if none.
        seconds: Total enrolment speech.
        segments: ``(start, end)`` of the segments used.
        centroid: Unit-length mean embedding, empty if not enrolled.
    """

    status: str
    phase: str = ""
    seconds: float = 0.0
    segments: list[tuple[float, float]] = field(default_factory=list)
    centroid: list[float] = field(default_factory=list)


def _unit(v: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 and np.isfinite(v).all() else None


def enrol(
    candidates: dict[str, list[tuple[Segment, np.ndarray | None]]],
    baby: dict[str, list[tuple[float, float]]],
    config: RolesConfig,
) -> Enrolment:
    """Enrol the mother's voice from female segments.

    The experimenter may speak in the enrolment phase too, so the female speech is not simply
    averaged. Among the female segments without baby overlap, the *dominant voice* is the
    segment whose embedding has the largest duration-weighted similarity to all the others
    (the medoid; the mother speaks most). The enrolment then keeps the segments with
    similarity ``>= tau_high`` to it, longest first, up to ``enrol_total_s``, and their
    duration-weighted mean is the mother's voice.

    Args:
        candidates: Phase mapped to its FEM segments with their embeddings (None if missing).
        baby: Phase mapped to the merged baby intervals.
        config: Enrolment settings.

    Returns:
        The enrolment; ``no_enrolment`` if neither phase has enough clean speech of one voice.
    """
    for phase in (config.enrol_phase, config.fallback_phase):
        usable = [
            (segment, vector)
            for segment, vector in candidates.get(phase, [])
            if vector is not None and not overlaps(segment, baby.get(phase, []))
        ]
        if not usable:
            continue
        vectors = np.stack([v for _, v in usable])
        durations = np.array([s.end_s - s.start_s for s, _ in usable])
        medoid = vectors[int(np.argmax((vectors @ vectors.T) @ durations))]
        same = [
            (s, v)
            for (s, v), sim in zip(usable, vectors @ medoid, strict=True)
            if sim >= config.tau_high
        ]
        same.sort(key=lambda item: item[0].end_s - item[0].start_s, reverse=True)
        chosen, total = [], 0.0
        for segment, vector in same:
            if total >= config.enrol_total_s:
                break
            chosen.append((segment, vector))
            total += segment.end_s - segment.start_s
        if total + 1e-9 < config.enrol_min_s:
            continue
        weights = np.array([s.end_s - s.start_s for s, _ in chosen])
        centroid = _unit((np.stack([v for _, v in chosen]) * weights[:, None]).sum(axis=0))
        if centroid is not None:
            spans = sorted((s.start_s, s.end_s) for s, _ in chosen)
            return Enrolment("enrolled", phase, total, spans, centroid.tolist())
    return Enrolment("no_enrolment")


def classify_female(
    similarity: float | None, enrolled: bool, config: RolesConfig
) -> tuple[str, str, str, float]:
    """Role of a female segment.

    Args:
        similarity: Cosine similarity to the enrolled voice, None without embedding.
        enrolled: Whether the session has an enrolment.
        config: Thresholds.

    Returns:
        ``(role, label, rule_id, confidence)``.
    """
    if not enrolled:
        return "other", "FEM_unresolved", "ROLE-FEM-UNRESOLVED", 0.0
    if similarity is None or not np.isfinite(similarity):
        return "other", "FEM_uncertain", "ROLE-FEM-UNCERTAIN", 0.0
    confidence = float(np.clip(similarity, 0.0, 1.0))
    if similarity >= config.tau_high:
        return "mother", "FEM", "ROLE-FEM-MOTHER", confidence
    if similarity <= config.tau_low:
        return "other", "FEM", "ROLE-FEM-OTHER", 1.0 - confidence
    return "other", "FEM_uncertain", "ROLE-FEM-UNCERTAIN", confidence


@dataclass
class ListeningItem:
    """Something the user should listen to and annotate.

    Attributes:
        session: Session identifier.
        phase: Experiment phase, empty for session-level items.
        item: What it is (``uncertain_female``, ``male_adult``, ``no_enrolment`` ...).
        start_s: Phase-local start, NaN for session-level items.
        end_s: Phase-local end, NaN for session-level items.
        detail: Extra information (similarity, rule).
    """

    session: str
    phase: str
    item: str
    start_s: float = float("nan")
    end_s: float = float("nan")
    detail: str = ""


def roles_dir(paths: DatasetPaths, session: str) -> Path:
    """Folder of the role outputs of a session."""
    return paths.audio_session_dir(session) / "roles"


def role_events_path(paths: DatasetPaths, session: str, phase: str) -> Path:
    """Role events of a phase."""
    return roles_dir(paths, session) / f"{phase}.csv"


def assign_roles(
    paths: DatasetPaths, session: str, config: AudioConfig, phases: Sequence[str] = PHASES
) -> list[ListeningItem]:
    """Assign roles to every VTC segment of a session and write them.

    Writes ``roles/<phase>.csv`` (events) and ``roles/enrolment.json``.

    Args:
        paths: Dataset locations.
        session: Session identifier.
        config: Audio configuration.
        phases: Phases to process.

    Returns:
        The session's listening items.
    """
    rc = config.roles
    segments = {p: vtc_segments(paths, session, p, rc.vtc_model) for p in phases}
    baby = {
        p: merge_intervals([s for s in segs if s.label in rc.baby_labels])
        for p, segs in segments.items()
    }
    female: dict[str, list[tuple[Segment, np.ndarray | None]]] = {}
    for phase in phases:
        result = load_raw(paths, rc.embedding_model, f"{session}/{phase}")
        vectors: dict[tuple[float, float], np.ndarray | None] = {}
        if result is not None and "embedding" in result.segments:
            scores = result.segments["embedding"]
            embeddings = scores.embeddings
            for i, s in enumerate(scores.segments):
                vectors[(s.start_s, s.end_s)] = None if embeddings is None else _unit(embeddings[i])
        female[phase] = [
            (s, vectors.get((s.start_s, s.end_s))) for s in segments[phase] if s.label == "FEM"
        ]
    enrolment = enrol(female, baby, rc)
    centroid = np.array(enrolment.centroid) if enrolment.centroid else None
    items: list[ListeningItem] = []
    if centroid is None:
        items.append(ListeningItem(session, "", "no_enrolment", detail="ROLE-FEM-UNRESOLVED"))
    for phase in phases:
        events = [
            Event(a, b, "baby", "baby", 1.0, "roles", None, "ROLE-BABY") for a, b in baby[phase]
        ]
        for s in segments[phase]:
            if s.label == "MAL":
                events.append(
                    Event(s.start_s, s.end_s, "other", "MAL", 1.0, "roles", None, "ROLE-MAL")
                )
                items.append(
                    ListeningItem(session, phase, "male_adult", s.start_s, s.end_s, "ROLE-MAL")
                )
        for s, vector in female[phase]:
            similarity = None
            if centroid is not None and vector is not None:
                similarity = float(vector @ centroid)
            role, label, rule, confidence = classify_female(similarity, centroid is not None, rc)
            threshold = rc.tau_high if rule == "ROLE-FEM-MOTHER" else None
            events.append(
                Event(s.start_s, s.end_s, role, label, confidence, "roles", threshold, rule)
            )
            if rule == "ROLE-FEM-UNCERTAIN":
                detail = "no embedding" if similarity is None else f"similarity {similarity:.3f}"
                items.append(
                    ListeningItem(session, phase, "uncertain_female", s.start_s, s.end_s, detail)
                )
        events.sort(key=lambda e: (e.start_s, e.role))
        write_events(events, role_events_path(paths, session, phase))
    text = json.dumps(asdict(enrolment), indent=2) + "\n"
    write_atomic(roles_dir(paths, session) / "enrolment.json", lambda tmp: tmp.write_text(text))
    logger.info("%s: enrolment %s, %d listening items", session, enrolment.status, len(items))
    return items


def role_segments(
    paths: DatasetPaths, session: str, phase: str, roles: Sequence[str]
) -> list[Segment]:
    """Role segments of a phase.

    Args:
        paths: Dataset locations.
        session: Session identifier.
        phase: Experiment phase.
        roles: Roles to keep (``mother``, ``baby``, ``other``).

    Returns:
        Segments with the role and the label, empty if roles were not assigned.
    """
    path = role_events_path(paths, session, phase)
    if not path.exists():
        return []
    return [
        Segment(e.start_s, e.end_s, e.role, e.label) for e in read_events(path) if e.role in roles
    ]


def segment_builder(source: str, vtc_model: str = "vtc") -> RequestBuilder:
    """Request builder for a segment model.

    Args:
        source: ``vtc:<labels>`` or ``roles:<roles>`` (comma-separated names).
        vtc_model: Name of the VTC model whose segments ``vtc:`` sources read.

    Returns:
        A builder giving one request per phase with the matching segments (None if there
        are none or no WAV).
    """
    kind, _, names = source.partition(":")
    wanted = tuple(n.strip() for n in names.split(",") if n.strip())
    if kind not in ("vtc", "roles") or not wanted:
        raise ValueError(f"Bad segment source {source!r}")

    def build(paths: DatasetPaths, session: str, phase: str) -> InferenceRequest | None:
        wav = wav_path(paths, session, phase)
        if not wav.exists():
            return None
        if kind == "vtc":
            found = [s for s in vtc_segments(paths, session, phase, vtc_model) if s.label in wanted]
        else:
            found = role_segments(paths, session, phase, wanted)
        if not found:
            return None
        return InferenceRequest(f"{session}/{phase}", wav, wav_duration(wav), tuple(found))

    return build


LISTENING_COLUMNS: tuple[str, ...] = ("session", "phase", "item", "start_s", "end_s", "detail")
"""Columns of the listening list CSV."""


def write_listening(items: Sequence[ListeningItem], path: Path) -> None:
    """Write listening items as CSV, atomically.

    Args:
        items: Items.
        path: Output file.
    """

    def write(tmp: Path) -> None:
        with tmp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LISTENING_COLUMNS)
            writer.writeheader()
            for item in items:
                row = asdict(item)
                for key in ("start_s", "end_s"):
                    row[key] = "" if np.isnan(row[key]) else round(row[key], 3)
                writer.writerow(row)

    write_atomic(path, write)
