"""Exports: Annie files, QC records and the dataset tables.

Per session and phase:

- ``annie/<sid>_<phase>.json``: events in Annie's own nested export shape
  (``video_id -> {fps, tracks: {track: [event, ...]}, meta}``); a track is
  ``<role>:<source>`` (e.g. ``baby:fusion``) and bands become events too (voice activity,
  silence, gong, overlap, roles);
- ``annie/<sid>_<phase>_lanes.csv`` + ``_lanes.json``: every lane and band (fusion and
  affect) on the 10 Hz grid, with its legend.

Per session ``qc.json`` (stage completeness, coverage, disagreement, flags), and for the
dataset ``qc_summary.csv`` plus ``listening.csv`` (every flagged item exactly once).
"""

import csv
import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from stillface.config import DatasetPaths
from stillface.constants import PHASES, TARGET_FPS
from stillface.features.audio.config import AudioConfig
from stillface.features.audio.fusion import fusion_paths
from stillface.features.audio.infer import raw_path
from stillface.features.audio.lanes import (
    Band,
    Event,
    LaneTable,
    read_events,
    read_lanes,
    runs,
    write_lanes,
)
from stillface.features.audio.prepare import wav_path
from stillface.features.audio.provenance import read_sidecar
from stillface.features.audio.roles import ListeningItem, role_events_path, write_listening
from stillface.features.io import write_atomic

logger = logging.getLogger(__name__)

BAND_TRACKS: dict[str, tuple[str, str]] = {
    "fusion.vad.dyad": ("dyad:vad", "voice"),
    "fusion.silence.dyad": ("dyad:vad", "silence"),
    "fusion.vad_disagree.dyad": ("dyad:vad", "disagreement"),
    "fusion.gong.context": ("context:gong", "gong"),
    "fusion.overlap.dyad": ("dyad:overlap", "overlap"),
}
"""Bands exported as Annie events: band name mapped to ``(track, label)``."""


def video_id(session: str, camera: str, phase: str) -> str:
    """Annie video id of a phase clip (``<sid>_<camera>_<phase>``)."""
    return f"{session}_{camera}_{phase}"


def band_events(
    band: Band, track: str, label: str, rate_hz: float, fps: float = TARGET_FPS
) -> list[dict]:
    """Runs of a band as Annie event records.

    Args:
        band: Band on the grid.
        track: Annie track.
        label: Event label.
        rate_hz: Grid rate.
        fps: Frame rate of the Annie video.

    Returns:
        Event records with frames at ``fps``.
    """
    out = []
    for start, stop in runs(band.values):
        out.append(
            annie_event(
                Event(
                    start / rate_hz,
                    stop / rate_hz,
                    track.split(":")[0],
                    label,
                    1.0,
                    "fusion",
                    band.spec.threshold,
                    band.spec.rule_id,
                ),
                track,
                fps,
            )
        )
    return out


def annie_event(event: Event, track: str, fps: float = TARGET_FPS) -> dict:
    """One event in Annie's export record layout.

    Args:
        event: Event.
        track: Annie track.
        fps: Frame rate of the Annie video.

    Returns:
        Record with frames and seconds, label, type and the provenance as attributes.
    """
    return {
        "track": track,
        "start_frame": int(round(event.start_s * fps)),
        "end_frame": int(round(event.end_s * fps)),
        "start_sec": round(event.start_s, 3),
        "end_sec": round(event.end_s, 3),
        "label": event.label,
        "event_type": event.label,
        "note": "",
        "attributes": {
            "source": event.source,
            "confidence": round(event.confidence, 3),
            "threshold": event.threshold,
            "rule_id": event.rule_id,
        },
    }


def build_annie_tree(
    session: str,
    phase: str,
    camera: str,
    events: Sequence[Event],
    table: LaneTable,
    fps: float = TARGET_FPS,
) -> dict:
    """Annie export tree of one phase.

    Args:
        session: Session identifier.
        phase: Experiment phase.
        camera: Camera whose clip the audio came from (the Annie video).
        events: Role and fusion events.
        table: Fusion lane table (its exported bands become events).
        fps: Frame rate of the Annie video (the clip the audio came from).

    Returns:
        ``{video_id: {"fps", "tracks", "meta"}}``.
    """
    tracks: dict[str, list[dict]] = {}
    for event in events:
        track = f"{event.role}:{event.source}"
        record = annie_event(event, track, fps)
        record.pop("track")
        tracks.setdefault(track, []).append(record)
    for band in table.bands:
        if band.spec.name in BAND_TRACKS:
            track, label = BAND_TRACKS[band.spec.name]
            for record in band_events(band, track, label, table.rate_hz, fps):
                record.pop("track")
                tracks.setdefault(track, []).append(record)
    for records in tracks.values():
        records.sort(key=lambda r: (r["start_frame"], r["label"]))
        for number, record in enumerate(records, start=1):
            record["event_id"] = number
    meta = {"session": session, "phase": phase, "audio_camera": camera}
    return {video_id(session, camera, phase): {"fps": float(fps), "tracks": tracks, "meta": meta}}


def export_phase(paths: DatasetPaths, session: str, phase: str) -> Path | None:
    """Write the Annie files of one phase.

    Args:
        paths: Dataset locations.
        session: Session identifier.
        phase: Experiment phase.

    Returns:
        The Annie JSON path, None if the phase has no fusion output.
    """
    out = fusion_paths(paths, session, phase)
    if not out["lanes"].exists():
        return None
    table = read_lanes(out["lanes"], out["legend"])
    events = read_events(out["events"]) if out["events"].exists() else []
    roles = role_events_path(paths, session, phase)
    events = (read_events(roles) if roles.exists() else []) + events
    affect = paths.audio_session_dir(session) / "affect" / f"{phase}.csv"
    lanes = list(table.lanes)
    if affect.exists():
        lanes += read_lanes(affect, affect.with_suffix(".json")).lanes
    side = read_sidecar(wav_path(paths, session, phase)) or {}
    camera = str(side.get("camera", "baby"))
    fps = float(side.get("video_fps") or TARGET_FPS)
    folder = paths.audio_session_dir(session) / "annie"
    merged = LaneTable(table.duration_s, lanes, table.bands, table.rate_hz, dict(table.notes))
    write_lanes(
        merged, folder / f"{session}_{phase}_lanes.csv", folder / f"{session}_{phase}_lanes.json"
    )
    tree = build_annie_tree(session, phase, camera, events, table, fps)
    target = folder / f"{session}_{phase}.json"
    text = json.dumps(tree, indent=2, ensure_ascii=False) + "\n"
    write_atomic(target, lambda tmp: tmp.write_text(text))
    return target


def _share(table: LaneTable, name: str) -> float:
    try:
        return float(table.band(name).values.mean())
    except KeyError:
        return math.nan


def session_qc(
    paths: DatasetPaths, session: str, config: AudioConfig, phases: Sequence[str] = PHASES
) -> dict:
    """QC record of a session.

    Args:
        paths: Dataset locations.
        session: Session identifier.
        config: Audio configuration (enabled models).
        phases: Phases checked.

    Returns:
        Per stage the phases done or missing, per phase the key shares, and the flags.
    """
    prepared = [p for p in phases if wav_path(paths, session, p).exists()]
    record: dict = {
        "session": session,
        "prepared": prepared,
        "models": {},
        "phases": {},
        "flags": [],
    }
    for name, model in config.models.items():
        if not model.enabled:
            record["models"][name] = "disabled"
            continue
        track = model.options.get("track", "")
        done = [
            p
            for p in prepared
            if raw_path(paths, name, f"{session}/{p}" + (f"/{track}" if track else "")).exists()
        ]
        record["models"][name] = {"done": done, "missing": [p for p in prepared if p not in done]}
    enrolment = paths.audio_session_dir(session) / "roles" / "enrolment.json"
    record["enrolment"] = (
        json.loads(enrolment.read_text())["status"] if enrolment.exists() else "not_run"
    )
    if record["enrolment"] != "enrolled":
        record["flags"].append(f"enrolment:{record['enrolment']}")
    for phase in prepared:
        info: dict = {}
        roles = role_events_path(paths, session, phase)
        events = read_events(roles) if roles.exists() else []
        info["uncertain_female"] = sum(e.label == "FEM_uncertain" for e in events)
        out = fusion_paths(paths, session, phase)
        if out["lanes"].exists():
            table = read_lanes(out["lanes"], out["legend"])
            for key, band in (
                ("voice_share", "fusion.vad.dyad"),
                ("silence_share", "fusion.silence.dyad"),
                ("disagreement_share", "fusion.vad_disagree.dyad"),
                ("gong_share", "fusion.gong.context"),
                ("baby_share", "roles.active.baby"),
                ("mother_share", "roles.active.mother"),
                ("overlap_share", "fusion.overlap.dyad"),
            ):
                info[key] = _share(table, band)
            summary = json.loads(out["summary"].read_text()) if out["summary"].exists() else {}
            info["context_shares"] = summary.get("context_shares", {})
        else:
            record["flags"].append(f"{phase}:not_fused")
        asr = paths.audio_session_dir(session) / "asr" / f"{phase}.json"
        if asr.exists():
            info["outside_mother_share"] = json.loads(asr.read_text()).get("outside_mother_share")
        record["phases"][phase] = info
    affect = paths.audio_session_dir(session) / "affect" / "summary.json"
    record["affect_agreement"] = (
        json.loads(affect.read_text())["agreement"] if affect.exists() else None
    )
    for name, status in record["models"].items():
        if isinstance(status, dict) and status["missing"]:
            record["flags"].append(f"{name}:missing:{','.join(status['missing'])}")
    text = json.dumps(record, indent=2, default=lambda v: None) + "\n"
    write_atomic(paths.audio_session_dir(session) / "qc.json", lambda tmp: tmp.write_text(text))
    return record


QC_COLUMNS: tuple[str, ...] = (
    "session",
    "prepared_phases",
    "models_complete",
    "models_incomplete",
    "enrolment",
    "uncertain_female",
    "mean_disagreement_share",
    "mean_outside_mother_share",
    "flags",
)
"""Columns of the dataset QC summary."""


def qc_row(record: dict) -> dict:
    """One summary row of a session QC record.

    Args:
        record: Output of :func:`session_qc`.

    Returns:
        Row with :data:`QC_COLUMNS`.
    """
    models = [s for s in record["models"].values() if isinstance(s, dict)]
    phases = record["phases"].values()

    def mean(key: str) -> float | str:
        values = [
            p[key] for p in phases if isinstance(p.get(key), int | float) and not math.isnan(p[key])
        ]
        return round(float(np.mean(values)), 4) if values else ""

    return {
        "session": record["session"],
        "prepared_phases": len(record["prepared"]),
        "models_complete": sum(not s["missing"] for s in models),
        "models_incomplete": sum(bool(s["missing"]) for s in models),
        "enrolment": record["enrolment"],
        "uncertain_female": sum(p.get("uncertain_female", 0) for p in phases),
        "mean_disagreement_share": mean("disagreement_share"),
        "mean_outside_mother_share": mean("outside_mother_share"),
        "flags": len(record["flags"]),
    }


def read_listening(path: Path) -> list[ListeningItem]:
    """Read a listening CSV written by :func:`~stillface.features.audio.roles.write_listening`.

    Args:
        path: CSV file; missing files give an empty list.

    Returns:
        The items.
    """
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return [
            ListeningItem(
                r["session"],
                r["phase"],
                r["item"],
                float(r["start_s"]) if r["start_s"] else math.nan,
                float(r["end_s"]) if r["end_s"] else math.nan,
                r["detail"],
            )
            for r in csv.DictReader(handle)
        ]


def unique_items(items: Sequence[ListeningItem]) -> list[ListeningItem]:
    """Drop repeated listening items (same session, phase, item and interval).

    Args:
        items: Items in any order.

    Returns:
        First occurrences, sorted by session, phase and start.
    """
    seen, out = set(), []
    for item in items:
        key = (
            item.session,
            item.phase,
            item.item,
            round(item.start_s, 3) if not math.isnan(item.start_s) else None,
        )
        if key not in seen:
            seen.add(key)
            out.append(item)
    return sorted(
        out, key=lambda i: (i.session, i.phase, -1 if math.isnan(i.start_s) else i.start_s, i.item)
    )


def write_dataset_tables(paths: DatasetPaths, records: Sequence[dict]) -> None:
    """Write ``qc_summary.csv`` and the merged ``listening.csv``.

    Args:
        paths: Dataset locations.
        records: Session QC records.
    """

    def write(tmp: Path) -> None:
        with tmp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=QC_COLUMNS)
            writer.writeheader()
            for record in records:
                writer.writerow(qc_row(record))

    write_atomic(paths.audio_dir / "qc_summary.csv", write)
    items = []
    for name in ("listening_roles.csv", "listening_fusion.csv"):
        items += read_listening(paths.audio_dir / name)
    write_listening(unique_items(items), paths.audio_dir / "listening.csv")
    logger.info("QC of %d sessions, %d listening items", len(records), len(items))
