"""Inference stage: run one model over every pending session phase.

The stage is model-major: it collects every request whose output is missing or stale,
hands them to the adapter in one batch (one model load), and stores each result as
``raw/<model>/<phase>[_<track>].npz`` with its text artefacts next to it and a provenance
sidecar. An output is stale when the input WAV, the scored segments, the model settings, the
worker version or the weights revision changed.
"""

import hashlib
import json
import logging
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from stillface.config import DatasetPaths
from stillface.constants import PHASES
from stillface.features.audio.config import ModelConfig
from stillface.features.audio.media import read_wav
from stillface.features.audio.models import (
    InferenceError,
    InferenceRequest,
    InferenceResult,
    ModelAdapter,
    load_result,
    save_result,
)
from stillface.features.audio.prepare import wav_path
from stillface.features.audio.provenance import (
    file_sha256,
    is_up_to_date,
    read_sidecar,
    write_sidecar,
)
from stillface.features.io import write_atomic

logger = logging.getLogger(__name__)

RequestBuilder = Callable[[DatasetPaths, str, str], InferenceRequest | None]
"""Builds the request of ``(paths, session, phase)``, None if its input is not available."""


def split_key(key: str) -> tuple[str, str, str]:
    """Split a request key.

    Args:
        key: ``<session>/<phase>[/<track>]``.

    Returns:
        ``(session, phase, track)``, track empty for the full phase audio.
    """
    parts = key.split("/")
    if len(parts) not in (2, 3):
        raise ValueError(f"Malformed request key {key!r}")
    return parts[0], parts[1], parts[2] if len(parts) == 3 else ""


def raw_path(paths: DatasetPaths, model: str, key: str) -> Path:
    """Raw output of a model for a request key.

    Args:
        paths: Dataset locations.
        model: Model name.
        key: Request key.

    Returns:
        ``<audio session dir>/raw/<model>/<phase>[_<track>].npz``.
    """
    session, phase, track = split_key(key)
    name = f"{phase}_{track}" if track else phase
    return paths.audio_session_dir(session) / "raw" / model / f"{name}.npz"


def wav_duration(path: Path) -> float:
    """Duration of a prepared WAV, from its sidecar if present.

    Args:
        path: Prepared WAV.

    Returns:
        Seconds.
    """
    side = read_sidecar(path)
    if side and "duration_s" in side:
        return float(side["duration_s"])
    samples, rate = read_wav(path)
    return len(samples) / rate


def full_wav_request(paths: DatasetPaths, session: str, phase: str) -> InferenceRequest | None:
    """Request for the whole prepared WAV of a phase (first-pass models).

    Args:
        paths: Dataset locations.
        session: Session identifier.
        phase: Experiment phase.

    Returns:
        The request, None if the WAV was not prepared.
    """
    wav = wav_path(paths, session, phase)
    if not wav.exists():
        return None
    return InferenceRequest(f"{session}/{phase}", wav, wav_duration(wav))


def segments_digest(request: InferenceRequest) -> str | None:
    """Hash of the scored segments, None for whole-WAV requests.

    Args:
        request: Request.

    Returns:
        Short SHA-256 of the segment list.
    """
    if request.segments is None:
        return None
    text = json.dumps([asdict(s) for s in request.segments])
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def expected_provenance(
    adapter: ModelAdapter, model: ModelConfig, request: InferenceRequest
) -> dict[str, Any]:
    """Sidecar keys that must match for an output to be up to date.

    Args:
        adapter: Model adapter.
        model: Model settings.
        request: Request of the output.

    Returns:
        Input hash, segment hash, model name and settings digest, worker version and
        weights revision.
    """
    return {
        "model": adapter.name,
        "input_sha256": file_sha256(request.wav),
        "segments_sha256": segments_digest(request),
        "model_config": model.digest,
        "worker_version": adapter.worker_version,
        "weights_revision": adapter.weights_revision,
    }


@dataclass
class StageReport:
    """Outcome of an inference stage.

    Attributes:
        model: Model name.
        written: Request keys whose output was written.
        kept: Request keys already up to date.
        skipped: ``session/phase`` items without input.
        failed: Request key mapped to the error message.
    """

    model: str
    written: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


def run_model(
    paths: DatasetPaths,
    adapter: ModelAdapter,
    model: ModelConfig,
    sessions: Iterable[str],
    phases: Sequence[str] = PHASES,
    builder: RequestBuilder = full_wav_request,
    overwrite: bool = False,
) -> StageReport:
    """Run a model on every pending session phase in one batch.

    Args:
        paths: Dataset locations.
        adapter: Model adapter.
        model: Model settings (their digest goes into the sidecars).
        sessions: Sessions to process.
        phases: Phases to process.
        builder: Request builder; whole prepared WAVs by default.
        overwrite: Recompute up-to-date outputs.

    Returns:
        The report, also written to ``<audio dir>/runs/<model>.json``.
    """
    report = StageReport(adapter.name)
    pending: list[tuple[InferenceRequest, Path, dict[str, Any]]] = []
    for session in sessions:
        for phase in phases:
            request = builder(paths, session, phase)
            if request is None:
                report.skipped.append(f"{session}/{phase}")
                continue
            out = raw_path(paths, adapter.name, request.key)
            expected = expected_provenance(adapter, model, request)
            if is_up_to_date(out, expected, overwrite):
                report.kept.append(request.key)
            else:
                pending.append((request, out, expected))
    if pending:
        started = time.time()
        results = adapter.run([request for request, _, _ in pending])
        finished = time.time()
        for (request, out, expected), result in zip(pending, results, strict=True):
            if isinstance(result, InferenceError):
                logger.error("%s failed on %s: %s", adapter.name, request.key, result.message)
                report.failed[request.key] = result.message
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            write_atomic(out, lambda tmp, r=result: save_result(r, tmp))
            for suffix, text in result.artefacts.items():
                artefact = out.with_suffix(f".{suffix}")
                write_atomic(artefact, lambda tmp, t=text: tmp.write_text(t))
            wav = request.wav.relative_to(paths.processed_dir).as_posix()
            write_sidecar(
                out,
                {
                    **expected,
                    "key": request.key,
                    "input": wav,
                    "started": started,
                    "finished": finished,
                    "meta": result.meta,
                    "artefacts": sorted(result.artefacts),
                },
            )
            report.written.append(request.key)
    logger.info(
        "%s: wrote %d, kept %d, skipped %d, failed %d",
        adapter.name,
        len(report.written),
        len(report.kept),
        len(report.skipped),
        len(report.failed),
    )
    text = json.dumps(asdict(report), indent=2) + "\n"
    write_atomic(paths.audio_dir / "runs" / f"{adapter.name}.json", lambda t: t.write_text(text))
    return report


def load_raw(paths: DatasetPaths, model: str, key: str) -> InferenceResult | None:
    """Load a stored raw output.

    Args:
        paths: Dataset locations.
        model: Model name.
        key: Request key.

    Returns:
        The stored result, None if missing.
    """
    out = raw_path(paths, model, key)
    return load_result(out) if out.exists() else None
