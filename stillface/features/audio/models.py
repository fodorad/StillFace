"""Model adapters: one interface for every model of the audio pipeline.

A model adapter turns a batch of :class:`InferenceRequest` (a prepared WAV, optionally
with segments to score) into one :class:`InferenceResult` or :class:`InferenceError` per
request. Heavy models run in their own environments as worker processes
(:class:`~stillface.features.audio.subprocess_adapter.SubprocessAdapter`), light ones
in-process; the stages never know which.

This module is imported by the worker processes too, so it depends on nothing but the
standard library and numpy.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class Segment:
    """A time interval to score.

    Attributes:
        start_s: Phase-local start.
        end_s: Phase-local end.
        role: Role of the segment (``baby``, ``mother``, ...).
        label: Optional label, e.g. the VTC class.
    """

    start_s: float
    end_s: float
    role: str = ""
    label: str = ""


@dataclass(frozen=True)
class InferenceRequest:
    """One input of a model.

    Attributes:
        key: Identifier of the item, ``<session>/<phase>[/<track>]``.
        wav: Prepared 16 kHz mono WAV.
        duration_s: Duration of the WAV.
        segments: Segments to score; None for models that read the whole WAV.
        options: Per-request options.
    """

    key: str
    wav: Path
    duration_s: float
    segments: tuple[Segment, ...] | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameScores:
    """Scores on a time axis.

    Attributes:
        times: Frame centre times in seconds, shape ``(n,)``.
        names: Score names, length ``k``.
        values: Scores, shape ``(n, k)``.
    """

    times: np.ndarray
    names: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        """Validate shapes."""
        self.times = np.asarray(self.times, dtype=float).reshape(-1)
        self.values = np.asarray(self.values, dtype=float).reshape(len(self.times), -1)
        if self.values.shape[1] != len(self.names):
            raise ValueError(f"{self.values.shape[1]} score columns for {len(self.names)} names")


@dataclass
class SegmentScores:
    """Scores per segment.

    Attributes:
        segments: Scored segments, length ``m``.
        names: Score names, length ``k``.
        values: Scores, shape ``(m, k)``.
        embeddings: Optional vectors, shape ``(m, d)``.
    """

    segments: tuple[Segment, ...]
    names: tuple[str, ...]
    values: np.ndarray
    embeddings: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate shapes."""
        m = len(self.segments)
        self.values = np.asarray(self.values, dtype=float).reshape(m, len(self.names))
        if self.embeddings is not None:
            self.embeddings = np.asarray(self.embeddings, dtype=float).reshape(m, -1)


@dataclass
class InferenceResult:
    """Native output of a model for one request.

    Attributes:
        key: Request key.
        frames: Named frame-score groups.
        segments: Named segment-score groups.
        artefacts: Named text outputs (RTTM, transcript JSON, TextGrid) by file suffix.
        meta: Extra provenance reported by the model (device, model version).
    """

    key: str
    frames: dict[str, FrameScores] = field(default_factory=dict)
    segments: dict[str, SegmentScores] = field(default_factory=dict)
    artefacts: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceError:
    """A request a model could not process.

    Attributes:
        key: Request key.
        message: What went wrong.
    """

    key: str
    message: str


class ModelAdapter(Protocol):
    """Runs one model on batches of requests.

    Attributes:
        name: Model name, the folder of its raw outputs.
        worker_version: Version of the adapter or worker code.
        weights_revision: Exact weights revision (e.g. a Hugging Face commit hash).
    """

    name: str
    worker_version: str
    weights_revision: str

    def run(self, requests: list[InferenceRequest]) -> list[InferenceResult | InferenceError]:
        """Process requests, one result or error per request, in order.

        Args:
            requests: Inputs; the model is loaded once for the batch.

        Returns:
            One entry per request.
        """
        ...


def save_result(result: InferenceResult, path: Path) -> None:
    """Store a result as ``.npz`` (arrays) with its names and metadata embedded.

    Artefacts are stored inside the archive as text; :func:`write_artefacts` writes them
    out as files.

    Args:
        result: Result to store.
        path: ``.npz`` file; the parent folder must exist.
    """
    arrays: dict[str, Any] = {}
    header: dict[str, Any] = {"key": result.key, "meta": result.meta, "frames": {}, "segments": {}}
    for name, group in result.frames.items():
        arrays[f"frames/{name}/times"] = group.times
        arrays[f"frames/{name}/values"] = group.values
        header["frames"][name] = list(group.names)
    for name, group in result.segments.items():
        arrays[f"segments/{name}/bounds"] = np.array(
            [[s.start_s, s.end_s] for s in group.segments], dtype=float
        ).reshape(-1, 2)
        arrays[f"segments/{name}/values"] = group.values
        if group.embeddings is not None:
            arrays[f"segments/{name}/embeddings"] = group.embeddings
        header["segments"][name] = {
            "names": list(group.names),
            "roles": [s.role for s in group.segments],
            "labels": [s.label for s in group.segments],
        }
    header["artefacts"] = result.artefacts
    arrays["header"] = np.array(json.dumps(header, ensure_ascii=False))
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def load_result(path: Path) -> InferenceResult:
    """Load a result stored by :func:`save_result`.

    Args:
        path: ``.npz`` file.

    Returns:
        The result.
    """
    with np.load(path, allow_pickle=False) as data:
        header = json.loads(str(data["header"]))
        frames = {
            name: FrameScores(
                data[f"frames/{name}/times"], tuple(names), data[f"frames/{name}/values"]
            )
            for name, names in header["frames"].items()
        }
        segments = {}
        for name, info in header["segments"].items():
            bounds = data[f"segments/{name}/bounds"]
            segs = tuple(
                Segment(float(a), float(b), role, label)
                for (a, b), role, label in zip(bounds, info["roles"], info["labels"], strict=True)
            )
            key = f"segments/{name}/embeddings"
            segments[name] = SegmentScores(
                segs,
                tuple(info["names"]),
                data[f"segments/{name}/values"],
                data[key] if key in data.files else None,
            )
    return InferenceResult(header["key"], frames, segments, header["artefacts"], header["meta"])


def request_to_dict(request: InferenceRequest) -> dict[str, Any]:
    """JSON form of a request, for worker processes.

    Args:
        request: Request.

    Returns:
        Dictionary with plain types.
    """
    segments = None
    if request.segments is not None:
        segments = [[s.start_s, s.end_s, s.role, s.label] for s in request.segments]
    return {
        "key": request.key,
        "wav": str(request.wav),
        "duration_s": request.duration_s,
        "segments": segments,
        "options": request.options,
    }


def request_from_dict(data: dict[str, Any]) -> InferenceRequest:
    """Rebuild a request from :func:`request_to_dict`.

    Args:
        data: Dictionary read from JSON.

    Returns:
        The request.
    """
    segments = None
    if data["segments"] is not None:
        segments = tuple(Segment(float(a), float(b), r, lab) for a, b, r, lab in data["segments"])
    return InferenceRequest(
        data["key"], Path(data["wav"]), float(data["duration_s"]), segments, data["options"]
    )
