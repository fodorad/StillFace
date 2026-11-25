"""Hand-written fakes for the audio tests (no mocks)."""

from dataclasses import dataclass, field

import numpy as np

from stillface.features.audio.models import (
    FrameScores,
    InferenceError,
    InferenceRequest,
    InferenceResult,
    SegmentScores,
)


@dataclass
class FakeAdapter:
    """Deterministic model: per request, frame scores at ``rate_hz`` and segment scores.

    Frame score ``names[j]`` at time ``t`` is ``scores[j]`` (constant) unless ``frame_fn`` is
    given. Keys in ``fail`` give an error. Every call is recorded in ``calls``.
    """

    name: str = "fake"
    names: tuple[str, ...] = ("speech",)
    scores: tuple[float, ...] = (0.8,)
    rate_hz: float = 10.0
    worker_version: str = "1"
    weights_revision: str = "rev-1"
    fail: set[str] = field(default_factory=set)
    frame_fn: object = None
    segment_names: tuple[str, ...] = ("p",)
    segment_value: float = 0.5
    embedding: dict[str, np.ndarray] = field(default_factory=dict)
    calls: list[list[str]] = field(default_factory=list)

    def run(self, requests: list[InferenceRequest]) -> list[InferenceResult | InferenceError]:
        self.calls.append([r.key for r in requests])
        out: list[InferenceResult | InferenceError] = []
        for request in requests:
            if request.key in self.fail:
                out.append(InferenceError(request.key, "fake failure"))
                continue
            times = np.arange(0, request.duration_s + 1e-9, 1 / self.rate_hz)
            if callable(self.frame_fn):
                values = np.array([self.frame_fn(t) for t in times], dtype=float)
            else:
                values = np.tile(np.array(self.scores, dtype=float), (len(times), 1))
            result = InferenceResult(
                request.key,
                {"scores": FrameScores(times, self.names, values)},
                meta={"device": "cpu"},
            )
            if request.segments is not None:
                emb = None
                if self.embedding:
                    emb = (
                        np.stack([self.embedding[s.label] for s in request.segments])
                        if request.segments
                        else np.zeros((0, 2))
                    )
                result.segments["scores"] = SegmentScores(
                    request.segments,
                    self.segment_names,
                    np.full((len(request.segments), len(self.segment_names)), self.segment_value),
                    emb,
                )
            out.append(result)
        return out
