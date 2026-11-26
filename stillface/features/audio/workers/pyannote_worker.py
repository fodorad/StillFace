"""pyannote worker: speech activity and speaker embeddings (environment ``vad``).

Two outputs from the pyannote community-1 models (Bredin et al.):

- ``speech`` frame scores (whole-phase requests): the segmentation model's per-frame
  speaker activities, the maximum over its local speakers being the probability that anyone
  speaks. It is the third voice-activity opinion next to VTC and Silero.
- ``embedding`` segment vectors (segment requests): one L2-normalised speaker embedding per
  segment, used to tell the mother from other women by similarity to her enrolled voice.

The community-1 models are gated on Hugging Face; the token comes from ``HF_TOKEN``.
"""

import os
import sys
from typing import Any

import numpy as np

from stillface.features.audio.media import read_wav
from stillface.features.audio.models import (
    FrameScores,
    InferenceRequest,
    InferenceResult,
    Segment,
    SegmentScores,
)
from stillface.features.audio.workers.base import serve, torch_device

REPO: str = "pyannote/speaker-diarization-community-1"
"""Hugging Face repository holding the segmentation and embedding models."""

MIN_EMBEDDING_S: float = 0.5
"""Shorter segments get a NaN embedding."""


def max_over_speakers(scores: np.ndarray) -> np.ndarray:
    """Speech activity of each chunk frame: the maximum over the local speakers.

    The segmentation model is permutation-invariant (its speaker channels mean nothing
    across chunks), so pyannote only aggregates overlapping chunks after this hook has
    reduced them to one speaker-free channel, as its own voice activity pipeline does.

    Args:
        scores: Chunk outputs, shape ``(chunks, frames, speakers)``.

    Returns:
        Shape ``(chunks, frames, 1)``.
    """
    return np.max(scores, axis=-1, keepdims=True)


def speech_scores(start: float, step: float, duration: float, activity: np.ndarray) -> FrameScores:
    """Speech probability from aggregated per-frame activities.

    Args:
        start: Start of the first frame.
        step: Frame step.
        duration: Frame duration.
        activity: Aggregated activities, shape ``(n,)`` or ``(n, channels)``.

    Returns:
        ``speech`` = maximum activity, at the frame centres, clipped to ``[0, 1]``.

    Raises:
        ValueError: If the activities are still per chunk (3-D, not aggregated).
    """
    activity = np.asarray(activity, dtype=float)
    if activity.ndim == 3:
        raise ValueError(
            f"Activities of shape {activity.shape} are per chunk; aggregate them first "
            "(segmentation Inference with pre_aggregation_hook=max_over_speakers)"
        )
    if activity.ndim == 1:
        activity = activity[:, None]
    if activity.size:
        speech = np.clip(np.nan_to_num(activity).max(axis=1), 0.0, 1.0)
    else:
        speech = np.zeros(len(activity))
    times = start + np.arange(len(activity)) * step + duration / 2
    return FrameScores(times, ("speech",), speech)


def embedding_scores(
    segments: tuple[Segment, ...], vectors: list[np.ndarray | None]
) -> SegmentScores:
    """L2-normalised embeddings per segment.

    Args:
        segments: Requested segments.
        vectors: One vector per segment, None where it could not be computed.

    Returns:
        Segment scores with ``duration_s`` as value and the embeddings (NaN rows for None).

    Raises:
        ValueError: If the counts disagree.
    """
    if len(vectors) != len(segments):
        raise ValueError(f"{len(vectors)} embeddings for {len(segments)} segments")
    known = [v for v in vectors if v is not None]
    dim = len(np.asarray(known[0]).reshape(-1)) if known else 1
    rows = []
    for vector in vectors:
        if vector is None:
            rows.append(np.full(dim, np.nan))
            continue
        v = np.asarray(vector, dtype=float).reshape(-1)
        norm = np.linalg.norm(v)
        rows.append(v / norm if norm > 0 else np.full(dim, np.nan))
    durations = np.array([[s.end_s - s.start_s] for s in segments]).reshape(-1, 1)
    embeddings = np.array(rows).reshape(len(segments), dim)
    return SegmentScores(segments, ("duration_s",), durations, embeddings)


class PyannoteWorker:
    """Speech activity for whole phases, embeddings for segment requests."""

    def load(self, options: dict[str, Any]) -> dict[str, Any]:
        """Load the segmentation and embedding models at the pinned revision."""
        import torch
        from pyannote.audio import Inference, Model

        token = os.environ.get("HF_TOKEN")
        revision = options.get("weights_revision") or None
        repo = options.get("repo", REPO)
        self.device = torch_device(options.get("device", "cpu"))
        device = torch.device(self.device)
        segmentation = Model.from_pretrained(
            repo, subfolder="segmentation", revision=revision, token=token
        )
        embedding = Model.from_pretrained(
            repo, subfolder="embedding", revision=revision, token=token
        )
        self.segmentation = Inference(
            segmentation, device=device, pre_aggregation_hook=max_over_speakers
        )
        self.embedding = Inference(embedding, window="whole", device=device)
        return {"device": self.device, "repo": repo, "revision": revision}

    def forward(self, request: InferenceRequest) -> FrameScores | list[np.ndarray | None]:
        """Speech scores of a whole phase, or embeddings of the requested segments."""
        import torch
        from pyannote.core import Segment as Interval

        samples, rate = read_wav(request.wav)
        audio = {"waveform": torch.from_numpy(samples[None, :]), "sample_rate": rate}
        if request.segments is None:
            out = self.segmentation(audio)
            window = out.sliding_window
            return speech_scores(window.start, window.step, window.duration, out.data)
        vectors: list[np.ndarray | None] = []
        for s in request.segments:
            if s.end_s - s.start_s < MIN_EMBEDDING_S:
                vectors.append(None)
                continue
            vectors.append(np.asarray(self.embedding.crop(audio, Interval(s.start_s, s.end_s))))
        return vectors

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Run the request.

        Args:
            request: Whole-phase request (speech) or segment request (embeddings).

        Returns:
            ``speech`` frame scores or ``embedding`` segment scores.
        """
        out = self.forward(request)
        if isinstance(out, FrameScores):
            return InferenceResult(request.key, {"speech": out})
        segments = request.segments or ()
        return InferenceResult(request.key, segments={"embedding": embedding_scores(segments, out)})


if __name__ == "__main__":
    sys.exit(serve(PyannoteWorker()))
