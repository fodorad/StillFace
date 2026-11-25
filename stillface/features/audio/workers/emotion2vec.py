"""emotion2vec+ worker: nine emotion category probabilities per window (environment ``e2v``).

emotion2vec+ large (Ma et al. 2024) is a data2vec-style speech emotion model fine-tuned on
about 40k h of pseudo-labelled speech. Input: the raw 16 kHz waveform of one window;
output: probabilities of nine categories and an utterance embedding. It cross-checks the
audEERING dimensions for the mother (valence and arousal are derived from the categories in
the affect stage).
"""

import sys
from typing import Any

import numpy as np

from stillface.features.audio.media import read_wav
from stillface.features.audio.models import (
    InferenceRequest,
    InferenceResult,
    Segment,
    SegmentScores,
)
from stillface.features.audio.workers.base import segment_windows, serve, torch_device

MODEL_ID: str = "emotion2vec/emotion2vec_plus_large"
"""Hugging Face model id."""

CLASSES: tuple[str, ...] = (
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "unknown",
)
"""Output categories in model order (``tokens.txt``; ``<unk>`` is ``unknown``)."""


def class_name(token: str) -> str:
    """English category of a model token such as ``开心/happy`` or ``<unk>``.

    Args:
        token: Model label.

    Returns:
        One of :data:`CLASSES`.

    Raises:
        ValueError: If the token is not a known category.
    """
    name = "unknown" if token == "<unk>" else token.split("/")[-1].strip().lower()
    if name not in CLASSES:
        raise ValueError(f"Unknown emotion2vec label {token!r}")
    return name


def postprocess(
    windows: tuple[Segment, ...],
    labels: list[list[str]],
    scores: list[list[float]],
    embeddings: np.ndarray,
) -> SegmentScores:
    """Window probabilities in :data:`CLASSES` order, renormalised to sum to one.

    Args:
        windows: Scored windows.
        labels: Model labels per window.
        scores: Scores per window, aligned with ``labels``.
        embeddings: Utterance embeddings, shape ``(n, d)``.

    Returns:
        The window scores.

    Raises:
        ValueError: If counts disagree.
    """
    if not len(labels) == len(scores) == len(windows) == len(embeddings):
        raise ValueError(f"{len(labels)} outputs for {len(windows)} windows")
    values = np.zeros((len(windows), len(CLASSES)))
    for i, (names, row) in enumerate(zip(labels, scores, strict=True)):
        for token, score in zip(names, row, strict=True):
            values[i, CLASSES.index(class_name(token))] = float(score)
        total = values[i].sum()
        if total > 0:
            values[i] /= total
    return SegmentScores(windows, CLASSES, values, np.asarray(embeddings, dtype=float))


class Emotion2vecWorker:
    """Scores windows inside the requested segments."""

    def load(self, options: dict[str, Any]) -> dict[str, Any]:
        """Load emotion2vec+ through FunASR from the Hugging Face hub."""
        from funasr import AutoModel

        self.window_s = float(options.get("window_s", "3.0")) or 3.0
        self.hop_s = float(options.get("hop_s", "0.5")) or 0.5
        self.device = torch_device(options.get("device", "cpu"))
        revision = options.get("weights_revision") or None
        self.model = AutoModel(
            model=MODEL_ID,
            hub="hf",
            device=self.device,
            model_revision=revision,
            disable_update=True,
        )
        return {"device": self.device, "model_id": MODEL_ID, "revision": revision}

    def forward(
        self, clips: list[np.ndarray], rate: int
    ) -> tuple[list[list[str]], list[list[float]], np.ndarray]:
        """Labels, scores and embeddings for each clip."""
        labels, scores, feats = [], [], []
        for clip in clips:
            out = self.model.generate(
                clip, granularity="utterance", extract_embedding=True, fs=rate
            )[0]
            labels.append(list(out["labels"]))
            scores.append([float(v) for v in out["scores"]])
            feats.append(np.asarray(out["feats"], dtype=float).reshape(-1))
        dim = len(feats[0]) if feats else 1024
        return labels, scores, np.array(feats).reshape(-1, dim)

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Emotion probabilities of every window of the requested segments.

        Args:
            request: Segment request.

        Returns:
            ``windows`` segment scores.
        """
        samples, rate = read_wav(request.wav)
        windows = segment_windows(request.segments or (), self.window_s, self.hop_s)
        clips = [samples[round(w.start_s * rate) : round(w.end_s * rate)] for w in windows]
        labels, scores, feats = self.forward(clips, rate)
        return InferenceResult(
            request.key, segments={"windows": postprocess(windows, labels, scores, feats)}
        )


if __name__ == "__main__":
    sys.exit(serve(Emotion2vecWorker()))
