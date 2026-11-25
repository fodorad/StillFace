"""Addressee worker: who an adult speaks to (environment ``vtc``).

The LAAC-LSCP addressee classifier (Charlot et al., Interspeech 2026) is BabyHuBERT
fine-tuned to label adult (``FEM`` / ``MAL``) utterances as ``KCDS`` (key-child-directed,
i.e. infant-directed speech), ``ADS`` (adult-directed) or ``Other`` (overheard, unsure). It
sees each utterance with 10 s of context and was trained on utterances up to 30 s, so longer
segments are split before scoring. There is no audio-only equivalent for the baby.
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
from stillface.features.audio.workers.base import serve, torch_device

MODEL_ORDER: tuple[str, ...] = ("ADS", "KCDS", "Other")
"""Class order of the model's logits (``addressee.dataloaders.id2label``)."""

NAMES: tuple[str, ...] = ("KCDS", "ADS", "Other")
"""Output class order."""

ADULT_LABELS: tuple[str, ...] = ("FEM", "MAL")
"""VTC labels the model scores."""

MAX_UTTERANCE_S: float = 30.0
"""Longest utterance seen in training."""


def split_long_segments(
    segments: tuple[Segment, ...], max_s: float = MAX_UTTERANCE_S
) -> tuple[Segment, ...]:
    """Split segments longer than ``max_s`` into equal parts of at most ``max_s``.

    Args:
        segments: Segments.
        max_s: Longest allowed duration.

    Returns:
        Segments in order, long ones replaced by their parts (same role and label).
    """
    out = []
    for segment in segments:
        duration = segment.end_s - segment.start_s
        parts = max(1, int(np.ceil(duration / max_s - 1e-9)))
        edges = np.linspace(segment.start_s, segment.end_s, parts + 1)
        out += [
            Segment(float(a), float(b), segment.role, segment.label)
            for a, b in zip(edges[:-1], edges[1:], strict=True)
        ]
    return tuple(out)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax.

    Args:
        logits: Shape ``(m, k)``.

    Returns:
        Probabilities of the same shape.
    """
    z = np.asarray(logits, dtype=float)
    z = z - z.max(axis=1, keepdims=True) if z.size else z
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True) if z.size else e


def postprocess(segments: tuple[Segment, ...], logits: np.ndarray) -> SegmentScores:
    """Probabilities per segment in :data:`NAMES` order.

    Args:
        segments: Scored adult segments.
        logits: Model logits, shape ``(len(segments), 3)`` in :data:`MODEL_ORDER`.

    Returns:
        The segment scores.

    Raises:
        ValueError: If shapes disagree.
    """
    logits = np.asarray(logits, dtype=float).reshape(-1, len(MODEL_ORDER))
    if len(logits) != len(segments):
        raise ValueError(f"{len(logits)} logit rows for {len(segments)} segments")
    probs = softmax(logits)[:, [MODEL_ORDER.index(n) for n in NAMES]]
    return SegmentScores(segments, NAMES, probs)


def labels(scores: SegmentScores) -> list[str]:
    """Most probable class per segment.

    Args:
        scores: Output of :func:`postprocess`.

    Returns:
        One of :data:`NAMES` per segment.
    """
    return [scores.names[i] for i in np.argmax(scores.values, axis=1)] if len(scores.values) else []


class AddresseeWorker:
    """Scores adult segments of one phase."""

    def load(self, options: dict[str, Any]) -> dict[str, Any]:
        """Download the checkpoint and config at the pinned revision and build the model."""
        import lightning as pl
        from addressee.models.modeling_hubert import HubertFinetune
        from addressee.utils.config import load_config
        from huggingface_hub import hf_hub_download

        revision = options.get("weights_revision") or None
        self.model_path = hf_hub_download("coml/addressee", "best.ckpt", revision=revision)
        config_path = hf_hub_download("coml/addressee", "config.yaml", revision=revision)
        self.config = load_config(train_config=config_path)
        self.config.model_id = self.model_path
        self.config.train.batch_size = int(options.get("batch", "16"))
        self.model = HubertFinetune(self.config)
        self.model.eval()
        self.device = torch_device(options.get("device", "cpu"))
        accelerator = {"cuda": "gpu"}.get(self.device, self.device)
        self.trainer = pl.Trainer(accelerator=accelerator, devices=1, logger=False)
        return {"device": self.device, "revision": revision}

    def forward(self, request: InferenceRequest, segments: tuple[Segment, ...]) -> np.ndarray:
        """Logits of the adult segments of one request."""
        import pandas as pd
        from addressee.dataloaders import AddresseeDataloader

        samples, rate = read_wav(request.wav)
        rows = [
            {
                "uid": "audio",
                "label": s.label,
                "start_time_s": s.start_s,
                "duration_s": s.end_s - s.start_s,
                "duration(s)": s.end_s - s.start_s,
                "segment_onset": s.start_s * 1000,
                "segment_offset": s.end_s * 1000,
                "recording_duration": len(samples) / rate * 1000,
                "file_path": str(request.wav),
                "addressee": pd.NA,
                "binary_classes": "KCDS",
            }
            for s in segments
        ]
        data = AddresseeDataloader(
            dataset="addressee",
            dataset_path=self.config.data.dataset_path,
            inference=True,
            optional_df=pd.DataFrame(rows),
            config=self.config,
            n_workers=0,
        )
        predictions = self.trainer.predict(self.model, datamodule=data, ckpt_path=self.model_path)
        return np.concatenate([np.asarray(p.detach().cpu()) for p in predictions or []])

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Addressee probabilities of the adult segments of one phase.

        Args:
            request: Request whose segments carry VTC labels; non-adult ones are ignored.

        Returns:
            ``addressee`` segment scores.
        """
        adult = tuple(s for s in request.segments or () if s.label in ADULT_LABELS)
        adult = split_long_segments(adult)
        logits = self.forward(request, adult) if adult else np.zeros((0, len(MODEL_ORDER)))
        return InferenceResult(request.key, segments={"addressee": postprocess(adult, logits)})


if __name__ == "__main__":
    sys.exit(serve(AddresseeWorker()))
