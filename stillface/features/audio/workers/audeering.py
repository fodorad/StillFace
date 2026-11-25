"""audEERING worker: arousal, dominance and valence per window (environment ``hf``).

wav2vec 2.0 large-robust (12 layers) fine-tuned on MSP-Podcast for dimensional emotion
(Wagner et al. 2023). Input: the raw 16 kHz waveform of one window; output: arousal,
dominance and valence, each roughly in ``[0, 1]``, plus the pooled 1024-d hidden state. It is
already trained; we only run it. Trained on adult podcast speech: acceptable for the mother,
experimental for the baby.
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

MODEL_ID: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
"""Hugging Face model id."""

DIMENSIONS: tuple[str, ...] = ("arousal", "dominance", "valence")
"""Output order of the regression head."""


def postprocess(
    windows: tuple[Segment, ...], values: np.ndarray, embeddings: np.ndarray
) -> SegmentScores:
    """Window scores with values clipped to ``[0, 1]``.

    Args:
        windows: Scored windows.
        values: Regression outputs, shape ``(n, 3)`` in :data:`DIMENSIONS` order.
        embeddings: Pooled hidden states, shape ``(n, d)``.

    Returns:
        The window scores.

    Raises:
        ValueError: If shapes disagree.
    """
    values = np.asarray(values, dtype=float).reshape(-1, len(DIMENSIONS))
    if len(values) != len(windows) or len(embeddings) != len(windows):
        raise ValueError(f"{len(values)} outputs for {len(windows)} windows")
    return SegmentScores(
        windows, DIMENSIONS, np.clip(values, 0.0, 1.0), np.asarray(embeddings, dtype=float)
    )


class AudeeringWorker:
    """Scores windows inside the requested segments."""

    def load(self, options: dict[str, Any]) -> dict[str, Any]:
        """Build the model card's EmotionModel and load the pinned weights."""
        import torch
        from torch import nn
        from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel, Wav2Vec2Processor

        class RegressionHead(nn.Module):
            def __init__(self, config: Any) -> None:
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

            def forward(self, x: Any) -> Any:  # ty: ignore[invalid-mutable-override]
                x = self.dropout(x)
                x = torch.tanh(self.dense(x))
                return self.out_proj(self.dropout(x))

        class EmotionModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config: Any) -> None:
                super().__init__(config)
                self.wav2vec2 = Wav2Vec2Model(config)
                self.classifier = RegressionHead(config)
                self.post_init()

            def forward(self, input_values: Any) -> Any:  # ty: ignore[invalid-mutable-override]
                hidden = self.wav2vec2(input_values)[0].mean(dim=1)
                return hidden, self.classifier(hidden)

        revision: str = options.get("weights_revision") or "main"
        self.window_s = float(options.get("window_s", "3.0")) or 3.0
        self.hop_s = float(options.get("hop_s", "0.5")) or 0.5
        self.device = torch_device(options.get("device", "cpu"))
        self.processor: Any = Wav2Vec2Processor.from_pretrained(MODEL_ID, revision=revision)
        self.model: Any = EmotionModel.from_pretrained(MODEL_ID, revision=revision)
        self.model.to(self.device).eval()
        return {"device": self.device, "model_id": MODEL_ID, "revision": revision}

    def forward(self, clips: list[np.ndarray], rate: int) -> tuple[np.ndarray, np.ndarray]:
        """Regression outputs and pooled states for each clip."""
        import torch

        values, states = [], []
        for clip in clips:
            inputs = self.processor(clip, sampling_rate=rate)["input_values"][0]
            tensor = torch.from_numpy(np.asarray(inputs, dtype=np.float32)[None, :]).to(self.device)
            with torch.no_grad():
                hidden, logits = self.model(tensor)
            values.append(logits.cpu().numpy()[0])
            states.append(hidden.cpu().numpy()[0])
        dim = states[0].shape[0] if states else 1024
        return np.array(values).reshape(-1, 3), np.array(states).reshape(-1, dim)

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Affect dimensions of every window of the requested segments.

        Args:
            request: Segment request.

        Returns:
            ``windows`` segment scores.
        """
        samples, rate = read_wav(request.wav)
        windows = segment_windows(request.segments or (), self.window_s, self.hop_s)
        clips = [samples[round(w.start_s * rate) : round(w.end_s * rate)] for w in windows]
        values, states = self.forward(clips, rate)
        return InferenceResult(
            request.key, segments={"windows": postprocess(windows, values, states)}
        )


if __name__ == "__main__":
    sys.exit(serve(AudeeringWorker()))
