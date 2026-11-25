"""AST AudioSet worker: 527 sound-event probabilities per window (environment ``hf``).

Audio Spectrogram Transformer fine-tuned on AudioSet (Gong et al. 2021): 128-bin log-mel
input, sigmoid output per class. Windows of ``window_s`` (default 2 s) advance by ``hop_s``
(default 0.25 s), as in the probe; each window's scores are placed at its centre.
"""

import sys
from typing import Any

import numpy as np

from stillface.features.audio.media import read_wav
from stillface.features.audio.models import FrameScores, InferenceRequest, InferenceResult
from stillface.features.audio.workers.base import serve, torch_device, window_starts

MODEL_ID: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
"""Hugging Face model id."""

N_CLASSES: int = 527
"""Number of AudioSet classes."""


def postprocess(
    starts: list[float], window_s: float, probs: np.ndarray, names: tuple[str, ...]
) -> FrameScores:
    """Turn per-window probabilities into frame scores at the window centres.

    Args:
        starts: Window start times.
        window_s: Window length.
        probs: Sigmoid outputs, shape ``(len(starts), len(names))``.
        names: Class names in index order.

    Returns:
        The frame scores.

    Raises:
        ValueError: If shapes disagree or probabilities leave ``[0, 1]``.
    """
    probs = np.asarray(probs, dtype=float)
    if probs.shape != (len(starts), len(names)):
        raise ValueError(f"probabilities {probs.shape} for {len(starts)} windows")
    if probs.size and (probs.min() < 0 or probs.max() > 1):
        raise ValueError("probabilities outside [0, 1]")
    times = np.asarray(starts, dtype=float) + window_s / 2
    return FrameScores(times, names, probs)


class AstWorker:
    """Runs AST on overlapping windows of whole phases."""

    def load(self, options: dict[str, Any]) -> dict[str, Any]:
        """Load the extractor and model at the configured revision."""
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        revision: str = options.get("weights_revision") or "main"
        model_id: str = options.get("model_id", MODEL_ID)
        self.window_s = float(options.get("window_s", "2.0")) or 2.0
        self.hop_s = float(options.get("hop_s", "0.25")) or 0.25
        self.batch = int(options.get("batch", "32"))
        self.device = torch_device(options.get("device", "cpu"))
        self.extractor: Any = ASTFeatureExtractor.from_pretrained(model_id, revision=revision)
        self.model: Any = ASTForAudioClassification.from_pretrained(model_id, revision=revision)
        self.model.to(self.device).eval()
        labels: Any = self.model.config.id2label
        self.names = tuple(labels[i] for i in range(len(labels)))
        return {"device": self.device, "model_id": model_id, "revision": revision}

    def forward(self, windows: list[np.ndarray], rate: int) -> np.ndarray:
        """Sigmoid outputs of a list of windows."""
        import torch

        out = []
        for i in range(0, len(windows), self.batch):
            inputs = self.extractor(
                windows[i : i + self.batch], sampling_rate=rate, return_tensors="pt"
            )
            with torch.no_grad():
                logits = self.model(**{k: v.to(self.device) for k, v in inputs.items()}).logits
            out.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.names)))

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Class probabilities of one phase.

        Args:
            request: Whole-phase request.

        Returns:
            The ``audioset`` frame scores.
        """
        samples, rate = read_wav(request.wav)
        starts = window_starts(len(samples) / rate, self.window_s, self.hop_s)
        size = round(self.window_s * rate)
        windows = [samples[round(s * rate) : round(s * rate) + size] for s in starts]
        scores = postprocess(starts, self.window_s, self.forward(windows, rate), self.names)
        return InferenceResult(request.key, {"audioset": scores})


if __name__ == "__main__":
    sys.exit(serve(AstWorker()))
