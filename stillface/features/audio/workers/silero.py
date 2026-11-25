"""Silero VAD worker: speech probability per 32 ms chunk (environment ``vad``).

Silero is trained on speech; how it treats infant cries is untested, which is why it is
only a second opinion next to the VTC voice-type union.
"""

import sys
from typing import Any

import numpy as np

from stillface.features.audio.media import read_wav
from stillface.features.audio.models import FrameScores, InferenceRequest, InferenceResult
from stillface.features.audio.workers.base import serve

CHUNK: int = 512
"""Samples per Silero chunk at 16 kHz (32 ms)."""


def postprocess(probs: np.ndarray, rate: int, chunk: int = CHUNK) -> FrameScores:
    """Turn chunk probabilities into frame scores at the chunk centres.

    Args:
        probs: Speech probability per chunk, shape ``(n,)``.
        rate: Sample rate.
        chunk: Samples per chunk.

    Returns:
        One ``speech`` score per chunk, centred at ``(i + 0.5) * chunk / rate``.
    """
    probs = np.clip(np.asarray(probs, dtype=float).reshape(-1), 0.0, 1.0)
    times = (np.arange(len(probs)) + 0.5) * chunk / rate
    return FrameScores(times, ("speech",), probs)


class SileroWorker:
    """Runs Silero VAD on whole phases."""

    def load(self, options: dict[str, Any]) -> dict[str, Any]:
        """Load the packaged Silero model (CPU, its weights ship with the package)."""
        from importlib.metadata import version

        from silero_vad import load_silero_vad

        self.model = load_silero_vad()
        return {"device": "cpu", "silero_vad": version("silero-vad")}

    def forward(self, samples: np.ndarray, rate: int) -> np.ndarray:
        """Chunk probabilities of a mono signal."""
        import torch

        self.model.reset_states()
        with torch.no_grad():
            out = self.model.audio_forward(torch.from_numpy(samples.astype(np.float32)), rate)
        return out.reshape(-1).cpu().numpy()

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Speech probabilities of one phase.

        Args:
            request: Whole-phase request.

        Returns:
            The ``speech`` frame scores.
        """
        samples, rate = read_wav(request.wav)
        return InferenceResult(
            request.key, {"speech": postprocess(self.forward(samples, rate), rate)}
        )


if __name__ == "__main__":
    sys.exit(serve(SileroWorker()))
