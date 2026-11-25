"""LittleBeats worker: speaker and vocalisation types per window (environment ``littlebeats``).

wav2vec 2.0 pre-trained on 4300 h of LittleBeats and LENA home recordings and fine-tuned
with three heads (Li et al. 2023, SpeechBrain recipe ``wav2vec_LittleBeats``) on 2 s
windows:

- speaker: ``SIL`` silence, ``CHN`` target child, ``FAN`` female adult, ``MAN`` male adult,
  ``CXN`` other child;
- infant vocalisation: ``CRY``, ``FUS`` (fuss), ``BAB`` (babble);
- adult vocalisation: ``CDS`` (child-directed), ``ADS`` (adult-directed), ``LAU`` (laugh),
  ``SNG`` (singing / rhythmic).

The label orders follow the recipe's ``dict_map``. Each head's softmax is reported for every
window; the vocalisation heads are only meaningful where the speaker head (or VTC) says the
infant / an adult vocalises.
"""

import sys
from typing import Any

import numpy as np

from stillface.features.audio.media import read_wav
from stillface.features.audio.models import FrameScores, InferenceRequest, InferenceResult
from stillface.features.audio.workers.addressee import softmax
from stillface.features.audio.workers.base import fixed_windows, serve, torch_device, window_starts

REPO: str = "lijialudew/wav2vec_LittleBeats_LENA"
"""Hugging Face repository of the checkpoints."""

FINE_TUNED: str = "LL_4300_fine_tuned/save/CKPT+2022-11-26+14-06-17+00"
"""Folder of the fine-tuned SpeechBrain checkpoint."""

SPEAKERS: tuple[str, ...] = ("SIL", "CHN", "FAN", "MAN", "CXN")
"""Speaker head classes in output order."""

INFANT: tuple[str, ...] = ("CRY", "FUS", "BAB")
"""Infant vocalisation head classes in output order."""

ADULT: tuple[str, ...] = ("CDS", "ADS", "LAU", "SNG")
"""Adult vocalisation head classes in output order (``ADS`` = the recipe's FAN/MAN)."""


def postprocess(
    starts: list[float],
    window_s: float,
    speaker: np.ndarray,
    infant: np.ndarray,
    adult: np.ndarray,
) -> dict[str, FrameScores]:
    """Softmax of each head at the window centres.

    Args:
        starts: Window start times.
        window_s: Window length.
        speaker: Speaker logits, shape ``(n, 5)``.
        infant: Infant vocalisation logits, shape ``(n, 3)``.
        adult: Adult vocalisation logits, shape ``(n, 4)``.

    Returns:
        ``speaker``, ``infant`` and ``adult`` frame scores.

    Raises:
        ValueError: If a head has the wrong shape.
    """
    times = np.asarray(starts, dtype=float) + window_s / 2
    out = {}
    for name, logits, labels in (
        ("speaker", speaker, SPEAKERS),
        ("infant", infant, INFANT),
        ("adult", adult, ADULT),
    ):
        logits = np.asarray(logits, dtype=float)
        if logits.shape != (len(starts), len(labels)):
            raise ValueError(
                f"{name} head: logits {logits.shape}, expected {(len(starts), len(labels))}"
            )
        out[name] = FrameScores(times, labels, softmax(logits))
    return out


class LittleBeatsWorker:
    """Runs the three-head model on overlapping 2 s windows."""

    def load(self, options: dict[str, Any]) -> dict[str, Any]:
        """Download the pinned checkpoints and rebuild the recipe's modules."""
        import tempfile

        import torch
        from huggingface_hub import hf_hub_download
        from speechbrain.lobes.models.CRDNN import DNN_Block
        from speechbrain.lobes.models.fairseq_wav2vec import FairseqWav2Vec2
        from speechbrain.nnet.linear import Linear
        from speechbrain.nnet.pooling import StatisticsPooling

        revision = options.get("weights_revision") or None
        self.window_s = float(options.get("window_s", "2.0")) or 2.0
        self.hop_s = float(options.get("hop_s", "1.0")) or 1.0
        self.device = torch_device(options.get("device", "cpu"))

        def fetch(name: str) -> str:
            return hf_hub_download(REPO, name, revision=revision)

        self.tmp = tempfile.TemporaryDirectory()
        self.wav2vec2 = FairseqWav2Vec2(
            pretrained_path=fetch("LL_4300/checkpoint_best.pt"),
            save_path=f"{self.tmp.name}/w2v.pt",
            output_norm=True,
            freeze=True,
        )
        self.wav2vec2.load_state_dict(
            torch.load(fetch(f"{FINE_TUNED}/wav2vec2.ckpt"), map_location="cpu")
        )
        self.pool = StatisticsPooling(return_std=False)
        dnns = [DNN_Block(input_shape=[None, 768], neurons=384, dropout=0.1) for _ in range(3)]
        heads = [Linear(input_size=384, n_neurons=n, bias=False) for n in (5, 3, 4)]
        self.heads = torch.nn.ModuleList([*dnns, *heads])
        self.heads.load_state_dict(
            torch.load(fetch(f"{FINE_TUNED}/model.ckpt"), map_location="cpu")
        )
        self.wav2vec2.to(self.device).eval()
        self.heads.to(self.device).eval()
        return {"device": self.device, "revision": revision}

    def forward(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Logits of the three heads for a batch of equal-length windows."""
        import torch

        with torch.no_grad():
            wavs = torch.from_numpy(windows.astype(np.float32)).to(self.device)
            lens = torch.ones(len(wavs), device=self.device)
            pooled = self.pool(self.wav2vec2(wavs), lens).view(len(wavs), -1)
            dnn_sp, dnn_chn, dnn_adu, out_sp, out_chn, out_adu = self.heads
            return (
                out_sp(dnn_sp(pooled)).cpu().numpy(),
                out_chn(dnn_chn(pooled)).cpu().numpy(),
                out_adu(dnn_adu(pooled)).cpu().numpy(),
            )

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Head probabilities over one phase.

        Args:
            request: Whole-phase request.

        Returns:
            ``speaker``, ``infant`` and ``adult`` frame scores.
        """
        samples, rate = read_wav(request.wav)
        starts = window_starts(len(samples) / rate, self.window_s, self.hop_s)
        windows = fixed_windows(samples, rate, starts, self.window_s)
        speaker, infant, adult = self.forward(windows)
        return InferenceResult(
            request.key, postprocess(starts, self.window_s, speaker, infant, adult)
        )


if __name__ == "__main__":
    sys.exit(serve(LittleBeatsWorker()))
