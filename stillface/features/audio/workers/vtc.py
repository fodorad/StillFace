"""VTC 2.2 worker: voice-type scores and segments (environment ``vtc``).

The Voice Type Classifier (Lavechin et al.; VTC 2 on BabyHuBERT, Charlot et al. 2025) gives
four independent sigmoid scores every 20 ms: ``KCHI`` key child, ``OCH`` other child,
``MAL`` adult male, ``FEM`` adult female. With no siblings present, KCHI and OCH are both the
baby. Scores are thresholded per class (``f1`` balanced set by default, ``hp`` high
precision) and merged into segments, written as RTTM. VTC classifies voice *type*: it cannot
tell the mother from a female experimenter; the roles stage does that with voice embeddings.
"""

import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from stillface.features.audio.models import (
    FrameScores,
    InferenceRequest,
    InferenceResult,
    Segment,
    SegmentScores,
)
from stillface.features.audio.workers.base import serve, torch_device

LABELS: tuple[str, ...] = ("KCHI", "OCH", "MAL", "FEM")
"""VTC classes in output order."""

FRAME_S: float = 0.02
"""VTC frame step (320 samples at 16 kHz)."""

URI: str = "audio"
"""File stem used for the temporary input of a request."""


def parse_rttm(text: str) -> list[Segment]:
    """Parse RTTM ``SPEAKER`` lines.

    Args:
        text: RTTM content.

    Returns:
        Segments with the speaker label as ``label``, sorted by start.
    """
    segments = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[0] == "SPEAKER":
            start, duration = float(fields[3]), float(fields[4])
            segments.append(Segment(start, start + duration, "", fields[7]))
    return sorted(segments, key=lambda s: (s.start_s, s.label))


def postprocess(
    probs: dict[str, np.ndarray], rttm_text: str, frame_s: float = FRAME_S
) -> tuple[FrameScores, SegmentScores]:
    """Frame scores and scored segments from VTC's outputs.

    Args:
        probs: Class name mapped to its per-frame probability.
        rttm_text: Merged RTTM of the thresholded segments.
        frame_s: Frame step.

    Returns:
        ``(frames, segments)``; each segment's ``mean_prob`` is the mean probability of its
        class over the frames it covers.

    Raises:
        ValueError: If a class is missing or the arrays differ in length.
    """
    missing = [label for label in LABELS if label not in probs]
    if missing:
        raise ValueError(f"VTC probabilities missing classes {missing}")
    columns = [np.asarray(probs[label], dtype=float).reshape(-1) for label in LABELS]
    if len({len(c) for c in columns}) != 1:
        raise ValueError("VTC class probabilities differ in length")
    values = np.clip(np.stack(columns, axis=1), 0.0, 1.0)
    times = (np.arange(len(values)) + 0.5) * frame_s
    segments = tuple(parse_rttm(rttm_text))
    means = []
    for segment in segments:
        inside = (times >= segment.start_s) & (times < segment.end_s)
        column = values[inside, LABELS.index(segment.label)] if segment.label in LABELS else []
        means.append(float(np.mean(column)) if len(column) else float("nan"))
    frames = FrameScores(times, LABELS, values)
    scored = SegmentScores(segments, ("mean_prob",), np.array(means).reshape(-1, 1))
    return frames, scored


def load_probs(path: Path) -> Any:
    """Load VTC's saved probabilities with the weights-only loader.

    VTC pickles its class names as ``tomlkit`` string objects. Exactly the ``tomlkit``
    types the file needs are allowed; any other non-tensor type is refused.

    Args:
        path: ``<uri>-probs_dict_t.pt`` written by VTC.

    Returns:
        Class name mapped to its probability tensor.

    Raises:
        ValueError: If the file needs a type outside ``tomlkit``.
    """
    import importlib

    import torch

    allowed = []
    for name in torch.serialization.get_unsafe_globals_in_checkpoint(str(path)):
        module, _, attr = name.rpartition(".")
        if not module.startswith("tomlkit"):
            raise ValueError(f"Unexpected type {name} in the VTC probabilities")
        allowed.append(getattr(importlib.import_module(module), attr))
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, weights_only=True)


class VtcWorker:
    """Runs VTC on whole phases."""

    def load(self, options: dict[str, Any]) -> dict[str, Any]:
        """Prepare the VTC run settings (the package loads the model per call)."""
        from importlib.metadata import version

        import vtc_inference

        self.run_vtc = vtc_inference.run_vtc
        self.vtc_version = options.get("vtc_version", "2.2")
        self.high_precision = options.get("thresholds", "f1") == "hp"
        self.device = torch_device(options.get("device", "cpu"))
        return {
            "device": self.device,
            "vtc_package": version("vtc"),
            "vtc_model": self.vtc_version,
            "thresholds": "hp" if self.high_precision else "f1",
        }

    def forward(self, wav: Path) -> tuple[dict[str, np.ndarray], str, dict[str, Any]]:
        """Run VTC on one WAV: probabilities, merged RTTM and the thresholds used."""
        with tempfile.TemporaryDirectory() as tmp:
            wavs, out = Path(tmp) / "wavs", Path(tmp) / "out"
            wavs.mkdir()
            (wavs / f"{URI}.wav").symlink_to(wav.resolve())
            self.run_vtc(
                output=str(out),
                wavs=str(wavs),
                config=self.vtc_version,
                checkpoint=self.vtc_version,
                device=self.device,
                save_probs=True,
                high_precision=self.high_precision,
            )
            saved = load_probs(out / "probs" / f"{URI}-probs_dict_t.pt")
            probs = {str(k): v.cpu().numpy() for k, v in saved.items()}
            rttm = out / "rttm" / f"{URI}.rttm"
            text = rttm.read_text() if rttm.exists() else ""
            name = "hp.toml" if self.high_precision else "f1.toml"
            thresholds = tomllib.loads((out / name).read_text()) if (out / name).exists() else {}
        return probs, text, thresholds

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Voice types of one phase.

        Args:
            request: Whole-phase request.

        Returns:
            ``voice_type`` frame scores, ``voice_type`` segments and the RTTM artefact.
        """
        probs, rttm, thresholds = self.forward(request.wav)
        frames, segments = postprocess(probs, rttm)
        return InferenceResult(
            request.key,
            {"voice_type": frames},
            {"voice_type": segments},
            {"rttm": rttm},
            {"applied_thresholds": thresholds},
        )


if __name__ == "__main__":
    sys.exit(serve(VtcWorker()))
