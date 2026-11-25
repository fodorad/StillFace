"""Prosodic low-level descriptors, computed in-process.

Two in-process models implement the adapter protocol, so the stages treat them like any
other model:

- ``prosody`` (first pass, whole phase): Parselmouth (Praat) F0 in an adult range and in an
  infant range, intensity and harmonics-to-noise ratio, every 10 ms. The fusion stage picks
  the F0 range by the role of each segment.
- ``egemaps`` (second pass, per role segment): the 88 openSMILE eGeMAPS v02 functionals
  (Eyben et al. 2016), the standard compact descriptor set of computational paralinguistics.

These are numbers, not labels: they describe *how* someone vocalises (pitch height and
range, loudness, voice quality) and feed the IDS-strength index, cry acoustics, arousal and
prosodic synchrony. Both libraries come with the optional ``audio`` extra.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from stillface.features.audio.config import ModelConfig
from stillface.features.audio.media import read_wav
from stillface.features.audio.models import (
    FrameScores,
    InferenceError,
    InferenceRequest,
    InferenceResult,
    SegmentScores,
)
from stillface.features.run_record import installed_version

PROSODY_VERSION: str = "1"
"""Version of the prosody adapters' output layout."""

DEFAULTS: dict[str, float] = {
    "time_step": 0.01,
    "adult_f0_min": 100.0,
    "adult_f0_max": 700.0,
    "infant_f0_min": 150.0,
    "infant_f0_max": 1000.0,
}
"""Default prosody options; the config's ``models.prosody.options`` override them."""

HNR_UNVOICED: float = -199.0
"""Praat marks unvoiced HNR frames with -200 dB; values at or below this are NaN."""


def _option(options: dict[str, str], key: str) -> float:
    return float(options.get(key, DEFAULTS[key]))


def _revision(*packages: str) -> str:
    return ";".join(f"{p}-{installed_version(p) or 'missing'}" for p in packages)


@dataclass
class ProsodyAdapter:
    """Frame-level F0, intensity and HNR with Parselmouth.

    Attributes:
        name: Model name.
        model: Model settings; ``options`` may override :data:`DEFAULTS`.
        worker_version: Output layout version.
        weights_revision: Installed library version (no weights).
    """

    name: str
    model: ModelConfig = field(default_factory=ModelConfig)
    worker_version: str = PROSODY_VERSION
    weights_revision: str = field(default_factory=lambda: _revision("praat-parselmouth"))

    def run(self, requests: list[InferenceRequest]) -> list[InferenceResult | InferenceError]:
        """Measure every request; a failing request becomes an error.

        Args:
            requests: Whole-phase requests.

        Returns:
            One result or error per request.
        """
        out: list[InferenceResult | InferenceError] = []
        for request in requests:
            try:
                out.append(self._measure(request))
            except Exception as error:
                out.append(InferenceError(request.key, f"{type(error).__name__}: {error}"))
        return out

    def _measure(self, request: InferenceRequest) -> InferenceResult:
        import parselmouth

        options = self.model.options
        step = _option(options, "time_step")
        samples, rate = read_wav(request.wav)
        sound = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=rate)
        frames: dict[str, FrameScores] = {}
        for group in ("adult", "infant"):
            pitch = sound.to_pitch_ac(
                time_step=step,
                pitch_floor=_option(options, f"{group}_f0_min"),
                pitch_ceiling=_option(options, f"{group}_f0_max"),
            )
            f0 = pitch.selected_array["frequency"].astype(float)
            f0[f0 <= 0] = np.nan
            frames[f"f0_{group}"] = FrameScores(pitch.xs(), ("f0_hz",), f0)
        intensity = sound.to_intensity(time_step=step)
        frames["intensity"] = FrameScores(intensity.xs(), ("intensity_db",), intensity.values[0])
        hnr = sound.to_harmonicity_cc(time_step=step)
        values = hnr.values[0].astype(float)
        values[values <= HNR_UNVOICED] = np.nan
        frames["hnr"] = FrameScores(hnr.xs(), ("hnr_db",), values)
        return InferenceResult(request.key, frames, meta={"options": {**DEFAULTS, **options}})


@dataclass
class EgemapsAdapter:
    """eGeMAPS v02 functionals per segment with openSMILE.

    Attributes:
        name: Model name.
        model: Model settings.
        worker_version: Output layout version.
        weights_revision: Installed library version (no weights).
    """

    name: str
    model: ModelConfig = field(default_factory=ModelConfig)
    worker_version: str = PROSODY_VERSION
    weights_revision: str = field(default_factory=lambda: _revision("opensmile"))
    _smile: Any = None

    def run(self, requests: list[InferenceRequest]) -> list[InferenceResult | InferenceError]:
        """Compute the functionals of every segment of every request.

        Args:
            requests: Requests with segments.

        Returns:
            One result or error per request; a request without segments is an error.
        """
        out: list[InferenceResult | InferenceError] = []
        for request in requests:
            try:
                out.append(self._measure(request))
            except Exception as error:
                out.append(InferenceError(request.key, f"{type(error).__name__}: {error}"))
        return out

    def _measure(self, request: InferenceRequest) -> InferenceResult:
        import opensmile

        if request.segments is None:
            raise ValueError("eGeMAPS needs segments")
        if self._smile is None:
            self._smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
        names = tuple(self._smile.feature_names)
        samples, rate = read_wav(request.wav)
        rows = []
        for segment in request.segments:
            start, stop = round(segment.start_s * rate), round(segment.end_s * rate)
            frame = self._smile.process_signal(samples[start:stop], rate)
            rows.append(frame.to_numpy(dtype=float).reshape(-1))
        values = np.array(rows, dtype=float).reshape(len(rows), len(names))
        scores = SegmentScores(request.segments, names, values)
        return InferenceResult(request.key, segments={"egemaps": scores})
