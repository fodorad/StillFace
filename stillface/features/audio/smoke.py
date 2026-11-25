"""Smoke test of the installed models on a synthetic clip (local only, not in CI).

``python -m stillface.features.audio.smoke`` writes a 10 s synthetic clip (tones and noise
bursts), runs every configured model whose environment exists, and reports per model
whether it returned well-formed output. It checks the plumbing (environments, workers,
weights), not model quality.
"""

import argparse
import logging
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from stillface.features.audio.config import DEFAULT_CONFIG_PATH, AudioConfig, load_audio_config
from stillface.features.audio.media import write_wav
from stillface.features.audio.models import InferenceError, InferenceRequest, Segment
from stillface.features.audio.registry import build_adapter, env_python, envs_root

logger = logging.getLogger(__name__)

SMOKE_SECONDS: float = 10.0
"""Length of the synthetic clip."""

SMOKE_SEGMENTS: tuple[Segment, ...] = (
    Segment(1.0, 4.0, "mother", "FEM"),
    Segment(5.0, 9.0, "baby", "KCHI"),
)
"""Segments given to segment models."""


def synthetic_clip(path: Path, seconds: float = SMOKE_SECONDS, rate: int = 16000) -> Path:
    """Write a clip with a 220 Hz tone, a 450 Hz tone and noise bursts.

    Args:
        path: Output WAV.
        seconds: Duration.
        rate: Sample rate.

    Returns:
        The path.
    """
    t = np.arange(round(seconds * rate)) / rate
    rng = np.random.default_rng(0)
    signal = np.where((t > 1) & (t < 4), 0.3 * np.sin(2 * np.pi * 220 * t), 0.0)
    signal += np.where((t > 5) & (t < 9), 0.3 * np.sin(2 * np.pi * 450 * t), 0.0)
    signal += np.where((t % 2) < 0.2, rng.normal(0, 0.05, t.size), 0.0)
    write_wav(path, signal, rate)
    return path


def smoke(
    config: AudioConfig, root: Path, wav: Path, models: Sequence[str] | None = None
) -> dict[str, str]:
    """Run the models on a clip.

    Args:
        config: Audio configuration.
        root: Folder of the model environments.
        wav: Input clip.
        models: Model names; every configured model if None.

    Returns:
        Model name mapped to ``ok: <kind>/<group>, ...``, ``disabled``, ``not installed`` or
        ``error: <message>``.
    """
    status = {}
    for name in models or sorted(config.models):
        model = config.models[name]
        if not model.enabled:
            status[name] = "disabled"
            continue
        if model.env and not env_python(root, model.env).exists():
            status[name] = "not installed"
            continue
        segments = SMOKE_SEGMENTS if model.input == "segments" else None
        request = InferenceRequest("smoke/play", wav, SMOKE_SECONDS, segments)
        try:
            result = build_adapter(name, config, root).run([request])[0]
        except Exception as error:
            status[name] = f"error: {error}"
            continue
        if isinstance(result, InferenceError):
            status[name] = f"error: {result.message}"
        elif not (result.frames or result.segments or result.artefacts):
            status[name] = "error: empty result"
        else:
            groups = [f"frames/{k}" for k in result.frames]
            groups += [f"segments/{k}" for k in result.segments]
            groups += [f"artefacts/{k}" for k in result.artefacts]
            status[name] = "ok: " + ", ".join(groups)
    return status


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point.

    Args:
        argv: Arguments; ``sys.argv[1:]`` if None.

    Returns:
        0 if no installed model failed.
    """
    parser = argparse.ArgumentParser(description="Run every installed audio model once.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--envs", type=Path, help="Model environments ($STILLFACE_AUDIO_ENVS)")
    parser.add_argument("--model", nargs="+", help="Models to test (default: all)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_audio_config(args.config)
    with tempfile.TemporaryDirectory() as tmp:
        wav = synthetic_clip(Path(tmp) / "smoke.wav")
        status = smoke(config, args.envs or envs_root(), wav, args.model)
    for name, text in status.items():
        logger.info("%-12s %s", name, text)
    return 1 if any(text.startswith("error") for text in status.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
