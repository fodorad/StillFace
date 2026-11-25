"""Build the adapter of a configured model."""

import os
from collections.abc import Callable, Mapping
from pathlib import Path

from stillface.features.audio.config import AudioConfig, ModelConfig
from stillface.features.audio.models import ModelAdapter
from stillface.features.audio.subprocess_adapter import SubprocessAdapter

ENVS_VAR: str = "STILLFACE_AUDIO_ENVS"
"""Environment variable naming the folder that holds one virtual environment per model."""

InProcessFactory = Callable[[str, ModelConfig], ModelAdapter]
"""Builds an in-process adapter from the model name and its settings."""


def _prosody(name: str, model: ModelConfig) -> ModelAdapter:
    from stillface.features.audio.prosody import ProsodyAdapter

    return ProsodyAdapter(name, model)


def _egemaps(name: str, model: ModelConfig) -> ModelAdapter:
    from stillface.features.audio.prosody import EgemapsAdapter

    return EgemapsAdapter(name, model)


IN_PROCESS: dict[str, InProcessFactory] = {"prosody": _prosody, "egemaps": _egemaps}
"""In-process models (no ``env``) by name."""


def envs_root(environ: Mapping[str, str] = os.environ) -> Path:
    """Folder of the model environments.

    Args:
        environ: Environment variables.

    Returns:
        The path from :data:`ENVS_VAR`.

    Raises:
        RuntimeError: If the variable is not set.
    """
    value = environ.get(ENVS_VAR, "")
    if not value:
        raise RuntimeError(f"Set {ENVS_VAR} to the folder of the model environments")
    return Path(value)


def env_python(root: Path, env: str) -> Path:
    """Interpreter of a model environment (POSIX layout).

    Args:
        root: Folder of the model environments.
        env: Environment name.

    Returns:
        ``<root>/<env>/bin/python``.
    """
    return root / env / "bin" / "python"


def worker_options(model: ModelConfig) -> dict[str, str]:
    """Options passed to a worker's ``load``.

    Args:
        model: Model settings.

    Returns:
        Device, window and hop (as text) merged with the model-specific options.
    """
    return {
        "device": model.device,
        "window_s": repr(model.window_s),
        "hop_s": repr(model.hop_s),
        **model.options,
    }


def build_adapter(name: str, config: AudioConfig, root: Path | None = None) -> ModelAdapter:
    """Create the adapter of a configured model.

    Args:
        name: Model name, a key of ``config.models``.
        config: Audio configuration.
        root: Folder of the model environments; from :data:`ENVS_VAR` if None.

    Returns:
        A subprocess adapter, or the registered in-process adapter for models without ``env``.

    Raises:
        ValueError: If the model is unknown or has no runner.
    """
    if name not in config.models:
        raise ValueError(f"Unknown model {name!r}, configured: {sorted(config.models)}")
    model = config.models[name]
    if not model.env:
        if name not in IN_PROCESS:
            raise ValueError(f"Model {name!r} has no environment and no in-process adapter")
        return IN_PROCESS[name](name, model)
    python = env_python(root or envs_root(), model.env)
    return SubprocessAdapter(
        name,
        python,
        model.module,
        model.worker_version,
        model.weights_revision,
        worker_options(model),
        model.timeout_s,
    )
