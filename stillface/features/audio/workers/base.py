"""Worker process entry point shared by every model worker.

A worker is started as ``python -m <worker module> <request.json>``. The request file
names the output folder, the model options and the requests. The model is loaded once;
each request then writes ``<i>.npz`` (see :func:`~stillface.features.audio.models.save_result`)
or ``<i>.error.txt``, so one bad input never loses the others. A failure while loading the
model exits non-zero and the adapter turns it into an error for every request.
"""

import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from stillface.features.audio.models import (
    InferenceRequest,
    InferenceResult,
    request_from_dict,
    save_result,
)
from stillface.features.io import write_atomic

logger = logging.getLogger(__name__)

ERROR_SUFFIX: str = ".error.txt"
"""Suffix of the per-request error file written instead of a result."""


class Worker(Protocol):
    """A model inside its worker process."""

    def load(self, options: dict[str, Any]) -> dict[str, Any]:
        """Load the model.

        Args:
            options: Model options from the config (device, window, thresholds ...).

        Returns:
            Provenance added to every result's ``meta`` (e.g. the device actually used).
        """
        ...

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Run the model on one request.

        Args:
            request: Input.

        Returns:
            The native result.
        """
        ...


def result_path(out_dir: Path, index: int) -> Path:
    """Result file of the request at ``index``."""
    return out_dir / f"{index}.npz"


def error_path(out_dir: Path, index: int) -> Path:
    """Error file of the request at ``index``."""
    return out_dir / f"{index}{ERROR_SUFFIX}"


def serve(worker: Worker, argv: Sequence[str] | None = None) -> int:
    """Process a request file with a worker.

    Args:
        worker: Model worker.
        argv: ``[request.json]``; ``sys.argv[1:]`` if None.

    Returns:
        Exit code 0 once every request has a result or error file.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    data = json.loads(Path(args[0]).read_text())
    out_dir = Path(data["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = worker.load(data.get("options", {}))
    for index, item in enumerate(data["requests"]):
        request = request_from_dict(item)
        try:
            result = worker.infer(request)
            result.meta = {**meta, **result.meta}
            write_atomic(result_path(out_dir, index), lambda tmp, r=result: save_result(r, tmp))
        except Exception as error:
            logger.exception("Request %s failed", request.key)
            message = f"{type(error).__name__}: {error}"
            write_atomic(error_path(out_dir, index), lambda tmp, m=message: tmp.write_text(m))
    return 0


def torch_device(preferred: str) -> str:
    """Pick a torch device, falling back to CPU when the preferred one is unavailable.

    Args:
        preferred: ``mps``, ``cuda`` or ``cpu``.

    Returns:
        The device name to use.
    """
    import torch

    if preferred == "mps" and torch.backends.mps.is_available():
        return "mps"
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def window_starts(duration_s: float, window_s: float, hop_s: float) -> list[float]:
    """Start times of analysis windows covering a signal.

    Windows advance by ``hop_s``; the last start is the last one whose window still
    starts inside the signal and ends at most at ``duration_s``. A signal shorter than one
    window gets a single window at 0 (the model pads it).

    Args:
        duration_s: Signal duration.
        window_s: Window length.
        hop_s: Hop between window starts.

    Returns:
        Start times in seconds.
    """
    if window_s <= 0 or hop_s <= 0:
        raise ValueError("window and hop must be positive")
    last = duration_s - window_s
    if last <= 0:
        return [0.0]
    n = int(last // hop_s + 1e-9) + 1
    return [round(i * hop_s, 6) for i in range(n)]


def fixed_windows(
    samples: np.ndarray, rate: int, starts: list[float], window_s: float
) -> np.ndarray:
    """Cut equal-length windows, zero-padding the ones that run past the end.

    Args:
        samples: Mono samples.
        rate: Sample rate.
        starts: Window start times.
        window_s: Window length.

    Returns:
        Array of shape ``(len(starts), round(window_s * rate))``.
    """
    size = round(window_s * rate)
    out = np.zeros((len(starts), size), dtype=np.float32)
    for i, start in enumerate(starts):
        part = samples[round(start * rate) :][:size]
        out[i, : len(part)] = part
    return out
