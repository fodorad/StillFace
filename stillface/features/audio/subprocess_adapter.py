"""Run a model worker in its own environment as a child process."""

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from stillface.features.audio.models import (
    InferenceError,
    InferenceRequest,
    InferenceResult,
    load_result,
    request_to_dict,
)
from stillface.features.audio.workers.base import error_path, result_path

logger = logging.getLogger(__name__)

PACKAGE_ROOT: Path = Path(__file__).resolve().parents[3]
"""Folder containing the ``stillface`` package, put on the worker's ``PYTHONPATH``."""


@dataclass
class SubprocessAdapter:
    """Model adapter that runs a worker module with another interpreter.

    Attributes:
        name: Model name.
        python: Interpreter of the model environment.
        module: Worker module, run with ``-m``.
        worker_version: Version of the worker code.
        weights_revision: Exact weights revision; required.
        options: Model options passed to the worker's ``load``.
        timeout_s: Wall-clock limit for one batch.
        pythonpath: Extra import roots; the package root by default.
    """

    name: str
    python: Path
    module: str
    worker_version: str
    weights_revision: str
    options: dict[str, Any] = field(default_factory=dict)
    timeout_s: float = 3600.0
    pythonpath: tuple[Path, ...] = (PACKAGE_ROOT,)

    def __post_init__(self) -> None:
        """Refuse a run whose weights cannot be traced."""
        if not self.weights_revision:
            raise ValueError(f"{self.name}: a weights revision is required for provenance")

    def run(self, requests: list[InferenceRequest]) -> list[InferenceResult | InferenceError]:
        """Run the worker once for the whole batch.

        Args:
            requests: Inputs.

        Returns:
            One result or error per request; a timeout or crash turns every request without
            an output into an error.
        """
        if not requests:
            return []
        with tempfile.TemporaryDirectory(prefix=f"stillface-{self.name}-") as tmp:
            out_dir = Path(tmp) / "out"
            request_file = Path(tmp) / "request.json"
            request_file.write_text(
                json.dumps(
                    {
                        "out_dir": str(out_dir),
                        "options": self.options,
                        # Absolute paths: the worker starts in its own folder.
                        "requests": [
                            request_to_dict(replace(r, wav=Path(r.wav).resolve())) for r in requests
                        ],
                    }
                )
            )
            env = {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, self.pythonpath))}
            failure = ""
            try:
                done = subprocess.run(
                    [str(self.python), "-m", self.module, str(request_file)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    env=env,
                    # python -m puts the working directory first on sys.path: start in the
                    # private folder so only the PYTHONPATH code is importable.
                    cwd=tmp,
                )
                if done.returncode != 0:
                    tail = done.stderr.strip().splitlines()[-5:]
                    failure = f"worker exited with {done.returncode}: " + " | ".join(tail)
            except subprocess.TimeoutExpired:
                failure = f"worker timed out after {self.timeout_s} s"
            except OSError as error:
                failure = f"worker could not start: {error}"
            if failure:
                logger.error("%s: %s", self.name, failure)
            return [self._collect(out_dir, i, r, failure) for i, r in enumerate(requests)]

    def _collect(
        self, out_dir: Path, index: int, request: InferenceRequest, failure: str
    ) -> InferenceResult | InferenceError:
        result, error = result_path(out_dir, index), error_path(out_dir, index)
        if result.exists():
            return load_result(result)
        if error.exists():
            return InferenceError(request.key, error.read_text())
        return InferenceError(request.key, failure or "worker wrote no output")
