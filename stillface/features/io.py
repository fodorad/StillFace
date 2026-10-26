"""Crash-safe output writing shared by the feature extractors."""

import logging
import os
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


def write_atomic(path: Path, writer: Callable[[Path], object]) -> None:
    """Produce a file so that it never exists half-written under its final name.

    ``writer`` fills a temporary file in the same folder, which is renamed to ``path`` once
    the writer returned. The temporary name keeps the extension, so ffmpeg can infer the
    container from it.

    Args:
        path: Final output path, parent folders are created.
        writer: Callable receiving the temporary path to write to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.partial{path.suffix}")
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def is_finished(path: Path, overwrite: bool = False) -> bool:
    """Tell whether an output can be skipped.

    Args:
        path: Final output path; only complete outputs exist under it (see
            :func:`write_atomic`).
        overwrite: Recompute even if the output exists.

    Returns:
        True if the output exists and no recomputation was requested.
    """
    return path.exists() and not overwrite
