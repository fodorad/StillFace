"""Provenance sidecars: what produced an output, so it can be traced and recomputed.

Every audio output ``<name>`` has a JSON sidecar ``<name>.json`` next to it. A sidecar
stores hashes of the inputs, the config digest and the code version; an output is up to
date only if all of them still match.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from stillface.features.io import write_atomic

SIDECAR_SUFFIX: str = ".json"
"""Suffix appended to an output's file name to name its sidecar."""


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file.

    Args:
        path: File to hash.
        chunk: Read size in bytes.

    Returns:
        Hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def sidecar_path(output: Path) -> Path:
    """Sidecar file of an output.

    Args:
        output: Output file.

    Returns:
        ``<output>.json``.
    """
    return output.with_name(output.name + SIDECAR_SUFFIX)


def write_sidecar(output: Path, record: dict[str, Any]) -> None:
    """Write the sidecar of an output atomically.

    Args:
        output: Output file the record describes.
        record: JSON-compatible provenance.
    """
    text = json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    write_atomic(sidecar_path(output), lambda tmp: tmp.write_text(text))


def read_sidecar(output: Path) -> dict[str, Any] | None:
    """Read the sidecar of an output.

    Args:
        output: Output file.

    Returns:
        The record, None if there is no readable sidecar.
    """
    path = sidecar_path(output)
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def is_up_to_date(output: Path, expected: dict[str, Any], overwrite: bool = False) -> bool:
    """Whether an output exists and its sidecar matches the expected provenance keys.

    Args:
        output: Output file.
        expected: Keys (e.g. input hash, config digest, code version) that must match.
        overwrite: Force recomputation.

    Returns:
        True if the output can be kept.
    """
    if overwrite or not output.exists():
        return False
    record = read_sidecar(output)
    return record is not None and all(record.get(k) == v for k, v in expected.items())
