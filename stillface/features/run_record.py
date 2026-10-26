"""Run record: what produced a set of feature outputs."""

import json
import subprocess
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from stillface.features.io import write_atomic


@dataclass
class RunRecord:
    """Provenance of one extraction run over a session.

    Attributes:
        session_id: Session identifier.
        model_variant: Model that produced the signals (``none`` for the head pilot).
        config: Copy of the configuration used.
        exordium_version: Installed exordium version, None if not installed.
        stillface_version: Installed StillFace version.
        git_sha: Commit of the StillFace checkout, None outside a git repository.
        clips: Names of the processed clips.
        flags: Issue name mapped to the affected items, e.g. ``roles_ambiguous`` or ``failed``.
    """

    session_id: str
    model_variant: str
    config: dict
    exordium_version: str | None = None
    stillface_version: str | None = None
    git_sha: str | None = None
    clips: list[str] = field(default_factory=list)
    flags: dict[str, list[str]] = field(default_factory=dict)


def write_run_record(record: RunRecord, path: Path) -> None:
    """Write a run record as JSON, atomically.

    Args:
        record: Record to write.
        path: Output file.
    """
    write_atomic(path, lambda tmp: tmp.write_text(json.dumps(asdict(record), indent=2) + "\n"))


def read_run_record(path: Path) -> RunRecord:
    """Read a run record.

    Args:
        path: File written by :func:`write_run_record`.

    Returns:
        The record.
    """
    return RunRecord(**json.loads(path.read_text()))


def current_git_sha(cwd: Path) -> str | None:
    """Commit hash of the checkout containing ``cwd``.

    Args:
        cwd: Any folder inside the checkout.

    Returns:
        The 40-character hash, None if ``cwd`` is not in a git repository or git is missing.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def installed_version(package: str) -> str | None:
    """Installed version of a package.

    Args:
        package: Distribution name.

    Returns:
        The version string, None if the package is not installed.
    """
    try:
        return version(package)
    except PackageNotFoundError:
        return None
