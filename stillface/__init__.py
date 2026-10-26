"""StillFace: tools for the analysis of mother-baby Still Face experiment recordings."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("stillface")
except PackageNotFoundError:  # pragma: no cover - source tree without installed metadata
    __version__ = "0.0.0"
"""Installed package version."""
