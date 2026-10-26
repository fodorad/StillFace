"""Experiment-level constants shared across the package."""

CAMERAS: tuple[str, ...] = ("mother", "baby", "window", "door")
"""Camera names, in the order used for the 2x2 session grid."""

PHASES: tuple[str, ...] = ("baseline", "play", "stillface", "reunion")
"""Experiment phases in chronological order."""

TARGET_FPS: int = 60
"""Frame rate all camera sources are normalized to before synchronization."""
