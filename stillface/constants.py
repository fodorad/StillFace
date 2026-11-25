"""Experiment-level constants shared across the package."""

CAMERAS: tuple[str, ...] = ("mother", "baby", "window", "door")
"""Camera names, in the order used for the 2x2 session grid."""

SPLIT_WINDOW_CAMERAS: tuple[str, ...] = ("window_1", "window_2")
"""Two window camcorders some early sessions used instead of one, both from about the
usual window viewpoint: ``window_1`` mostly framed the mother, ``window_2`` mostly the baby.
Only the audio source selection uses them; they are not part of the session grid."""

CLIP_CAMERAS: tuple[str, ...] = CAMERAS + SPLIT_WINDOW_CAMERAS
"""Every camera name a phase clip can carry."""

PHASES: tuple[str, ...] = ("baseline", "play", "stillface", "reunion")
"""Experiment phases in chronological order."""

TARGET_FPS: int = 60
"""Frame rate all camera sources are normalized to before synchronization."""
