"""Dataset locations.

The ``data/`` folder of a checkout holds soft links to the unmodified recordings
(``DB``) and to the derived products (``DB_PROCESSED``). Raw data is only read,
every generated file is written below the processed root.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from stillface.constants import CAMERAS, PHASES

logger = logging.getLogger(__name__)

DATASET_NAME: str = "ELTE-PPK_StillFace"
"""Directory name of the dataset below the raw and processed roots."""

RAW_ENV_VAR: str = "STILLFACE_RAW_DIR"
"""Environment variable overriding the raw dataset directory."""

PROCESSED_ENV_VAR: str = "STILLFACE_PROCESSED_DIR"
"""Environment variable overriding the processed dataset directory."""


@dataclass(frozen=True)
class DatasetPaths:
    """Resolves every dataset location from a raw and a processed root.

    Attributes:
        raw_dir: Directory of the original, untouched recordings and metadata.
        processed_dir: Directory receiving all derived data.
    """

    raw_dir: Path
    processed_dir: Path

    @classmethod
    def default(cls, base: Path = Path("data")) -> "DatasetPaths":
        """Build paths from the environment, falling back to ``base``/DB[_PROCESSED].

        Args:
            base: Folder holding the ``DB`` and ``DB_PROCESSED`` links.

        Returns:
            Resolved dataset paths.
        """
        raw = os.environ.get(RAW_ENV_VAR, str(base / "DB" / DATASET_NAME))
        processed = os.environ.get(PROCESSED_ENV_VAR, str(base / "DB_PROCESSED" / DATASET_NAME))
        return cls(raw_dir=Path(raw), processed_dir=Path(processed))

    @property
    def metadata_path(self) -> Path:
        """Session metadata spreadsheet."""
        return self.raw_dir / "metadata_database.xlsx"

    def original_dir(self, session_id: str) -> Path:
        """Original recordings of a session (read-only)."""
        return self.raw_dir / "Sessions" / session_id / "original"

    def phase_clip(self, session_id: str, camera: str, phase: str) -> Path:
        """Path of a camera's clip of one phase in the original dataset (read-only).

        These per-camera, per-phase clips are the input of every feature extractor.

        Args:
            session_id: Session identifier.
            camera: One of :data:`stillface.constants.CAMERAS`.
            phase: One of :data:`stillface.constants.PHASES`.

        Returns:
            Path of ``<camera>_<phase>.mp4``; the file may not exist.

        Raises:
            ValueError: If ``camera`` or ``phase`` is unknown.
        """
        if camera not in CAMERAS:
            raise ValueError(f"Unknown camera {camera!r}, expected one of {CAMERAS}")
        if phase not in PHASES:
            raise ValueError(f"Unknown phase {phase!r}, expected one of {PHASES}")
        return self.raw_dir / "Sessions" / session_id / "processed" / f"{camera}_{phase}.mp4"

    def available_phase_clips(self, session_id: str) -> dict[tuple[str, str], Path]:
        """Find the existing phase clips of a session.

        Args:
            session_id: Session identifier.

        Returns:
            ``(camera, phase)`` mapped to the clip path. Missing clips are logged and omitted.
        """
        clips: dict[tuple[str, str], Path] = {}
        for camera in CAMERAS:
            for phase in PHASES:
                clip = self.phase_clip(session_id, camera, phase)
                if clip.exists():
                    clips[(camera, phase)] = clip
                else:
                    logger.info("Phase clip not found: %s", clip)
        return clips

    def _processed_session(self, session_id: str) -> Path:
        return self.processed_dir / "Sessions" / session_id

    def normalized_dir(self, session_id: str) -> Path:
        """Original sources converted to a common container and frame rate."""
        return self._processed_session(session_id) / "normalized"

    def synced_dir(self, session_id: str) -> Path:
        """Time-aligned camera videos."""
        return self._processed_session(session_id) / "synced"

    def cut_dir(self, session_id: str) -> Path:
        """Per-phase clips of the synced videos."""
        return self._processed_session(session_id) / "cut"

    def visualize_dir(self, session_id: str) -> Path:
        """Stacked overview videos."""
        return self._processed_session(session_id) / "visualize"

    def thumbnail_path(self, session_id: str) -> Path:
        """Overview image of a session."""
        return self._processed_session(session_id) / "thumbnail.png"

    def features_dir(self, session_id: str, feature: str) -> Path:
        """Folder receiving the output of one feature extractor."""
        return self._processed_session(session_id) / "features" / feature

    def gaze_dir(self, session_id: str) -> Path:
        """Gaze timelines and run record of a session."""
        return self.features_dir(session_id, "gaze")

    def gaze_overlay_dir(self, session_id: str) -> Path:
        """Gaze overlay clips of a session."""
        return self.gaze_dir(session_id) / "overlay"

    @property
    def thumbnails_dir(self) -> Path:
        """Flat folder with one thumbnail per session."""
        return self.processed_dir / "Thumbnails"

    @property
    def logs_dir(self) -> Path:
        """Per-run bookkeeping files (synced, failed sessions)."""
        return self.processed_dir / "logs"
