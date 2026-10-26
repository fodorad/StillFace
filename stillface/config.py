"""Dataset locations.

The ``data/`` folder of a checkout holds soft links to the unmodified recordings
(``DB``) and to the derived products (``DB_PROCESSED``). Raw data is only read,
every generated file is written below the processed root.
"""

import os
from dataclasses import dataclass
from pathlib import Path

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

    @property
    def thumbnails_dir(self) -> Path:
        """Flat folder with one thumbnail per session."""
        return self.processed_dir / "Thumbnails"

    @property
    def logs_dir(self) -> Path:
        """Per-run bookkeeping files (synced, failed sessions)."""
        return self.processed_dir / "logs"
