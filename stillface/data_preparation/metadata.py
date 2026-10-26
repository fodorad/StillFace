"""Session metadata spreadsheet access."""

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stillface.constants import PHASES

logger = logging.getLogger(__name__)

HRV_COLUMNS: tuple[str, ...] = ("polar_mother", "polar_baby")
"""Metadata columns flagging the availability of the heart rate recordings."""

MISSING_CAMERA_COLUMNS: tuple[str, ...] = ("mother", "baby", "window", "door")
"""Metadata columns flagging the availability of the camera recordings."""


@dataclass(frozen=True)
class PhaseWindows:
    """Start and end time of each experiment phase.

    Attributes:
        windows: Phase name mapped to ``(start, end)`` strings in ``MM:SS`` format.
    """

    windows: dict[str, tuple[str, str]]

    @classmethod
    def from_row(cls, row: pd.Series) -> "PhaseWindows":
        """Read the phase windows of one metadata row.

        Args:
            row: Metadata row with ``<phase>_start_(MM:SS)``/``<phase>_end_(MM:SS)`` columns.

        Returns:
            The phase windows, with missing cells stored as ``"nan"``.
        """
        return cls(
            {
                phase: (str(row[f"{phase}_start_(MM:SS)"]), str(row[f"{phase}_end_(MM:SS)"]))
                for phase in PHASES
            }
        )


def load_metadata(path: Path) -> pd.DataFrame:
    """Load the session metadata table.

    Args:
        path: Spreadsheet (``.xlsx``) or ``.csv`` file.

    Returns:
        One row per session.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file type is not supported.
    """
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    if path.suffix == ".xlsx":
        return pd.read_excel(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported metadata file format: {path.suffix}")


def missing_recordings(metadata: pd.DataFrame) -> pd.DataFrame:
    """List the camera and heart rate recordings flagged as missing.

    Args:
        metadata: Table returned by :func:`load_metadata`.

    Returns:
        One row per missing recording with ``ID``, session date, hour and a description.
    """
    checks = [(col, "camera") for col in MISSING_CAMERA_COLUMNS]
    checks += [(col, "HRV") for col in HRV_COLUMNS]
    entries = [
        {
            "ID": row["ID"],
            "session_date": row["Session_date_(YYYY-HH-DD)"],
            "session_hour": row["Session_hour_(HH:MM)"],
            "missing": f"{kind} {{{column}}}",
        }
        for _, row in metadata.iterrows()
        for column, kind in checks
        if str(row[column]).strip().lower() == "n"
    ]
    return pd.DataFrame(entries, columns=["ID", "session_date", "session_hour", "missing"])


def sessions_to_sync(metadata: pd.DataFrame) -> list[str]:
    """Select the sessions that are automatically processable and not yet synchronized.

    Args:
        metadata: Table returned by :func:`load_metadata`.

    Returns:
        Session IDs, in table order.
    """
    selected = metadata[(metadata["Auto"] != "n") & metadata["offset_mother-baby_(ms)"].isna()]
    return [str(session_id) for session_id in selected["ID"]]


def sessions_to_cut(metadata: pd.DataFrame) -> list[str]:
    """Select the sessions that are automatically processable.

    Args:
        metadata: Table returned by :func:`load_metadata`.

    Returns:
        Session IDs, in table order.
    """
    return [str(session_id) for session_id in metadata.loc[metadata["Auto"] != "n", "ID"]]


def phase_windows(metadata: pd.DataFrame, session_id: str) -> PhaseWindows:
    """Look up the phase windows of a session.

    Args:
        metadata: Table returned by :func:`load_metadata`.
        session_id: Session identifier.

    Returns:
        The session's phase windows.

    Raises:
        KeyError: If the session is not in the table.
    """
    rows = metadata[metadata["ID"].astype(str) == session_id]
    if rows.empty:
        raise KeyError(f"Session not found in metadata: {session_id}")
    return PhaseWindows.from_row(rows.iloc[0])
