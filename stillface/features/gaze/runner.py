"""Session runner shared by the head pilot and the gaze pipeline.

Walks over the phase clips of a session, writes one timeline CSV and (optionally) one
overlay clip per clip and one stacked overlay grid per phase, skips finished outputs,
isolates failures and records the run.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from stillface.config import DatasetPaths
from stillface.constants import CAMERAS, PHASES
from stillface.features.gaze.config import GazeConfig
from stillface.features.gaze.roles import RoleOverrides, load_role_overrides
from stillface.features.io import is_finished
from stillface.features.overlay import OverlayItem, render_overlay, render_overlay_grid
from stillface.features.run_record import (
    RunRecord,
    current_git_sha,
    installed_version,
    write_run_record,
)
from stillface.features.timeline import TimelineRow, write_timeline

logger = logging.getLogger(__name__)

ROLES_FEATURE: str = "gaze"
"""Feature folder holding the per-session ``roles.json`` used by every gaze-related run."""


@dataclass
class ClipOutput:
    """Result of processing one clip.

    Attributes:
        rows: Timeline rows.
        overlay_items: Sampled frame index mapped to the annotations to draw.
        flags: Issues of the clip, e.g. ``roles_ambiguous``.
    """

    rows: list[TimelineRow]
    overlay_items: dict[int, list[OverlayItem]]
    flags: tuple[str, ...] = field(default=())


ClipProcessor = Callable[[Path, str, str, RoleOverrides], ClipOutput]
"""Processes ``(clip, camera, phase, role overrides)`` into a :class:`ClipOutput`."""


def run_session(
    paths: DatasetPaths,
    session_id: str,
    out_dir: Path,
    processor: ClipProcessor,
    model_variant: str,
    config: GazeConfig,
    cameras: Sequence[str] = CAMERAS,
    phases: Sequence[str] = PHASES,
    overlay: bool = True,
    overwrite: bool = False,
) -> RunRecord:
    """Process the phase clips of a session and write the outputs.

    Existing outputs are kept unless ``overwrite``. A failing clip is logged, recorded under
    the ``failed`` flag of the run record and does not stop the others.

    Args:
        paths: Dataset locations.
        session_id: Session identifier.
        out_dir: Output folder; CSVs go here, overlays to ``overlay/``, the run record to
            ``run.json``.
        processor: Turns one clip into rows and overlay annotations.
        model_variant: Model recorded in the run record.
        config: Configuration, copied into the run record.
        cameras: Cameras to process.
        phases: Phases to process.
        overlay: Also render overlay clips and grids.
        overwrite: Recompute existing outputs.

    Returns:
        The run record, also written to ``out_dir/run.json``.
    """
    overrides = load_role_overrides(paths.features_dir(session_id, ROLES_FEATURE) / "roles.json")
    record = RunRecord(
        session_id=session_id,
        model_variant=model_variant,
        config=asdict(config),
        exordium_version=installed_version("exordium"),
        stillface_version=installed_version("stillface"),
        git_sha=current_git_sha(Path(__file__).parent),
    )

    for (camera, phase), clip in paths.available_phase_clips(session_id).items():
        if camera not in cameras or phase not in phases:
            continue
        name = f"{camera}_{phase}"
        csv_path = out_dir / f"{name}.csv"
        overlay_path = out_dir / "overlay" / f"{name}.mp4"
        if is_finished(csv_path, overwrite) and (
            not overlay or is_finished(overlay_path, overwrite)
        ):
            logger.info("Skipping finished %s", name)
            record.clips.append(clip.name)
            continue
        try:
            output = processor(clip, camera, phase, overrides)
            if not is_finished(csv_path, overwrite):
                write_timeline(output.rows, csv_path)
            if overlay:
                render_overlay(clip, output.overlay_items, overlay_path, overwrite)
        except Exception as error:
            logger.exception("Processing failed for %s", name)
            record.flags.setdefault("failed", []).append(f"{name}: {error}")
            continue
        record.clips.append(clip.name)
        for flag in output.flags:
            record.flags.setdefault(flag, []).append(name)

    if overlay:
        for phase in phases:
            overlays = {
                camera: out_dir / "overlay" / f"{camera}_{phase}.mp4"
                for camera in cameras
                if (out_dir / "overlay" / f"{camera}_{phase}.mp4").exists()
            }
            render_overlay_grid(overlays, out_dir / "overlay" / f"session_{phase}.mp4", overwrite)

    write_run_record(record, out_dir / "run.json")
    return record
