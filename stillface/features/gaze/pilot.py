"""Head pilot: heads, roles and overlays without any gaze model.

The pilot validates the detection, tracking and role assignment on chosen sessions before
the gaze model is run. It writes the same timeline format as the gaze pipeline, with empty
gaze fields, plus overlay clips showing the heads and their roles.
"""

import logging
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from stillface.config import DatasetPaths
from stillface.constants import CAMERAS, PHASES
from stillface.features.gaze.classes import NK
from stillface.features.gaze.config import GazeConfig
from stillface.features.gaze.heads import HeadTracker
from stillface.features.gaze.prepare import ClipHeads, prepare_clip
from stillface.features.gaze.roles import camera_persons, load_role_overrides, missing_reason
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

PILOT_FEATURE: str = "heads"
"""Name of the pilot's output folder below ``features/``."""


def head_rows(session_id: str, camera: str, phase: str, prepared: ClipHeads) -> list[TimelineRow]:
    """Timeline rows of the pilot: heads found or not, no gaze values.

    Args:
        session_id: Session identifier.
        camera: Camera name.
        phase: Experiment phase.
        prepared: Heads of the clip.

    Returns:
        One row per sampled frame for each person the camera can show; missing heads are
        ``NK`` rows with the reason.
    """
    rows = []
    for position, frame in enumerate(prepared.frame_indices):
        for person in camera_persons(camera):
            found = prepared.heads[person][position] is not None
            rows.append(
                TimelineRow(
                    session_id=session_id,
                    camera=camera,
                    phase=phase,
                    frame_index=frame,
                    time_s=frame / prepared.clip_fps,
                    person=person,
                    gaze_x=None,
                    gaze_y=None,
                    inout=None,
                    partner_mass=None,
                    gaze_class="" if found else NK,
                    reason="" if found else missing_reason(camera, person),
                )
            )
    return rows


def head_overlay_items(camera: str, prepared: ClipHeads) -> dict[int, list[OverlayItem]]:
    """Overlay annotations of the pilot, keyed by sampled frame.

    Args:
        camera: Camera name.
        prepared: Heads of the clip.

    Returns:
        Frame index mapped to one item per person.
    """
    items: dict[int, list[OverlayItem]] = {}
    for position, frame in enumerate(prepared.frame_indices):
        items[frame] = [
            OverlayItem(
                person,
                head,
                None,
                "" if head is not None else f"NK:{missing_reason(camera, person)}",
            )
            for person in camera_persons(camera)
            for head in [prepared.heads[person][position]]
        ]
    return items


def run_heads(
    paths: DatasetPaths,
    session_id: str,
    tracker: HeadTracker,
    config: GazeConfig,
    cameras: Sequence[str] = CAMERAS,
    phases: Sequence[str] = PHASES,
    overlay: bool = True,
    overwrite: bool = False,
) -> RunRecord:
    """Run the head pilot on the clips of one session.

    Per clip a timeline CSV and (optionally) an overlay clip are written; per phase a
    stacked overlay grid. Existing outputs are kept unless ``overwrite``. A failing clip is
    recorded in the run record and does not stop the others.

    Args:
        paths: Dataset locations.
        session_id: Session identifier.
        tracker: Head detector and tracker.
        config: Gaze configuration.
        cameras: Cameras to process.
        phases: Phases to process.
        overlay: Also render overlay clips and grids.
        overwrite: Recompute existing outputs.

    Returns:
        The run record, also written to ``run.json`` in the pilot folder.
    """
    out_dir = paths.features_dir(session_id, PILOT_FEATURE)
    overrides = load_role_overrides(out_dir / "roles.json")
    available = paths.available_phase_clips(session_id)
    record = RunRecord(
        session_id=session_id,
        model_variant="none",
        config=asdict(config),
        exordium_version=installed_version("exordium"),
        stillface_version=installed_version("stillface"),
        git_sha=current_git_sha(Path(__file__).parent),
    )

    for (camera, phase), clip in available.items():
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
            prepared = prepare_clip(clip, camera, phase, tracker, config, overrides=overrides)
            if not is_finished(csv_path, overwrite):
                write_timeline(head_rows(session_id, camera, phase, prepared), csv_path)
            if overlay:
                render_overlay(clip, head_overlay_items(camera, prepared), overlay_path, overwrite)
        except Exception as error:
            logger.exception("Head pilot failed for %s", name)
            record.flags.setdefault("failed", []).append(f"{name}: {error}")
            continue
        record.clips.append(clip.name)
        for flag in prepared.assignment.flags:
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
