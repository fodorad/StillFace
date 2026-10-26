"""Head pilot: heads, roles and overlays without any gaze model.

The pilot validates the detection, tracking and role assignment on chosen sessions before
the gaze model is run. It writes the same timeline format as the gaze pipeline, with empty
gaze fields, plus overlay clips showing the heads and their roles.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

from stillface.config import DatasetPaths
from stillface.constants import CAMERAS, PHASES
from stillface.features.gaze.classes import NK
from stillface.features.gaze.config import GazeConfig
from stillface.features.gaze.heads import HeadTracker
from stillface.features.gaze.prepare import ClipHeads, prepare_clip
from stillface.features.gaze.roles import RoleOverrides, camera_persons
from stillface.features.gaze.runner import ClipOutput, run_session
from stillface.features.overlay import OverlayItem
from stillface.features.run_record import RunRecord
from stillface.features.timeline import TimelineRow

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
                    reason="" if found else prepared.reason_missing(camera, person),
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
                "" if head is not None else f"NK:{prepared.reason_missing(camera, person)}",
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
    stacked overlay grid. See :func:`~stillface.features.gaze.runner.run_session` for the
    resume and failure behaviour. Pinned roles are read from the session's ``roles.json``
    in the gaze feature folder.

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

    def process(clip: Path, camera: str, phase: str, overrides: RoleOverrides) -> ClipOutput:
        prepared = prepare_clip(clip, camera, phase, tracker, config, overrides=overrides)
        return ClipOutput(
            head_rows(session_id, camera, phase, prepared),
            head_overlay_items(camera, prepared),
            prepared.assignment.flags,
        )

    return run_session(
        paths, session_id, paths.features_dir(session_id, PILOT_FEATURE), process, "none",
        config, cameras, phases, overlay, overwrite,
    )  # fmt: skip
