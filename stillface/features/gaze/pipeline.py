"""Gaze extraction pipeline: heads, gaze model, classes, timelines and overlays."""

import itertools
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np

from stillface.config import DatasetPaths
from stillface.constants import CAMERAS, PHASES
from stillface.features.gaze.classes import GazeDecision, classify
from stillface.features.gaze.config import GazeConfig
from stillface.features.gaze.estimator import GazeEstimator
from stillface.features.gaze.heads import HeadBox, HeadTracker
from stillface.features.gaze.prepare import ClipHeads, prepare_clip
from stillface.features.gaze.roles import SIDE_CAMERAS, RoleOverrides, camera_persons
from stillface.features.gaze.runner import ClipOutput, run_session
from stillface.features.overlay import OverlayItem
from stillface.features.run_record import RunRecord
from stillface.features.timeline import TimelineRow

logger = logging.getLogger(__name__)


def read_frames(clip: Path, frame_indices: Sequence[int]) -> Iterator[np.ndarray]:
    """Stream selected frames of a clip.

    Args:
        clip: Video file.
        frame_indices: Ascending frame indices to return.

    Yields:
        RGB ``uint8`` frames of shape ``(height, width, 3)``, in the order of the indices.
        The stream ends early if the clip has fewer decodable frames than its metadata
        claims (decoders disagree by a frame on real clips); callers must handle that.

    Raises:
        RuntimeError: If the clip cannot be read.
    """
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot read clip: {clip}")
    try:
        wanted = iter(frame_indices)
        target = next(wanted, None)
        index = 0
        while target is not None:
            if not cap.grab():
                logger.warning("%s ended at frame %d, before frame %d", clip.name, index, target)
                return
            if index == target:
                ok, frame = cap.retrieve()
                if not ok:
                    raise RuntimeError(f"Cannot decode frame {index} of {clip}")
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                target = next(wanted, None)
            index += 1
    finally:
        cap.release()


def _decide(
    prepared: ClipHeads,
    camera: str,
    clip: Path,
    estimator: GazeEstimator,
    config: GazeConfig,
    batch_size: int,
) -> dict[tuple[int, str], GazeDecision]:
    """Estimate and classify the gaze of every person on every sampled frame."""
    persons = camera_persons(camera)
    decisions: dict[tuple[int, str], GazeDecision] = {}
    with_head = [
        k for k in range(len(prepared.frame_indices))
        if any(prepared.heads[p][k] is not None for p in persons)
    ]  # fmt: skip
    frames = read_frames(clip, [prepared.frame_indices[k] for k in with_head])

    decoded: set[int] = set()
    for positions in itertools.batched(with_head, batch_size, strict=False):
        images = list(itertools.islice(frames, len(positions)))
        positions = positions[: len(images)]
        if not images:
            break
        decoded.update(positions)
        batch = np.stack(images)
        boxes: list[list[HeadBox | None]] = [
            [prepared.heads[p][k] for p in persons] for k in positions
        ]
        estimate = estimator.estimate(batch, boxes)
        for b, k in enumerate(positions):
            for slot, person in enumerate(persons):
                head = prepared.heads[person][k]
                if head is None:
                    continue
                partner = None
                if camera in SIDE_CAMERAS:
                    partner = prepared.heads["baby" if person == "mother" else "mother"][k]
                inout = None if estimate.inout is None else float(estimate.inout[b, slot])
                decisions[(k, person)] = classify(
                    person, camera, estimate.heatmap[b, slot], inout, partner, config
                )
    for k in with_head:
        if k not in decoded:
            for person in persons:
                if prepared.heads[person][k] is not None:
                    decisions[(k, person)] = GazeDecision.not_codable("no_frame")
    return decisions


def gaze_outputs(
    session_id: str,
    camera: str,
    phase: str,
    prepared: ClipHeads,
    decisions: dict[tuple[int, str], GazeDecision],
) -> ClipOutput:
    """Timeline rows and overlay annotations from the decisions of a clip.

    Args:
        session_id: Session identifier.
        camera: Camera name.
        phase: Experiment phase.
        prepared: Heads of the clip.
        decisions: ``(sample position, person)`` mapped to the decision; people without a
            head on a frame are not codable.

    Returns:
        Rows and overlay items of every sampled frame.
    """
    rows: list[TimelineRow] = []
    items: dict[int, list[OverlayItem]] = {}
    for k, frame in enumerate(prepared.frame_indices):
        items[frame] = []
        for person in camera_persons(camera):
            decision = decisions.get((k, person)) or GazeDecision.not_codable(
                prepared.reason_missing(camera, person)
            )
            rows.append(
                TimelineRow(
                    session_id=session_id,
                    camera=camera,
                    phase=phase,
                    frame_index=frame,
                    time_s=frame / prepared.clip_fps,
                    person=person,
                    gaze_x=decision.gaze_x,
                    gaze_y=decision.gaze_y,
                    inout=decision.inout,
                    partner_mass=decision.partner_mass,
                    gaze_class=decision.gaze_class,
                    reason=decision.reason,
                )
            )
            gaze = None if decision.gaze_x is None else (decision.gaze_x, decision.gaze_y or 0.0)
            label = decision.gaze_class if not decision.reason else f"NK:{decision.reason}"
            items[frame].append(OverlayItem(person, prepared.heads[person][k], gaze, label))
    return ClipOutput(rows, items, prepared.assignment.flags)


def run_gaze(
    paths: DatasetPaths,
    session_id: str,
    tracker: HeadTracker,
    estimator: GazeEstimator,
    config: GazeConfig,
    cameras: Sequence[str] = CAMERAS,
    phases: Sequence[str] = PHASES,
    overlay: bool = True,
    overwrite: bool = False,
    batch_size: int = 8,
) -> RunRecord:
    """Extract the gaze of the mother and the baby for one session.

    Per clip a timeline CSV and (optionally) an overlay clip are written; per phase a
    stacked overlay grid; per session ``run.json`` records the model variant used. See
    :func:`~stillface.features.gaze.runner.run_session` for the resume and failure
    behaviour.

    Args:
        paths: Dataset locations.
        session_id: Session identifier.
        tracker: Head detector and tracker.
        estimator: Gaze target model.
        config: Gaze configuration.
        cameras: Cameras to process.
        phases: Phases to process.
        overlay: Also render overlay clips and grids.
        overwrite: Recompute existing outputs.
        batch_size: Frames per model call.

    Returns:
        The run record, also written to ``run.json`` in the gaze folder.
    """

    def process(clip: Path, camera: str, phase: str, overrides: RoleOverrides) -> ClipOutput:
        prepared = prepare_clip(clip, camera, phase, tracker, config, overrides=overrides)
        decisions = _decide(prepared, camera, clip, estimator, config, batch_size)
        return gaze_outputs(session_id, camera, phase, prepared, decisions)

    return run_session(
        paths, session_id, paths.gaze_dir(session_id), process, estimator.variant, config,
        cameras, phases, overlay, overwrite,
    )  # fmt: skip
