"""Head detection and tracking with exordium.

Faces are found with exordium's YOLO11 face detector on the sampled frames only and
followed with its IoU tracker. Needs the ``features`` extra (``exordium``).
"""

import logging
from collections.abc import Sequence
from pathlib import Path

from stillface.features.gaze.heads import HeadBox, HeadTrack

logger = logging.getLogger(__name__)


class ExordiumHeadTracker:
    """:class:`~stillface.features.gaze.heads.HeadTracker` backed by exordium.

    Args:
        device_id: Torch device index; negative selects the CPU, otherwise MPS or CUDA if available.
        conf: Detector confidence threshold.
        min_score: Detections below this score are not tracked.
        batch_size: Frames per detector call.
        max_lost: Frames a track may be missing before it is closed.
        iou_threshold: Minimum box overlap to continue a track.
    """

    def __init__(
        self,
        device_id: int = 0,
        conf: float = 0.25,
        min_score: float = 0.5,
        batch_size: int = 16,
        max_lost: int = 60,
        iou_threshold: float = 0.2,
    ) -> None:
        from exordium.video.face.detector.yolo11 import YoloFace11Detector

        self.detector = YoloFace11Detector(device_id=device_id, conf=conf, batch_size=batch_size)
        self.min_score = min_score
        self.batch_size = batch_size
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold

    def track(self, clip: Path, frame_indices: Sequence[int]) -> list[HeadTrack]:
        """Detect and track the faces on the sampled frames of a clip.

        Args:
            clip: Video clip.
            frame_indices: Frames to analyze, ascending.

        Returns:
            One track per face, with boxes in normalized coordinates.
        """
        from exordium.video.core.detection import IouTracker, VideoDetections
        from exordium.video.core.io import open_video

        detections = VideoDetections()
        with open_video(clip) as video:
            width, height = video.width, video.height
            for start in range(0, len(frame_indices), self.batch_size):
                chunk = list(frame_indices[start : start + self.batch_size])
                frames = video.get_frames_at(chunk)
                per_frame = self.detector.run_detector(list(frames))
                for frame_id, frame_det in zip(chunk, per_frame, strict=True):
                    detections.add(
                        self.detector._build_frame_detections(
                            frame_det, frame_id=frame_id, source=str(clip)
                        )
                    )

        tracker = IouTracker(max_lost=self.max_lost, iou_threshold=self.iou_threshold)
        tracker.label(detections, min_score=self.min_score, max_lost=self.max_lost,
                      iou_threshold=self.iou_threshold)  # fmt: skip
        tracks = []
        for track_id, found in tracker.tracks.items():
            boxes = {}
            for detection in found.detections:
                x0, y0, x1, y1 = (float(v) for v in detection.bb_xyxy)
                boxes[detection.frame_id] = HeadBox(
                    x0 / width, y0 / height, x1 / width, y1 / height
                )
            tracks.append(HeadTrack(track_id, boxes))
        logger.info("%s: %d face tracks on %d frames", clip.name, len(tracks), len(frame_indices))
        return tracks
