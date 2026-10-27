"""Head detection and tracking with exordium.

Faces are found with exordium's YOLO11 face detector on the sampled frames only and
followed with its IoU tracker. Needs the ``features`` extra (``exordium``).

The small model at 1280 px finds the mother and the baby on all cameras of the development
sessions with scores around 0.7-0.9; the nano model at the default 640 px lost the baby on
the window camera (scores below 0.15). Because a baby's face in profile still scores low on
some frames, tracks are started by confident detections but continue through weak ones.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from stillface.features.gaze.heads import HeadBox, HeadTrack

logger = logging.getLogger(__name__)


class ExordiumHeadTracker:
    """:class:`~stillface.features.gaze.heads.HeadTracker` backed by exordium.

    Args:
        device_id: Torch device index; negative selects the CPU, otherwise MPS or CUDA if available.
        model_name: exordium YOLO11 face model.
        imgsz: Detector input size in pixels (long side).
        conf: Detector confidence threshold; weak detections are kept for track continuity.
        start_score: A track needs confident detections (at least this score) to be kept.
        min_confident: Number of confident detections a track needs.
        batch_size: Frames per detector call.
        max_lost: Frames a track may be missing before it is closed; at sparse sampling at
            least three sampling steps are allowed.
        iou_threshold: Minimum box overlap to continue a track.
    """

    def __init__(
        self,
        device_id: int = 0,
        model_name: str = "yolo11s-pose_widerface",
        imgsz: int = 1280,
        conf: float = 0.1,
        start_score: float = 0.5,
        min_confident: int = 3,
        batch_size: int = 16,
        max_lost: int = 60,
        iou_threshold: float = 0.2,
    ) -> None:
        from exordium.video.face.detector.yolo11 import YoloFace11Detector

        self.detector = YoloFace11Detector(
            model_name=model_name, device_id=device_id, conf=conf, batch_size=batch_size
        )
        self.imgsz = imgsz
        self.conf = conf
        self.start_score = start_score
        self.min_confident = min_confident
        self.batch_size = batch_size
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold

    def _detect(self, frames):
        """Run the face detector at the configured input size on RGB frames ``(B, 3, H, W)``."""
        images = [f.permute(1, 2, 0).contiguous().cpu().numpy()[:, :, ::-1] for f in frames]
        results = self.detector.model.predict(
            images, conf=self.conf, imgsz=self.imgsz, verbose=False
        )
        return self.detector._parse_results(results)

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
            # Frame counts from different decoders can differ by one on the last frame.
            decodable = [i for i in frame_indices if i < video.num_frames]
            for start in range(0, len(decodable), self.batch_size):
                chunk = decodable[start : start + self.batch_size]
                frames = video.get_frames_at(chunk)
                per_frame = self._detect(frames)
                for frame_id, frame_det in zip(chunk, per_frame, strict=True):
                    detections.add(
                        self.detector._build_frame_detections(
                            frame_det, frame_id=frame_id, source=str(clip)
                        )
                    )

        step = int(np.median(np.diff(frame_indices))) if len(frame_indices) > 1 else 1
        max_lost = max(self.max_lost, 3 * step)
        tracker = IouTracker(max_lost=max_lost, iou_threshold=self.iou_threshold)
        tracker.label(detections, min_score=self.conf, max_lost=max_lost,
                      iou_threshold=self.iou_threshold)  # fmt: skip
        tracks = []
        for track_id, found in tracker.tracks.items():
            confident = sum(d.score >= self.start_score for d in found.detections)
            if confident < self.min_confident:
                continue
            boxes, scores = {}, {}
            for detection in found.detections:
                scores[detection.frame_id] = float(detection.score)
                x0, y0, x1, y1 = (float(v) for v in detection.bb_xyxy)
                boxes[detection.frame_id] = HeadBox(
                    x0 / width, y0 / height, x1 / width, y1 / height
                )
            tracks.append(HeadTrack(track_id, boxes, scores=scores, faces=dict(scores)))
        logger.info("%s: %d face tracks on %d frames", clip.name, len(tracks), len(frame_indices))
        return tracks
