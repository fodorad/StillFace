"""Text-prompted head detection with OWLv2.

OWLv2 (Google, Apache-2.0, safetensors) finds objects described in words. Prompted with
"a baby's head" it puts a tight box on the baby's head even when the face is turned away,
reclined in profile or partly hidden, where face detectors fail. Because the prompt names
the person, the track carries its role. Needs the ``features`` extra (``transformers``).
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from stillface.features.gaze.combined import anchor_frames
from stillface.features.gaze.heads import HeadBox, HeadTrack, interpolate_boxes
from stillface.features.gaze.pipeline import read_frames
from stillface.features.gaze.prepare import clip_properties
from stillface.features.gaze.selection import Candidate, select_path

logger = logging.getLogger(__name__)

DEFAULT_MODEL: str = "google/owlv2-base-patch16-ensemble"
"""Hugging Face model used for the text-prompted detection."""

BABY_QUERY: str = "a baby's head"
"""Text prompt that finds the baby's head."""


class OwlHeadTracker:
    """:class:`~stillface.features.gaze.heads.HeadTracker` for one prompted person.

    The model runs on a sparse time grid (``rate_hz``); per anchor frame the candidate
    boxes are reduced to one smooth path (see :func:`select_path`), and the frames between
    anchors are interpolated.

    Args:
        query: Text prompt describing the head.
        role: ``mother`` or ``baby``, stored on the returned track.
        model_name: Hugging Face OWLv2 checkpoint.
        rate_hz: Anchor rate of the detector.
        max_candidates: Candidate boxes kept per frame.
        min_score: Candidates below this score are ignored.
        jump_penalty: Smoothness weight of the path selection.
        max_gap_s: Longest gap between anchors that is interpolated.
        batch_size: Frames per model call.
        device: Torch device name, the best available if None.
    """

    def __init__(
        self,
        query: str = BABY_QUERY,
        role: str = "baby",
        model_name: str = DEFAULT_MODEL,
        rate_hz: float = 3.0,
        max_candidates: int = 3,
        min_score: float = 0.1,
        jump_penalty: float = 2.0,
        max_gap_s: float = 1.5,
        batch_size: int = 4,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = (
            Owlv2ForObjectDetection.from_pretrained(model_name, use_safetensors=True)
            .eval()
            .to(device)
        )
        self.query, self.role = query, role
        self.rate_hz, self.max_candidates = rate_hz, max_candidates
        self.min_score, self.jump_penalty = min_score, jump_penalty
        self.max_gap_s, self.batch_size = max_gap_s, batch_size

    def _detect(self, frames: list[np.ndarray]) -> list[list[Candidate]]:
        """Top candidate boxes of the prompt on RGB frames, in normalized coordinates."""
        import torch
        from PIL import Image

        images = [Image.fromarray(f) for f in frames]
        inputs = self.processor(
            text=[[self.query]] * len(images), images=images, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
        height, width = frames[0].shape[:2]
        side = max(height, width)  # OWLv2 pads the image to a square
        results = self.processor.image_processor.post_process_object_detection(
            outputs,
            threshold=self.min_score,
            target_sizes=torch.tensor([[side, side]] * len(images)),
        )
        scale = np.array([width, height, width, height], dtype=float)
        found = []
        for result in results:
            boxes = np.clip(result["boxes"].cpu().numpy() / scale, 0.0, 1.0)
            order = np.argsort(-result["scores"].cpu().numpy())[: self.max_candidates]
            found.append(
                [
                    Candidate(HeadBox(*map(float, boxes[i])), float(result["scores"][i]))
                    for i in order
                ]
            )
        return found

    def track(self, clip: Path, frame_indices: Sequence[int]) -> list[HeadTrack]:
        """Find the prompted head on a sparse grid and interpolate to the requested frames.

        Args:
            clip: Video clip.
            frame_indices: Frames to produce boxes for, ascending.

        Returns:
            One role-tagged track, empty of boxes if the head was never found.
        """
        _, fps = clip_properties(clip)
        anchors = anchor_frames(frame_indices, fps, self.rate_hz)
        candidates: list[list[Candidate]] = []
        decoded: list[int] = []
        stream = read_frames(clip, anchors)
        while True:
            batch = [f for _, f in zip(range(self.batch_size), stream, strict=False)]
            if not batch:
                break
            candidates += self._detect(batch)
        decoded = anchors[: len(candidates)]
        path = select_path(candidates, self.jump_penalty)
        chosen = {frame: c for frame, c in zip(decoded, path, strict=True) if c is not None}
        max_gap = int(np.ceil(self.max_gap_s * fps))
        boxes = interpolate_boxes({f: c.box for f, c in chosen.items()}, frame_indices, max_gap)
        scores = {f: chosen[f].score for f in boxes if f in chosen}
        logger.info(
            "%s: %s found on %d of %d anchors", clip.name, self.query, len(chosen), len(anchors)
        )
        return [HeadTrack(0, boxes, scores=scores, role=self.role)]
