"""Head detection with Grounding DINO prompted by "a human head".

Grounding DINO (IDEA Research, Apache-2.0, safetensors) detects objects described in
words. The single prompt "a human head." puts tight boxes on every visible head, adult or
infant, frontal, in profile, from behind or partly occluded, which a face detector misses.
Needs the ``features`` extra (``transformers``).
"""

import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from stillface.features.gaze.heads import HeadBox, HeadTrack
from stillface.features.gaze.pipeline import read_frames
from stillface.features.gaze.prepare import clip_properties
from stillface.features.gaze.tracking import (
    Candidate,
    anchor_frames,
    densify,
    link_candidates,
    stitch_tracks,
)
from stillface.features.io import write_atomic

logger = logging.getLogger(__name__)

DEFAULT_MODEL: str = "IDEA-Research/grounding-dino-base"
"""Hugging Face Grounding DINO checkpoint."""

HEAD_PROMPT: str = "a human head."
"""Text prompt; Grounding DINO expects lower case phrases ending with a period."""


class GroundingDinoHeadTracker:
    """:class:`~stillface.features.gaze.heads.HeadTracker` based on Grounding DINO.

    The detector runs on a sparse time grid (``rate_hz``), its detections are linked into
    tracks, and the tracks are interpolated to the requested frames.

    Args:
        prompt: Text prompt.
        model_name: Hugging Face checkpoint.
        rate_hz: Detection rate.
        box_threshold: Minimum box confidence.
        text_threshold: Minimum phrase confidence.
        max_gap_s: Longest gap between detections that is interpolated.
        stitch_gap_s: Longest detection dropout after which a head's track is continued.
        batch_size: Frames per model call.
        device: Torch device name, the best available if None.
        cache_dir: Folder caching the raw detections per clip, so re-running the role and
            fusion logic does not run the model again; no caching if None.
    """

    def __init__(
        self,
        prompt: str = HEAD_PROMPT,
        model_name: str = DEFAULT_MODEL,
        rate_hz: float = 2.0,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        max_gap_s: float = 1.5,
        stitch_gap_s: float = 3.0,
        batch_size: int = 4,
        device: str | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(model_name, use_safetensors=True)
            .eval()
            .to(device)
        )
        self.prompt = prompt
        self.rate_hz = rate_hz
        self.box_threshold, self.text_threshold = box_threshold, text_threshold
        self.max_gap_s, self.batch_size = max_gap_s, batch_size
        self.stitch_gap_s = stitch_gap_s
        self.model_name = model_name
        self.cache_dir = cache_dir

    def detect(self, frames: list[np.ndarray]) -> list[list[Candidate]]:
        """Head candidates on RGB frames, in normalized coordinates.

        Args:
            frames: RGB ``uint8`` frames of equal size.

        Returns:
            Candidates per frame.
        """
        import torch
        from PIL import Image

        images = [Image.fromarray(f) for f in frames]
        inputs = self.processor(
            images=images, text=[self.prompt] * len(images), return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        height, width = frames[0].shape[:2]
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(height, width)] * len(images),
        )
        scale = np.array([width, height, width, height], dtype=float)
        found = []
        for result in results:
            boxes = np.clip(result["boxes"].cpu().numpy() / scale, 0.0, 1.0)
            found.append(
                [
                    Candidate(HeadBox(*map(float, box)), float(score))
                    for box, score in zip(boxes, result["scores"].cpu().numpy(), strict=True)
                ]
            )
        return found

    def _cache_path(self, clip: Path, anchors: list[int]) -> Path | None:
        if self.cache_dir is None:
            return None
        settings = [self.model_name, self.prompt, self.box_threshold, self.text_threshold, anchors]
        digest = hashlib.sha1(json.dumps(settings).encode()).hexdigest()[:12]
        return self.cache_dir / f"{clip.parent.parent.name}_{clip.stem}_{digest}.json"

    def detect_clip(self, clip: Path, anchors: list[int]) -> list[list[Candidate]]:
        """Detect heads on the anchor frames of a clip, using the cache when possible.

        Args:
            clip: Video clip.
            anchors: Frames to detect on, ascending.

        Returns:
            Candidates per decodable anchor frame.
        """
        cache = self._cache_path(clip, anchors)
        if cache is not None and cache.exists():
            stored = json.loads(cache.read_text())
            return [[Candidate(HeadBox(*c[:4]), c[4]) for c in frame] for frame in stored]
        candidates: list[list[Candidate]] = []
        stream = read_frames(clip, anchors)
        while batch := [f for _, f in zip(range(self.batch_size), stream, strict=False)]:
            candidates += self.detect(batch)
        if cache is not None:
            payload = [
                [[c.box.x0, c.box.y0, c.box.x1, c.box.y1, c.score] for c in frame]
                for frame in candidates
            ]
            write_atomic(cache, lambda tmp: tmp.write_text(json.dumps(payload)))
        return candidates

    def track(self, clip: Path, frame_indices: Sequence[int]) -> list[HeadTrack]:
        """Detect heads on a sparse grid, link them and interpolate to the requested frames.

        Args:
            clip: Video clip.
            frame_indices: Frames to produce boxes for, ascending.

        Returns:
            One track per head.
        """
        _, fps = clip_properties(clip)
        anchors = anchor_frames(frame_indices, fps, min(self.rate_hz, fps))
        candidates = self.detect_clip(clip, anchors)
        linked = link_candidates(candidates, anchors[: len(candidates)])
        tracks = stitch_tracks(linked, max_gap=int(np.ceil(self.stitch_gap_s * fps)))
        max_gap = int(np.ceil(self.max_gap_s * fps))
        logger.info("%s: %d head tracks on %d anchors", clip.name, len(tracks), len(candidates))
        return [densify(t, list(frame_indices), max_gap) for t in tracks]
