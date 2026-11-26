"""Tiny worker for the subprocess adapter tests (runs with the current interpreter).

Request options: ``fail`` (raise for this request), ``crash`` (exit 3 after the first
request), ``sleep`` (seconds to sleep in load), ``read`` (fail unless the WAV exists).
"""

import sys
import time

import numpy as np

from stillface.features.audio.models import FrameScores, InferenceResult, SegmentScores
from stillface.features.audio.workers.base import serve


class EchoWorker:
    def load(self, options):
        time.sleep(float(options.get("sleep", 0)))
        self.options = options
        return {"device": "cpu"}

    def infer(self, request):
        if request.options.get("fail"):
            raise ValueError("asked to fail")
        if request.options.get("crash"):
            sys.exit(3)
        if request.options.get("read") and not request.wav.is_file():
            raise FileNotFoundError(str(request.wav))
        times = np.arange(3) * 0.5
        frames = {"scores": FrameScores(times, ("a", "b"), np.ones((3, 2)) * request.duration_s)}
        segments = {}
        if request.segments:
            segments["seg"] = SegmentScores(
                request.segments, ("p",), np.full((len(request.segments), 1), 0.5)
            )
        return InferenceResult(request.key, frames, segments, {"txt": request.key})


if __name__ == "__main__":
    sys.exit(serve(EchoWorker()))
