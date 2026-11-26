"""CrisperWhisper 2 worker: verbatim and intended transcripts (environment ``cw2``).

CrisperWhisper 2.0 large (nyra labs) is a Whisper model trained for verbatim transcription:
the *verbatim* mode keeps fillers, repetitions, false starts and vocal events, the
*intended* mode gives the clean version the speaker meant. Both come with word timestamps
and confidences. It runs on the mother-only (time-gated) track, in Hungarian by default.
The transcripts stay local: they may contain names.
"""

import json
import re
import sys
from typing import Any

import numpy as np

from stillface.features.audio.models import (
    InferenceRequest,
    InferenceResult,
    Segment,
    SegmentScores,
)
from stillface.features.audio.workers.base import serve, torch_device

MODES: tuple[str, ...] = ("verbatim", "intended")
"""Transcription modes run for every request."""


MIN_RETRY_TOKENS: int = 32
"""Smallest token budget worth a retry after a too-long continuation prompt."""


def retry_token_budget(message: str) -> int | None:
    """New ``max_new_tokens`` after Whisper rejected a too-long decoder prompt.

    In long-form "continuation" decoding the previous chunk's text is the prompt of the next;
    in dense speech prompt + ``max_new_tokens`` can exceed the decoder's positions.

    Args:
        message: The ValueError message from transformers' generate.

    Returns:
        ``max_target_positions - prompt length - 1``; None for another error or when fewer
        than :data:`MIN_RETRY_TOKENS` tokens would remain.
    """
    prompt = re.search(r"previous tokens, is (\d+)", message)
    limit = re.search(r"max_target_positions` of the Whisper model: (\d+)", message)
    if not prompt or not limit:
        return None
    budget = int(limit.group(1)) - int(prompt.group(1)) - 1
    return budget if budget >= MIN_RETRY_TOKENS else None


def selected_modes(options: dict[str, Any]) -> tuple[str, ...]:
    """Transcription modes to run.

    Args:
        options: Worker options; ``modes`` is a comma-separated subset of :data:`MODES`.

    Returns:
        The modes in :data:`MODES` order; both when not set.

    Raises:
        ValueError: For an unknown mode.
    """
    asked = [m.strip() for m in str(options.get("modes", "")).split(",") if m.strip()]
    unknown = [m for m in asked if m not in MODES]
    if unknown:
        raise ValueError(f"Unknown CrisperWhisper modes {unknown}, expected {MODES}")
    return tuple(m for m in MODES if m in asked) if asked else MODES


def word_scores(words: list[tuple[float, float, str, float]], duration_s: float) -> SegmentScores:
    """Words as scored segments, clipped to the clip.

    Args:
        words: ``(start, end, word, probability)`` tuples.
        duration_s: Clip duration.

    Returns:
        One segment per word (label = the word), value ``probability``.
    """
    segments, values = [], []
    for start, end, word, probability in words:
        a, b = max(0.0, float(start)), min(duration_s, float(end))
        segments.append(Segment(a, max(a, b), "mother", word.strip()))
        values.append(float(np.clip(probability, 0.0, 1.0)))
    return SegmentScores(tuple(segments), ("probability",), np.array(values).reshape(-1, 1))


def postprocess(
    texts: dict[str, str],
    words: dict[str, list[tuple[float, float, str, float]]],
    duration_s: float,
) -> InferenceResult:
    """Result groups and the transcript artefact of one request.

    Args:
        texts: Mode mapped to its full text.
        words: Mode mapped to its words.
        duration_s: Clip duration.

    Returns:
        Result with ``<mode>_words`` segment groups and a ``transcript.json`` artefact
        (filled in by the caller with the request key).
    """
    segments = {f"{mode}_words": word_scores(words.get(mode, []), duration_s) for mode in MODES}
    document = {
        mode: {
            "text": texts.get(mode, ""),
            "words": [
                {
                    "start": round(a, 3),
                    "end": round(b, 3),
                    "word": w.strip(),
                    "p": round(float(p), 3),
                }
                for a, b, w, p in words.get(mode, [])
            ],
        }
        for mode in MODES
    }
    artefact = json.dumps(document, ensure_ascii=False, indent=1)
    return InferenceResult("", segments=segments, artefacts={"transcript.json": artefact})


class CrisperWhisperWorker:
    """Transcribes gated tracks in both modes."""

    def load(self, options: dict[str, Any]) -> dict[str, Any]:
        """Load CrisperWhisper (transformers backend)."""
        from importlib.metadata import version

        from crisperwhisper import CrisperWhisperModel

        self.language = options.get("language", "hu")
        self.modes = selected_modes(options)
        self.device = torch_device(options.get("device", "cpu"))
        # The package's "auto" means CUDA or CPU; the device must be passed for MPS.
        size = options.get("size") or "large"
        self.model = CrisperWhisperModel(size, device=self.device)
        return {
            "device": self.device,
            "crisperwhisper": version("crisperwhisper"),
            "language": self.language,
            "modes": ",".join(self.modes),
        }

    def forward(
        self, wav: str
    ) -> tuple[dict[str, str], dict[str, list[tuple[float, float, str, float]]]]:
        """Both transcripts of a WAV."""
        texts, words = {}, {}
        for mode in self.modes:
            try:
                result = self.model.transcribe(
                    wav, language=self.language, mode=mode, word_confidence=True
                )
            except ValueError as error:
                budget = retry_token_budget(str(error))
                if budget is None:
                    raise
                result = self.model.transcribe(
                    wav,
                    language=self.language,
                    mode=mode,
                    word_confidence=True,
                    max_new_tokens=budget,
                )
            texts[mode] = result.text or ""
            words[mode] = [(w.start, w.end, w.word, w.probability) for w in result.words or []]
        return texts, words

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Transcribe one gated track.

        Args:
            request: Whole-track request.

        Returns:
            Word groups and the transcript artefact.
        """
        texts, words = self.forward(str(request.wav))
        result = postprocess(texts, words, request.duration_s)
        result.key = request.key
        return result


if __name__ == "__main__":
    sys.exit(serve(CrisperWhisperWorker()))
