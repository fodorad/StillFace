"""ASR stage: what the mother said, when, and which words to distrust.

CrisperWhisper 2 runs on the mother-only (time-gated) track. Whisper models may "hear"
words in silence or in the baby's vocalisations, so every word whose midpoint falls outside
the padded mother segments is flagged ``outside_mother``. The verbatim words of each mother
segment form one utterance for the Montreal Forced Aligner, which gives word and phone
timing (speech rate, vowel measures).

This module also chooses the request builder of every model from its config
(:func:`builder_for`), so the CLI and the full chain share one wiring.
"""

import json
import logging
from collections.abc import Sequence

import numpy as np

from stillface.config import DatasetPaths
from stillface.constants import PHASES
from stillface.features.audio.config import AudioConfig, ModelConfig
from stillface.features.audio.infer import RequestBuilder, full_wav_request, load_raw, wav_duration
from stillface.features.audio.models import InferenceRequest, Segment
from stillface.features.audio.prepare import wav_path
from stillface.features.audio.provenance import read_sidecar
from stillface.features.audio.roles import segment_builder
from stillface.features.io import write_atomic

logger = logging.getLogger(__name__)

ASR_MODEL: str = "crisperwhisper"
"""Model name of the ASR in the config."""

ALIGNER_MODEL: str = "mfa"
"""Model name of the forced aligner in the config."""

TRACK: str = "mother"
"""Gated track the ASR reads."""


def gated_request(track: str) -> RequestBuilder:
    """Builder of whole-track requests on a gated track.

    Args:
        track: Gated track name (``mother`` or ``baby``).

    Returns:
        Builder with keys ``<session>/<phase>/<track>``; None if the track is missing.
    """

    def build(paths: DatasetPaths, session: str, phase: str) -> InferenceRequest | None:
        wav = wav_path(paths, session, phase, track)
        if not wav.exists():
            return None
        return InferenceRequest(f"{session}/{phase}/{track}", wav, wav_duration(wav))

    return build


def gated_intervals(
    paths: DatasetPaths, session: str, phase: str, track: str = TRACK
) -> list[tuple[float, float]]:
    """Padded intervals kept in a gated track.

    Args:
        paths: Dataset locations.
        session: Session identifier.
        phase: Experiment phase.
        track: Gated track.

    Returns:
        ``(start, end)`` intervals, empty if the track was not gated.
    """
    side = read_sidecar(wav_path(paths, session, phase, track))
    return [(float(a), float(b)) for a, b in side.get("intervals", [])] if side else []


def inside(time_s: float, intervals: Sequence[tuple[float, float]]) -> bool:
    """Whether a time lies in any interval.

    Args:
        time_s: Time.
        intervals: ``(start, end)`` pairs.

    Returns:
        True if ``start <= time_s < end`` for one of them.
    """
    return any(a <= time_s < b for a, b in intervals)


def utterances(words: Sequence[Segment], intervals: Sequence[tuple[float, float]]) -> list[Segment]:
    """Group words into one utterance per kept interval.

    Args:
        words: Word segments (label = word), phase time.
        intervals: Padded mother intervals.

    Returns:
        One segment per interval with words, its label the words joined by spaces.
    """
    out = []
    for a, b in intervals:
        text = [w.label for w in words if a <= (w.start_s + w.end_s) / 2 < b and w.label]
        if text:
            out.append(Segment(a, b, TRACK, " ".join(text)))
    return out


def utterance_request(asr_model: str = ASR_MODEL) -> RequestBuilder:
    """Builder of aligner requests: the mother track with one segment per utterance.

    Args:
        asr_model: Model whose verbatim words form the utterances.

    Returns:
        Builder; None if there is no transcript or no utterance.
    """

    def build(paths: DatasetPaths, session: str, phase: str) -> InferenceRequest | None:
        base = gated_request(TRACK)(paths, session, phase)
        result = load_raw(paths, asr_model, f"{session}/{phase}/{TRACK}")
        if base is None or result is None or "verbatim_words" not in result.segments:
            return None
        words = result.segments["verbatim_words"].segments
        found = utterances(words, gated_intervals(paths, session, phase))
        if not found:
            return None
        return InferenceRequest(base.key, base.wav, base.duration_s, tuple(found))

    return build


def builder_for(model: ModelConfig, config: AudioConfig) -> RequestBuilder:
    """Request builder of a configured model.

    ``wav`` models read the prepared phase WAV, or the gated track named by the ``track``
    option; ``segments`` models get their segments from ``segment_source`` (``vtc:``,
    ``roles:`` or ``asr:<track>`` for the aligner's utterances).

    Args:
        model: Model settings.
        config: Audio configuration.

    Returns:
        The builder.
    """
    if model.input == "wav":
        track = model.options.get("track", "")
        return gated_request(track) if track else full_wav_request
    if model.segment_source.startswith("asr:"):
        return utterance_request()
    return segment_builder(model.segment_source, config.roles.vtc_model)


def flag_words(
    words: Sequence[Segment], probabilities: np.ndarray, intervals: Sequence[tuple[float, float]]
) -> list[dict]:
    """Words with the ``outside_mother`` flag.

    Args:
        words: Word segments.
        probabilities: Word probabilities, shape ``(len(words), 1)``.
        intervals: Padded mother intervals.

    Returns:
        One record per word: start, end, word, probability, outside_mother.
    """
    return [
        {
            "start": round(w.start_s, 3),
            "end": round(w.end_s, 3),
            "word": w.label,
            "p": round(float(p), 3),
            "outside_mother": not inside((w.start_s + w.end_s) / 2, intervals),
        }
        for w, p in zip(words, np.asarray(probabilities).reshape(-1), strict=True)
    ]


def run_asr_flags(
    paths: DatasetPaths, session: str, config: AudioConfig, phases: Sequence[str] = PHASES
) -> dict[str, float]:
    """Write ``asr/<phase>.json`` with flagged words, texts and the aligner's OOV list.

    Args:
        paths: Dataset locations.
        session: Session identifier.
        config: Audio configuration.
        phases: Phases to process.

    Returns:
        Phase mapped to the share of verbatim words flagged ``outside_mother``.
    """
    shares = {}
    for phase in phases:
        key = f"{session}/{phase}/{TRACK}"
        result = load_raw(paths, ASR_MODEL, key)
        if result is None:
            continue
        intervals = gated_intervals(paths, session, phase)
        document: dict = {}
        for mode in ("verbatim", "intended"):
            group = result.segments.get(f"{mode}_words")
            if group is None:
                continue
            document[mode] = {
                "text": " ".join(s.label for s in group.segments),
                "words": flag_words(group.segments, group.values, intervals),
            }
        words = document.get("verbatim", {}).get("words", [])
        shares[phase] = float(np.mean([w["outside_mother"] for w in words])) if words else 0.0
        document["outside_mother_share"] = shares[phase]
        aligned = load_raw(paths, ALIGNER_MODEL, key)
        document["oov"] = aligned.meta.get("oov", []) if aligned is not None else None
        text = json.dumps(document, ensure_ascii=False, indent=1) + "\n"
        out = paths.audio_session_dir(session) / "asr" / f"{phase}.json"
        write_atomic(out, lambda tmp, t=text: tmp.write_text(t))
    logger.info("%s: ASR flags for %d phases", session, len(shares))
    return shares
