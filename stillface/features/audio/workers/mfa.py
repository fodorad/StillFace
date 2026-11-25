"""Montreal Forced Aligner worker: word and phone timing of the mother (environment ``mfa``).

For each mother utterance (segment label = its verbatim text) the clip and its transcript
are aligned with the Hungarian MFA dictionary and acoustic model (``hungarian_cv`` 2.0, XPF
phones). The per-utterance TextGrids are shifted to phase time and merged into one phase
TextGrid with ``words`` and ``phones`` tiers. Words MFA could not align (``spn`` phones or
``<unk>``) are reported as out-of-vocabulary, never dropped.
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from stillface.features.audio.media import read_wav, write_wav
from stillface.features.audio.models import (
    InferenceRequest,
    InferenceResult,
    Segment,
    SegmentScores,
)
from stillface.features.audio.textgrid import (
    EMPTY_LABELS,
    Interval,
    parse_textgrid,
    shift,
    write_textgrid,
)
from stillface.features.audio.workers.base import serve

UNKNOWN: frozenset[str] = frozenset({"<unk>", "spn"})
"""Labels MFA gives to words or phones it could not align."""


def merge_alignments(
    utterances: list[tuple[Segment, dict[str, list[Interval]]]], duration_s: float
) -> tuple[dict[str, list[Interval]], list[str]]:
    """Phase-level tiers and out-of-vocabulary words from per-utterance alignments.

    Args:
        utterances: Each mother segment with its utterance-local TextGrid tiers.
        duration_s: Phase duration.

    Returns:
        ``({"words": ..., "phones": ...}, oov)`` with silences removed and times in phase
        time; ``oov`` lists the transcript words MFA could not align, in order.
    """
    tiers: dict[str, list[Interval]] = {"words": [], "phones": []}
    oov: list[str] = []
    for segment, grid in utterances:
        expected = segment.label.split()
        aligned = [i for i in grid.get("words", []) if i.label not in EMPTY_LABELS]
        for name in tiers:
            kept = [i for i in grid.get(name, []) if i.label not in EMPTY_LABELS]
            tiers[name] += [i for i in shift(kept, segment.start_s) if i.start_s < duration_s]
        if any(i.label in UNKNOWN for i in aligned) or len(aligned) < len(expected):
            known = {i.label for i in aligned if i.label not in UNKNOWN}
            oov += [w for w in expected if w not in known]
    return tiers, oov


def to_scores(tiers: dict[str, list[Interval]]) -> dict[str, SegmentScores]:
    """Tiers as segment groups (label = word or phone, value = duration).

    Args:
        tiers: Phase-level tiers.

    Returns:
        ``words`` and ``phones`` groups.
    """
    out = {}
    for name, intervals in tiers.items():
        segments = tuple(Segment(i.start_s, i.end_s, "mother", i.label) for i in intervals)
        durations = np.array([[i.end_s - i.start_s] for i in intervals]).reshape(-1, 1)
        out[name] = SegmentScores(segments, ("duration_s",), durations)
    return out


class MfaWorker:
    """Aligns the mother's utterances of one phase."""

    def load(self, options: dict[str, Any]) -> dict[str, Any]:
        """Check the MFA command and download the Hungarian models if needed."""
        self.dictionary = options.get("dictionary", "hungarian_cv")
        self.acoustic = options.get("acoustic_model", "hungarian_cv")
        self.mfa = str(Path(sys.executable).with_name("mfa"))
        for kind, name in (("dictionary", self.dictionary), ("acoustic", self.acoustic)):
            subprocess.run(
                [self.mfa, "model", "download", kind, name], check=True, capture_output=True
            )
        version = subprocess.run([self.mfa, "version"], check=True, capture_output=True, text=True)
        return {"device": "cpu", "mfa": version.stdout.strip(), "dictionary": self.dictionary}

    def forward(self, request: InferenceRequest) -> list[tuple[Segment, dict[str, list[Interval]]]]:
        """Align every utterance of the request with one MFA run."""
        samples, rate = read_wav(request.wav)
        utterances = [s for s in request.segments or () if s.label.strip()]
        with tempfile.TemporaryDirectory() as tmp:
            corpus, out = Path(tmp) / "corpus", Path(tmp) / "out"
            corpus.mkdir()
            for i, s in enumerate(utterances):
                write_wav(
                    corpus / f"u{i:04d}.wav",
                    samples[round(s.start_s * rate) : round(s.end_s * rate)],
                    rate,
                )
                (corpus / f"u{i:04d}.lab").write_text(s.label)
            if utterances:
                subprocess.run(
                    [
                        self.mfa,
                        "align",
                        str(corpus),
                        self.dictionary,
                        self.acoustic,
                        str(out),
                        "--clean",
                        "--quiet",
                    ],
                    check=True,
                    capture_output=True,
                )
            return [
                (
                    s,
                    parse_textgrid(grid.read_text())
                    if (grid := out / f"u{i:04d}.TextGrid").exists()
                    else {},
                )
                for i, s in enumerate(utterances)
            ]

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Phase TextGrid, word and phone groups, OOV list.

        Args:
            request: Segment request; each segment's label is its transcript.

        Returns:
            ``words`` / ``phones`` groups, ``TextGrid`` artefact and ``oov`` in meta.
        """
        tiers, oov = merge_alignments(self.forward(request), request.duration_s)
        return InferenceResult(
            request.key,
            segments=to_scores(tiers),
            artefacts={"TextGrid": write_textgrid(tiers, request.duration_s)},
            meta={"oov": oov},
        )


if __name__ == "__main__":
    sys.exit(serve(MfaWorker()))
