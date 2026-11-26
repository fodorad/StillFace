"""Montreal Forced Aligner worker: word and phone timing of the mother (environment ``mfa``).

The whole phase is aligned as one file: the WAV and an input TextGrid with one interval per
mother utterance (label = its verbatim text), with the Hungarian MFA dictionary and acoustic
model (``hungarian_cv`` 2.0, XPF phones). MFA returns ``words`` and ``phones`` tiers in phase
time (JSON output). One file per phase is MFA's long-recording mode; a corpus of many short
utterance files made MFA 3.4 pair alignments with the wrong files. Words MFA could not align
(``spn`` phones or ``<unk>``) are reported as out-of-vocabulary, never dropped.
"""

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from stillface.features.audio.models import (
    InferenceRequest,
    InferenceResult,
    Segment,
    SegmentScores,
)
from stillface.features.audio.textgrid import (
    EMPTY_LABELS,
    Interval,
    write_textgrid,
)
from stillface.features.audio.workers.base import serve

UNKNOWN: frozenset[str] = frozenset({"<unk>", "spn"})
"""Labels MFA gives to words or phones it could not align."""


TIER: str = "mother"
"""Speaker tier of the input TextGrid."""


def input_textgrid(utterances: Sequence[Segment], duration_s: float) -> str:
    """MFA input TextGrid of a phase: one interval per utterance, labelled with its text.

    Padded mother intervals may overlap; an utterance then starts where the previous ends.

    Args:
        utterances: Mother utterances in phase time (label = text).
        duration_s: Phase duration.

    Returns:
        TextGrid content with the :data:`TIER` tier.
    """
    intervals: list[Interval] = []
    for u in sorted(utterances, key=lambda x: x.start_s):
        start = max(u.start_s, intervals[-1].end_s if intervals else 0.0)
        end = min(u.end_s, duration_s)
        if end > start:
            intervals.append(Interval(start, end, u.label))
    return write_textgrid({TIER: intervals}, duration_s)


def parse_alignment_json(text: str) -> dict[str, list[Interval]]:
    """Tiers of an MFA JSON alignment.

    Args:
        text: ``--output_format json`` file content.

    Returns:
        Tier name mapped to its intervals, in file order.
    """
    data = json.loads(text)
    return {
        name: [Interval(float(a), float(b), str(label)) for a, b, label in tier["entries"]]
        for name, tier in data.get("tiers", {}).items()
    }


def normalise_word(word: str) -> str:
    """A transcript word as MFA's dictionary spells it: lower case, no edge punctuation.

    Args:
        word: Transcript token, e.g. ``"Baba!"``.

    Returns:
        E.g. ``"baba"``.
    """
    return word.strip().strip(".,!?;:\"'()[]«»„”…-–").lower()


def phase_alignment(
    utterances: Sequence[Segment], tiers: dict[str, list[Interval]], duration_s: float
) -> tuple[dict[str, list[Interval]], list[str]]:
    """Kept word and phone tiers and the out-of-vocabulary words of a phase.

    Args:
        utterances: The aligned utterances (label = text).
        tiers: MFA's tiers in phase time.
        duration_s: Phase duration.

    Returns:
        ``({"words": ..., "phones": ...}, oov)`` without silences; ``oov`` lists, per
        utterance in order, the transcript words MFA could not align.
    """
    kept = {
        name: [
            i for i in tiers.get(name, []) if i.label not in EMPTY_LABELS and i.start_s < duration_s
        ]
        for name in ("words", "phones")
    }
    oov: list[str] = []
    for u in utterances:
        aligned = [w for w in kept["words"] if u.start_s <= (w.start_s + w.end_s) / 2 < u.end_s]
        expected = u.label.split()
        if any(w.label in UNKNOWN for w in aligned) or len(aligned) < len(expected):
            known = {w.label for w in aligned if w.label not in UNKNOWN}
            oov += [w for w in expected if normalise_word(w) not in known]
    return kept, oov


WIDE_BEAM: tuple[str, ...] = ("--beam", "100", "--retry_beam", "400")
"""MFA's advice when no utterance aligns at the default beam (10 / 40)."""


def retry_arguments(message: str) -> list[str] | None:
    """Extra ``mfa align`` arguments for a retry after a failed alignment.

    Args:
        message: The failure message.

    Returns:
        A wider beam after ``NoAlignmentsError``; None for other failures.
    """
    return list(WIDE_BEAM) if "NoAlignmentsError" in message else None


def activated_env(bin_dir: Path, base: Mapping[str, str]) -> dict[str, str]:
    """Environment variables of an activated conda / pixi environment.

    MFA calls its third-party tools (OpenFst, Kaldi) by name, so the environment's ``bin``
    must lead the ``PATH`` even when the worker runs without activation.

    Args:
        bin_dir: The environment's ``bin`` folder.
        base: Inherited environment.

    Returns:
        A copy of ``base`` with ``bin_dir`` first on the ``PATH``.
    """
    path = base.get("PATH", "")
    return {**base, "PATH": f"{bin_dir}{os.pathsep}{path}" if path else str(bin_dir)}


def run_command(args: Sequence[str], env: Mapping[str, str]) -> str:
    """Run a command and return its standard output.

    Args:
        args: Command and arguments.
        env: Extra environment variables on top of the current ones.

    Returns:
        Standard output.

    Raises:
        RuntimeError: With the end of the standard error when the command fails.
    """
    result = subprocess.run(list(args), capture_output=True, text=True, env={**os.environ, **env})
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip()[-1500:]
        raise RuntimeError(
            f"{Path(args[0]).name} {args[1] if len(args) > 1 else ''} failed: {tail}"
        )
    return result.stdout


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
        bin_dir = Path(sys.executable).parent
        self.mfa = str(bin_dir / "mfa")
        self.env = activated_env(bin_dir, os.environ)
        for kind, name in (("dictionary", self.dictionary), ("acoustic", self.acoustic)):
            run_command([self.mfa, "model", "download", kind, name], self.env)
        version = run_command([self.mfa, "version"], self.env)
        return {"device": "cpu", "mfa": version.strip(), "dictionary": self.dictionary}

    def forward(self, request: InferenceRequest) -> dict[str, list[Interval]]:
        """Align every utterance of the request in one MFA run over the whole phase."""
        utterances = [s for s in request.segments or () if s.label.strip()]
        if not utterances:
            return {}
        with tempfile.TemporaryDirectory() as tmp:
            # A unique corpus name: MFA keeps a database per corpus name.
            corpus = Path(tmp) / f"stillface_{Path(tmp).name}"
            out = Path(tmp) / "out"
            corpus.mkdir()
            (corpus / "phase.wav").symlink_to(Path(request.wav).resolve())
            (corpus / "phase.TextGrid").write_text(input_textgrid(utterances, request.duration_s))
            command = [
                self.mfa,
                "align",
                str(corpus),
                self.dictionary,
                self.acoustic,
                str(out),
                "--clean",
                "--quiet",
                "--output_format",
                "json",
            ]
            try:
                run_command(command, self.env)
            except RuntimeError as error:
                extra = retry_arguments(str(error))
                if extra is None:
                    raise
                run_command([*command, *extra], self.env)
            result = out / "phase.json"
            return parse_alignment_json(result.read_text()) if result.exists() else {}

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Phase TextGrid, word and phone groups, OOV list.

        Args:
            request: Segment request; each segment's label is its transcript.

        Returns:
            ``words`` / ``phones`` groups, ``TextGrid`` artefact and ``oov`` in meta.
        """
        utterances = [s for s in request.segments or () if s.label.strip()]
        tiers, oov = phase_alignment(utterances, self.forward(request), request.duration_s)
        return InferenceResult(
            request.key,
            segments=to_scores(tiers),
            artefacts={"TextGrid": write_textgrid(tiers, request.duration_s)},
            meta={"oov": oov},
        )


if __name__ == "__main__":
    sys.exit(serve(MfaWorker()))
