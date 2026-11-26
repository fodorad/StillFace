"""Output contract of every audio model: what each one returns, and a check for it.

One entry per configured model names its result groups (frame or segment scores), the
score names, the value range of finite values, the nominal frame rate and what the values
mean. :func:`check_result` tests a real result against it; the smoke test and the opt-in
integration tests use it, and :func:`contract_rows` renders the table in the docs.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from stillface.features.audio.models import FrameScores, InferenceResult, SegmentScores

TIME_TOLERANCE_S: float = 0.05
"""How far a frame time or segment edge may lie beyond the input's end."""


@dataclass(frozen=True)
class GroupContract:
    """One result group of a model.

    Attributes:
        kind: ``frames`` (time series) or ``segments`` (scored intervals).
        group: Group key in the result.
        names: Exact score names in order; None when only the count is fixed.
        value_range: Inclusive bounds of the finite values; None when unbounded.
        meaning: What the values are.
        n_names: Number of score names when ``names`` is None.
        rate_hz: Nominal frame rate (frames) for the docs; None for segments.
    """

    kind: str
    group: str
    names: tuple[str, ...] | None
    value_range: tuple[float, float] | None
    meaning: str
    n_names: int | None = None
    rate_hz: float | None = None


@dataclass(frozen=True)
class ModelContract:
    """Everything one model returns.

    Attributes:
        model: Model name in the config.
        groups: Result groups.
        artefacts: Text artefacts the result must carry.
    """

    model: str
    groups: tuple[GroupContract, ...]
    artefacts: tuple[str, ...] = ()


PROB = (0.0, 1.0)

CONTRACTS: dict[str, ModelContract] = {
    c.model: c
    for c in (
        ModelContract(
            "silero",
            (
                GroupContract(
                    "frames", "speech", ("speech",), PROB, "speech probability", None, 31.25
                ),
            ),
        ),
        ModelContract(
            "pyannote_vad",
            (
                GroupContract(
                    "frames",
                    "speech",
                    ("speech",),
                    PROB,
                    "speech probability (max over local speakers, chunks aggregated)",
                    None,
                    59.0,
                ),
            ),
        ),
        ModelContract(
            "vtc",
            (
                GroupContract(
                    "frames",
                    "voice_type",
                    ("KCHI", "OCH", "MAL", "FEM"),
                    PROB,
                    "voice-type probabilities: key child, other child, male, female adult",
                    None,
                    50.0,
                ),
                GroupContract(
                    "segments",
                    "voice_type",
                    ("mean_prob",),
                    PROB,
                    "thresholded voice-type segments (label = class), mean probability",
                ),
            ),
            ("rttm",),
        ),
        ModelContract(
            "ast",
            (
                GroupContract(
                    "frames",
                    "audioset",
                    None,
                    PROB,
                    "AudioSet class probabilities, 2 s windows",
                    527,
                    4.0,
                ),
            ),
        ),
        ModelContract(
            "prosody",
            (
                GroupContract(
                    "frames",
                    "f0_adult",
                    ("f0_hz",),
                    (100.0, 700.0),
                    "F0 in the adult range (Hz)",
                    None,
                    100.0,
                ),
                GroupContract(
                    "frames",
                    "f0_infant",
                    ("f0_hz",),
                    (150.0, 1000.0),
                    "F0 in the infant range (Hz)",
                    None,
                    100.0,
                ),
                GroupContract(
                    "frames",
                    "intensity",
                    ("intensity_db",),
                    (-50.0, 130.0),
                    "intensity (dB)",
                    None,
                    100.0,
                ),
                GroupContract(
                    "frames",
                    "hnr",
                    ("hnr_db",),
                    (-200.0, 100.0),
                    "harmonics-to-noise ratio (dB)",
                    None,
                    100.0,
                ),
            ),
        ),
        ModelContract(
            "egemaps",
            (
                GroupContract(
                    "segments",
                    "egemaps",
                    None,
                    None,
                    "eGeMAPS v02 functionals per role segment",
                    88,
                ),
            ),
        ),
        ModelContract(
            "littlebeats",
            (
                GroupContract(
                    "frames",
                    "speaker",
                    ("SIL", "CHN", "FAN", "MAN", "CXN"),
                    PROB,
                    "speaker: silence, key child, female adult, male adult, other child",
                    None,
                    1.0,
                ),
                GroupContract(
                    "frames",
                    "infant",
                    ("CRY", "FUS", "BAB"),
                    PROB,
                    "infant vocalisation: cry, fuss, babble",
                    None,
                    1.0,
                ),
                GroupContract(
                    "frames",
                    "adult",
                    ("CDS", "ADS", "LAU", "SNG"),
                    PROB,
                    "adult vocalisation: child-directed, adult-directed, laugh, sing",
                    None,
                    1.0,
                ),
            ),
        ),
        ModelContract(
            "addressee",
            (
                GroupContract(
                    "segments",
                    "addressee",
                    ("KCDS", "ADS", "Other"),
                    PROB,
                    "addressee of each adult segment: key child, adult, other",
                ),
            ),
        ),
        ModelContract(
            "pyannote_embedding",
            (
                GroupContract(
                    "segments",
                    "embedding",
                    ("duration_s",),
                    (0.0, 1e6),
                    "segment duration; L2-normalised speaker embedding per segment",
                ),
            ),
        ),
        ModelContract(
            "audeering",
            (
                GroupContract(
                    "segments",
                    "windows",
                    ("arousal", "dominance", "valence"),
                    PROB,
                    "dimensional affect per 3 s window inside role segments",
                ),
            ),
        ),
        ModelContract(
            "emotion2vec",
            (
                GroupContract(
                    "segments",
                    "windows",
                    (
                        "angry",
                        "disgusted",
                        "fearful",
                        "happy",
                        "neutral",
                        "other",
                        "sad",
                        "surprised",
                        "unknown",
                    ),
                    PROB,
                    "emotion category probabilities per 3 s window (mother)",
                ),
            ),
        ),
        ModelContract(
            "crisperwhisper",
            (
                GroupContract(
                    "segments",
                    "verbatim_words",
                    ("probability",),
                    PROB,
                    "verbatim words (fillers, repetitions kept), word probability",
                ),
                GroupContract(
                    "segments",
                    "intended_words",
                    ("probability",),
                    PROB,
                    "intended words (cleaned), word probability",
                ),
            ),
            ("transcript.json",),
        ),
        ModelContract(
            "mfa",
            (
                GroupContract(
                    "segments",
                    "words",
                    ("duration_s",),
                    (0.0, 1e6),
                    "force-aligned words of the mother",
                ),
                GroupContract(
                    "segments",
                    "phones",
                    ("duration_s",),
                    (0.0, 1e6),
                    "force-aligned phones (XPF)",
                ),
            ),
            ("TextGrid",),
        ),
    )
}
"""Output contract of every model in the default config."""


def _check_values(where: str, values: np.ndarray, contract: GroupContract) -> list[str]:
    problems = []
    finite = values[np.isfinite(values)]
    if values.size and finite.size == 0:
        problems.append(f"{where}: no finite values")
    if contract.value_range is not None and finite.size:
        low, high = contract.value_range
        if finite.min() < low or finite.max() > high:
            problems.append(
                f"{where}: values {finite.min():.4g}..{finite.max():.4g} outside range "
                f"[{low}, {high}]"
            )
    return problems


def _check_names(where: str, names: Sequence[str], contract: GroupContract) -> list[str]:
    if contract.names is not None and tuple(names) != contract.names:
        return [f"{where}: names {tuple(names)[:6]} != {contract.names[:6]}"]
    if contract.names is None and contract.n_names is not None and len(names) != contract.n_names:
        return [f"{where}: {len(names)} names, expected {contract.n_names}"]
    return []


def check_result(contract: ModelContract, result: InferenceResult, duration_s: float) -> list[str]:
    """Problems of a result against a model's contract.

    Args:
        contract: The model's contract.
        result: A real result.
        duration_s: Duration of the input clip.

    Returns:
        One message per problem; empty when the result keeps the contract.
    """
    problems: list[str] = []
    end = duration_s + TIME_TOLERANCE_S
    for g in contract.groups:
        where = f"{g.kind}/{g.group}"
        store = result.frames if g.kind == "frames" else result.segments
        if g.group not in store:
            problems.append(f"missing {where}")
            continue
        item = store[g.group]
        problems += _check_names(where, item.names, g)
        problems += _check_values(where, np.asarray(item.values, dtype=float), g)
        if isinstance(item, FrameScores):
            times = np.asarray(item.times, dtype=float)
            if np.any(np.diff(times) < 0):
                problems.append(f"{where}: times are not ordered")
            if times.size and (times.min() < 0 or times.max() > end):
                problems.append(f"{where}: times {times.min():.3f}..{times.max():.3f} outside")
        elif isinstance(item, SegmentScores):
            for s in item.segments:
                if s.end_s < s.start_s:
                    problems.append(f"{where}: segment ends before it starts ({s})")
                    break
                if s.start_s < 0 or s.end_s > end:
                    problems.append(f"{where}: segment outside the input ({s})")
                    break
    for name in contract.artefacts:
        if name not in result.artefacts:
            problems.append(f"missing artefact {name}")
    return problems


def contract_rows() -> list[str]:
    """Markdown table rows of every contract group (the docs table is checked against them).

    Returns:
        ``| model | kind/group | names | range | rate | meaning |`` rows.
    """
    rows = []
    for contract in CONTRACTS.values():
        for g in contract.groups:
            names = ", ".join(g.names) if g.names else f"{g.n_names} names"
            bounds = f"{g.value_range[0]:g} .. {g.value_range[1]:g}" if g.value_range else "-"
            rate = f"{g.rate_hz:g} Hz" if g.rate_hz else "per segment"
            rows.append(
                f"| {contract.model} | {g.kind}/{g.group} | {names} | {bounds} | {rate} | "
                f"{g.meaning} |"
            )
    return rows
