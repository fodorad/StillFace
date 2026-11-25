"""Lanes, bands and events: the one data model of the audio pipeline.

Every audio signal is stored twice on a common phase-local grid (seconds from the start of
the phase clip, :data:`GRID_RATE_HZ`):

- a **lane**: the raw continuous value of one quantity from one source, with a fixed range,
  NaN where it does not apply (e.g. mother affect outside mother segments);
- a **band**: a boolean presence derived from a lane by a documented threshold or rule.

Bands become **events** (maximal runs with optional merging and a minimum duration). Lanes
and bands are written as one wide CSV per phase, events as a long CSV, and the metadata of
both as a legend JSON, so a table can always be traced back to the rule that produced it.
"""

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np

from stillface.features.io import write_atomic

GRID_RATE_HZ: float = 10.0
"""Sampling rate of the common lane grid."""

ROLES: tuple[str, ...] = ("baby", "mother", "other", "context", "dyad")
"""Who a lane, band or event describes; ``context`` covers non-vocal sounds."""

ORIGINS: tuple[str, ...] = ("vendor", "probe", "config")
"""Where a threshold comes from: the model's published default, the probe, or our config."""

EVENT_COLUMNS: tuple[str, ...] = (
    "start_s",
    "end_s",
    "role",
    "label",
    "confidence",
    "source",
    "threshold",
    "rule_id",
)
"""Event CSV columns, in file order."""

LEGEND_VERSION: int = 1
"""Version of the legend JSON layout."""


class Aggregation(StrEnum):
    """How native-rate samples falling into one grid bin are combined."""

    MAX = "max"
    """Maximum; used for class probabilities so short peaks survive."""
    MEAN = "mean"
    """Mean of the finite samples; used for continuous features."""
    MAJORITY = "majority"
    """Most frequent category code; ties go to the lower code."""


@dataclass(frozen=True)
class LaneSpec:
    """What a lane contains.

    Attributes:
        source: Producing model or stage, e.g. ``vtc`` or ``fusion``.
        quantity: Measured quantity, e.g. ``KCHI`` or ``f0``.
        role: One of :data:`ROLES`.
        unit: Unit of the values, e.g. ``prob``, ``Hz``, ``dB``.
        value_range: Inclusive ``(low, high)`` every finite value must respect.
        aggregation: Bin aggregation used by :func:`to_grid`.
        categories: Category names of a categorical lane; values are indices into it.
        note: Free-text description for the legend.
    """

    source: str
    quantity: str
    role: str
    unit: str
    value_range: tuple[float, float]
    aggregation: Aggregation = Aggregation.MAX
    categories: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        """Validate the role, range and category settings."""
        if self.role not in ROLES:
            raise ValueError(f"Unknown role {self.role!r}, expected one of {ROLES}")
        low, high = self.value_range
        if not low <= high:
            raise ValueError(f"Empty value range {self.value_range} for {self.name}")
        if (self.aggregation is Aggregation.MAJORITY) != bool(self.categories):
            raise ValueError(f"{self.name}: categories are required exactly for majority lanes")

    @property
    def name(self) -> str:
        """Column name ``<source>.<quantity>.<role>``."""
        return f"{self.source}.{self.quantity}.{self.role}"


@dataclass(frozen=True)
class BandSpec:
    """How a band was derived.

    Attributes:
        name: Column name, by default ``<lane>.on``.
        lane: Name of the lane the band was derived from, empty for multi-lane rules.
        threshold: Threshold applied to the lane, None for rule-based bands.
        origin: One of :data:`ORIGINS`.
        rule_id: Identifier of the rule in the decision tables.
    """

    name: str
    lane: str
    threshold: float | None
    origin: str
    rule_id: str

    def __post_init__(self) -> None:
        """Validate the origin."""
        if self.origin not in ORIGINS:
            raise ValueError(f"Unknown origin {self.origin!r}, expected one of {ORIGINS}")


def grid_times(duration_s: float, rate_hz: float = GRID_RATE_HZ) -> np.ndarray:
    """Times of the lane grid of a phase.

    Args:
        duration_s: Phase duration in seconds.
        rate_hz: Grid rate.

    Returns:
        ``k / rate_hz`` for ``k = 0 … floor(rate_hz * duration_s)``.
    """
    if duration_s < 0:
        raise ValueError(f"Negative duration {duration_s}")
    n = math.floor(rate_hz * duration_s + 1e-9) + 1
    return np.arange(n) / rate_hz


@dataclass
class Lane:
    """One quantity on the grid.

    Attributes:
        spec: Lane description.
        values: Float array on the grid; NaN where the lane does not apply.
    """

    spec: LaneSpec
    values: np.ndarray

    def __post_init__(self) -> None:
        """Validate shape and range."""
        self.values = np.asarray(self.values, dtype=float)
        if self.values.ndim != 1:
            raise ValueError(f"{self.spec.name}: values must be 1-D")
        finite = self.values[np.isfinite(self.values)]
        low, high = self.spec.value_range
        if finite.size and (finite.min() < low or finite.max() > high):
            raise ValueError(
                f"{self.spec.name}: values outside {self.spec.value_range} "
                f"(min {finite.min()}, max {finite.max()})"
            )


@dataclass
class Band:
    """Boolean presence on the grid.

    Attributes:
        spec: Band description.
        values: Boolean array on the grid.
    """

    spec: BandSpec
    values: np.ndarray

    def __post_init__(self) -> None:
        """Coerce to a 1-D boolean array."""
        self.values = np.asarray(self.values, dtype=bool)
        if self.values.ndim != 1:
            raise ValueError(f"{self.spec.name}: values must be 1-D")


@dataclass(frozen=True)
class Event:
    """A time interval with a label.

    Attributes:
        start_s: Phase-local start.
        end_s: Phase-local end (exclusive).
        role: One of :data:`ROLES`.
        label: Event label, e.g. ``cry``.
        confidence: Confidence in ``[0, 1]``.
        source: Producing model or stage.
        threshold: Threshold of the band it came from, None for rules.
        rule_id: Identifier of the rule in the decision tables.
    """

    start_s: float
    end_s: float
    role: str
    label: str
    confidence: float
    source: str
    threshold: float | None
    rule_id: str


def to_grid(
    times: np.ndarray,
    values: np.ndarray,
    duration_s: float,
    spec: LaneSpec,
    rate_hz: float = GRID_RATE_HZ,
) -> Lane:
    """Resample native-rate samples onto the lane grid.

    Bin ``k`` collects the samples with ``round(t * rate_hz) == k``. Samples outside the
    phase and NaN samples are ignored; empty bins are NaN.

    Args:
        times: Sample times in seconds, phase-local.
        values: Sample values; category indices for majority lanes.
        duration_s: Phase duration.
        spec: Lane description, its aggregation is applied.
        rate_hz: Grid rate.

    Returns:
        The lane.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.shape != values.shape or times.ndim != 1:
        raise ValueError("times and values must be 1-D arrays of the same length")
    n = len(grid_times(duration_s, rate_hz))
    index = np.floor(times * rate_hz + 0.5).astype(int)
    keep = (index >= 0) & (index < n) & np.isfinite(values)
    index, values = index[keep], values[keep]
    out = np.full(n, np.nan)
    if spec.aggregation is Aggregation.MAX:
        np.fmax.at(out, index, values)
    elif spec.aggregation is Aggregation.MEAN:
        sums, counts = np.zeros(n), np.zeros(n)
        np.add.at(sums, index, values)
        np.add.at(counts, index, 1)
        filled = counts > 0
        out[filled] = sums[filled] / counts[filled]
    else:
        codes = values.astype(int)
        counts = np.zeros((n, len(spec.categories)))
        np.add.at(counts, (index, codes), 1)
        filled = counts.sum(axis=1) > 0
        out[filled] = counts[filled].argmax(axis=1)
    return Lane(spec, out)


def threshold_band(
    lane: Lane, threshold: float, origin: str, rule_id: str, name: str | None = None
) -> Band:
    """Presence where a lane reaches a threshold.

    Args:
        lane: Source lane; NaN never counts as present.
        threshold: Inclusive threshold.
        origin: One of :data:`ORIGINS`.
        rule_id: Identifier of the rule in the decision tables.
        name: Column name, defaults to ``<lane>.on``.

    Returns:
        The band.
    """
    with np.errstate(invalid="ignore"):
        present = np.nan_to_num(lane.values, nan=-np.inf) >= threshold
    spec = BandSpec(name or f"{lane.spec.name}.on", lane.spec.name, threshold, origin, rule_id)
    return Band(spec, present)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of True values.

    Args:
        mask: Boolean array.

    Returns:
        ``(start, stop)`` index pairs, ``stop`` exclusive.
    """
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def band_to_events(
    band: Band,
    role: str,
    label: str,
    source: str,
    confidence: Lane | None = None,
    merge_gap_s: float = 0.0,
    min_duration_s: float = 0.0,
    rate_hz: float = GRID_RATE_HZ,
) -> list[Event]:
    """Turn a band into events.

    Runs separated by at most ``merge_gap_s`` are merged first, then runs shorter than
    ``min_duration_s`` are dropped. An event of a run covering bins ``a … b-1`` spans
    ``[a / rate, b / rate)``.

    Args:
        band: Presence band.
        role: Role of the events.
        label: Label of the events.
        source: Producing model or stage.
        confidence: Lane whose mean over the run is the confidence (clipped to [0, 1]);
            confidence 1 if None or all NaN.
        merge_gap_s: Largest gap that is bridged.
        min_duration_s: Shortest kept event.
        rate_hz: Grid rate.

    Returns:
        Events in time order.
    """
    if role not in ROLES:
        raise ValueError(f"Unknown role {role!r}, expected one of {ROLES}")
    merged: list[list[int]] = []
    for start, stop in runs(band.values):
        if merged and (start - merged[-1][1]) / rate_hz <= merge_gap_s + 1e-9:
            merged[-1][1] = stop
        else:
            merged.append([start, stop])
    events = []
    for start, stop in merged:
        if (stop - start) / rate_hz + 1e-9 < min_duration_s:
            continue
        score = 1.0
        if confidence is not None:
            part = confidence.values[start:stop]
            if np.isfinite(part).any():
                score = float(np.clip(np.nanmean(part), 0.0, 1.0))
        events.append(
            Event(
                start / rate_hz,
                stop / rate_hz,
                role,
                label,
                score,
                source,
                band.spec.threshold,
                band.spec.rule_id,
            )
        )
    return events


@dataclass(frozen=True)
class Legend:
    """Metadata of a lane table.

    Attributes:
        duration_s: Phase duration.
        rate_hz: Grid rate.
        lanes: Lane descriptions in column order.
        bands: Band descriptions in column order (after the lanes).
        notes: Free-form extra metadata (e.g. session, phase, config hash).
    """

    duration_s: float
    rate_hz: float
    lanes: tuple[LaneSpec, ...]
    bands: tuple[BandSpec, ...]
    notes: dict[str, str] = field(default_factory=dict)

    @property
    def columns(self) -> tuple[str, ...]:
        """CSV columns: ``time_s``, then lanes, then bands."""
        return ("time_s", *(s.name for s in self.lanes), *(s.name for s in self.bands))


@dataclass
class LaneTable:
    """All lanes and bands of one session phase.

    Attributes:
        duration_s: Phase duration.
        lanes: Lanes on the grid.
        bands: Bands on the grid.
        rate_hz: Grid rate.
        notes: Free-form extra metadata copied to the legend.
    """

    duration_s: float
    lanes: list[Lane] = field(default_factory=list)
    bands: list[Band] = field(default_factory=list)
    rate_hz: float = GRID_RATE_HZ
    notes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check lengths and unique names."""
        n = len(self.times)
        names = [lane.spec.name for lane in self.lanes] + [b.spec.name for b in self.bands]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate lane or band names: {names}")
        for item in [*self.lanes, *self.bands]:
            if len(item.values) != n:
                raise ValueError(f"{item.spec.name}: {len(item.values)} values, grid has {n}")

    @property
    def times(self) -> np.ndarray:
        """Grid times."""
        return grid_times(self.duration_s, self.rate_hz)

    @property
    def legend(self) -> Legend:
        """Metadata of the table."""
        return Legend(
            self.duration_s,
            self.rate_hz,
            tuple(lane.spec for lane in self.lanes),
            tuple(band.spec for band in self.bands),
            dict(self.notes),
        )

    def lane(self, name: str) -> Lane:
        """Look up a lane by column name.

        Args:
            name: Lane column name.

        Returns:
            The lane.

        Raises:
            KeyError: If there is no such lane.
        """
        for lane in self.lanes:
            if lane.spec.name == name:
                return lane
        raise KeyError(name)

    def band(self, name: str) -> Band:
        """Look up a band by column name.

        Args:
            name: Band column name.

        Returns:
            The band.

        Raises:
            KeyError: If there is no such band.
        """
        for band in self.bands:
            if band.spec.name == name:
                return band
        raise KeyError(name)


def _float_cell(value: float) -> str:
    return "" if math.isnan(value) else repr(float(value))


def _parse_float(text: str) -> float:
    return math.nan if text == "" else float(text)


def legend_to_dict(legend: Legend) -> dict:
    """Serialisable form of a legend.

    Args:
        legend: Legend to convert.

    Returns:
        JSON-compatible dictionary including :data:`LEGEND_VERSION`.
    """
    data = asdict(legend)
    data["version"] = LEGEND_VERSION
    return data


def legend_from_dict(data: dict) -> Legend:
    """Rebuild a legend from :func:`legend_to_dict` output.

    Args:
        data: Dictionary read from JSON.

    Returns:
        The legend.

    Raises:
        ValueError: If the version is not :data:`LEGEND_VERSION`.
    """
    if data.get("version") != LEGEND_VERSION:
        raise ValueError(f"Unsupported legend version {data.get('version')!r}")
    lanes = tuple(
        LaneSpec(
            source=s["source"],
            quantity=s["quantity"],
            role=s["role"],
            unit=s["unit"],
            value_range=(float(s["value_range"][0]), float(s["value_range"][1])),
            aggregation=Aggregation(s["aggregation"]),
            categories=tuple(s["categories"]),
            note=s["note"],
        )
        for s in data["lanes"]
    )
    bands = tuple(BandSpec(**s) for s in data["bands"])
    return Legend(
        float(data["duration_s"]), float(data["rate_hz"]), lanes, bands, dict(data["notes"])
    )


def write_lanes(table: LaneTable, csv_path: Path, legend_path: Path) -> None:
    """Write a lane table as a wide CSV plus its legend JSON, atomically.

    NaN lane values are empty cells; band values are 0 / 1.

    Args:
        table: Table to write.
        csv_path: CSV output.
        legend_path: Legend JSON output.
    """
    legend = table.legend
    columns = [table.times, *(lane.values for lane in table.lanes)]
    flags = [band.values for band in table.bands]

    def write_csv(tmp: Path) -> None:
        with tmp.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(legend.columns)
            for i in range(len(table.times)):
                writer.writerow([_float_cell(c[i]) for c in columns] + [int(f[i]) for f in flags])

    write_atomic(csv_path, write_csv)
    text = json.dumps(legend_to_dict(legend), indent=2, ensure_ascii=False) + "\n"
    write_atomic(legend_path, lambda tmp: tmp.write_text(text))


def read_lanes(csv_path: Path, legend_path: Path) -> LaneTable:
    """Read a lane table written by :func:`write_lanes`.

    Args:
        csv_path: CSV file.
        legend_path: Legend JSON file.

    Returns:
        The table.

    Raises:
        ValueError: If the CSV header does not match the legend.
    """
    legend = legend_from_dict(json.loads(legend_path.read_text()))
    with csv_path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = tuple(next(reader))
        if header != legend.columns:
            raise ValueError(f"{csv_path}: header does not match the legend")
        rows = list(reader)
    n_lanes = len(legend.lanes)
    cells = list(zip(*rows, strict=True)) if rows else [()] * len(header)
    lanes = [
        Lane(spec, np.array([_parse_float(c) for c in cells[1 + i]], dtype=float))
        for i, spec in enumerate(legend.lanes)
    ]
    bands = [
        Band(spec, np.array([c == "1" for c in cells[1 + n_lanes + i]], dtype=bool))
        for i, spec in enumerate(legend.bands)
    ]
    return LaneTable(legend.duration_s, lanes, bands, legend.rate_hz, dict(legend.notes))


def write_events(events: list[Event], path: Path) -> None:
    """Write events as CSV, atomically.

    Args:
        events: Events; may be empty.
        path: Output file.
    """

    def write(tmp: Path) -> None:
        with tmp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=EVENT_COLUMNS)
            writer.writeheader()
            for event in events:
                row = asdict(event)
                row["threshold"] = "" if event.threshold is None else repr(event.threshold)
                for key in ("start_s", "end_s", "confidence"):
                    row[key] = repr(float(row[key]))
                writer.writerow(row)

    write_atomic(path, write)


def read_events(path: Path) -> list[Event]:
    """Read events written by :func:`write_events`.

    Args:
        path: Event CSV.

    Returns:
        The events.

    Raises:
        ValueError: If the header does not match :data:`EVENT_COLUMNS`.
    """
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EVENT_COLUMNS:
            raise ValueError(f"Unexpected event columns in {path}: {reader.fieldnames}")
        return [
            Event(
                start_s=float(r["start_s"]),
                end_s=float(r["end_s"]),
                role=r["role"],
                label=r["label"],
                confidence=float(r["confidence"]),
                source=r["source"],
                threshold=None if r["threshold"] == "" else float(r["threshold"]),
                rule_id=r["rule_id"],
            )
            for r in reader
        ]
