"""Minimal Praat TextGrid reading and writing (interval tiers only).

Montreal Forced Aligner writes one TextGrid per utterance with ``words`` and ``phones``
interval tiers. This module parses the long (``ooTextFile``) format MFA writes and writes
the merged phase-level TextGrid that Praat and Annie-side tools can open. Standard library
only, so model workers can use it.
"""

import re
from dataclasses import dataclass

EMPTY_LABELS: frozenset[str] = frozenset({"", "sil", "sp", "<eps>"})
"""Interval labels that mark silence rather than a word or phone."""


@dataclass(frozen=True)
class Interval:
    """One labelled interval.

    Attributes:
        start_s: Start time.
        end_s: End time.
        label: Text of the interval.
    """

    start_s: float
    end_s: float
    label: str


_TIER = re.compile(r'name\s*=\s*"(?P<name>[^"]*)"(?P<body>.*?)(?=item\s*\[\d+\]:|\Z)', re.S)
_INTERVAL = re.compile(
    r'xmin\s*=\s*(?P<a>[-\d.eE+]+)\s*xmax\s*=\s*(?P<b>[-\d.eE+]+)\s*text\s*=\s*"(?P<t>(?:[^"]|"")*)"',
    re.S,
)


def parse_textgrid(text: str) -> dict[str, list[Interval]]:
    """Parse the interval tiers of a long-format TextGrid.

    Args:
        text: TextGrid file content.

    Returns:
        Tier name mapped to its intervals (silences included), in file order.
    """
    tiers: dict[str, list[Interval]] = {}
    body = text.split("item []:", 1)[-1]
    for tier in _TIER.finditer(body):
        tiers[tier.group("name")] = [
            Interval(float(m.group("a")), float(m.group("b")), m.group("t").replace('""', '"'))
            for m in _INTERVAL.finditer(tier.group("body"))
        ]
    return tiers


def shift(intervals: list[Interval], offset_s: float) -> list[Interval]:
    """Move intervals by an offset (utterance-local to phase-local time).

    Args:
        intervals: Intervals.
        offset_s: Seconds added.

    Returns:
        Shifted intervals.
    """
    return [Interval(i.start_s + offset_s, i.end_s + offset_s, i.label) for i in intervals]


def write_textgrid(tiers: dict[str, list[Interval]], duration_s: float) -> str:
    """Write interval tiers as a long-format TextGrid, filling gaps with empty intervals.

    Args:
        tiers: Tier name mapped to non-overlapping intervals.
        duration_s: Total duration.

    Returns:
        TextGrid content.
    """
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        f"xmax = {duration_s}",
        "tiers? <exists>",
        f"size = {len(tiers)}",
        "item []:",
    ]
    for number, (name, intervals) in enumerate(tiers.items(), start=1):
        filled: list[Interval] = []
        cursor = 0.0
        for item in sorted(intervals, key=lambda i: i.start_s):
            if item.start_s > cursor:
                filled.append(Interval(cursor, item.start_s, ""))
            filled.append(item)
            cursor = max(cursor, item.end_s)
        if cursor < duration_s:
            filled.append(Interval(cursor, duration_s, ""))
        lines += [
            f"    item [{number}]:",
            '        class = "IntervalTier"',
            f'        name = "{name}"',
            "        xmin = 0",
            f"        xmax = {duration_s}",
            f"        intervals: size = {len(filled)}",
        ]
        for index, item in enumerate(filled, start=1):
            label = item.label.replace('"', '""')
            lines += [
                f"        intervals [{index}]:",
                f"            xmin = {item.start_s}",
                f"            xmax = {item.end_s}",
                f'            text = "{label}"',
            ]
    return "\n".join(lines) + "\n"
