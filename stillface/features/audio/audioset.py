"""AudioSet classes of the AST model, with priority tiers for the still-face paradigm.

AST scores all 527 AudioSet classes in every window; nothing is filtered before or during
inference. The tiers only say which classes the analysis looks at first:

- tier 1, target vocalisations: baby cry / fuss / laugh / babble, the mother's singing,
  humming and laughter, speech by sex and age;
- tier 2, context: the phase-change gong, bodily sounds, touch and play sounds, and the
  background sounds that can mask the dyad;
- tier 0: everything else.

The table is a CSV shipped with the package (``index, label, tier, group, reason``), in the
model's output order, so it can be filtered in any spreadsheet as well.
"""

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from stillface.features.audio.config import AudioConfig

AST_CLASSES_PATH: Path = Path(__file__).with_name("ast_classes.csv")
"""AudioSet class table shipped with the package."""

TIERS: tuple[int, ...] = (0, 1, 2)
"""Priority tiers: 1 target vocalisation, 2 context, 0 not prioritised."""

COLUMNS: tuple[str, ...] = ("index", "label", "tier", "group", "reason")
"""Columns of the class table, in file order."""


@dataclass(frozen=True)
class AudioSetClass:
    """One AudioSet class.

    Attributes:
        index: Output index in the AST model.
        label: AudioSet label, exactly as the model names it.
        tier: Priority tier, one of :data:`TIERS`.
        group: Analysis group of a prioritised class (e.g. ``baby_vocal``), empty for tier 0.
        reason: Why the class is prioritised, empty for tier 0.
    """

    index: int
    label: str
    tier: int
    group: str
    reason: str


def load_classes(path: Path = AST_CLASSES_PATH) -> tuple[AudioSetClass, ...]:
    """Read the class table.

    Args:
        path: CSV with :data:`COLUMNS`.

    Returns:
        The classes in file order.

    Raises:
        ValueError: If the columns differ or a tier is unknown.
    """
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"Unexpected columns in {path.name}: {reader.fieldnames}")
        classes = []
        for row in reader:
            tier = int(row["tier"])
            if tier not in TIERS:
                raise ValueError(f"Unknown tier {tier} for {row['label']!r}")
            classes.append(
                AudioSetClass(int(row["index"]), row["label"], tier, row["group"], row["reason"])
            )
    return tuple(classes)


def priority_labels(max_tier: int, classes: Sequence[AudioSetClass]) -> tuple[str, ...]:
    """Labels of the prioritised classes up to a tier.

    Args:
        max_tier: Highest tier included (1 = targets only, 2 = targets and context).
        classes: Class table.

    Returns:
        Labels with ``1 <= tier <= max_tier``, in model order.
    """
    return tuple(c.label for c in classes if 1 <= c.tier <= max_tier)


def config_ast_labels(config: AudioConfig) -> set[str]:
    """Every AudioSet label the fusion rules of a config name.

    Args:
        config: Audio configuration.

    Returns:
        Labels of the gong and context classes and of the baby and mother type rules.
    """
    fc = config.fusion
    labels = {*fc.gong.classes, *fc.context.classes}
    for rule in [*fc.baby_types.values(), *fc.mother_types.values()]:
        labels.update(rule.ast)
    return labels


def validate_labels(labels: Iterable[str], classes: Sequence[AudioSetClass]) -> None:
    """Check that labels exist in AudioSet (a typo would silently never fire).

    Args:
        labels: Labels to check.
        classes: Class table.

    Raises:
        ValueError: Naming every unknown label.
    """
    known = {c.label for c in classes}
    unknown = sorted(set(labels) - known)
    if unknown:
        raise ValueError(f"Unknown AudioSet labels: {unknown}")
