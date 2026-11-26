"""Tests of the AudioSet class table."""

import tempfile
import unittest
from pathlib import Path

from stillface.features.audio.audioset import (
    TIERS,
    config_ast_labels,
    load_classes,
    priority_labels,
    validate_labels,
)
from stillface.features.audio.config import load_audio_config


class TestAudioSetTable(unittest.TestCase):
    def setUp(self):
        self.classes = load_classes()

    def test_all_527_classes_in_model_order(self):
        self.assertEqual(len(self.classes), 527)
        self.assertEqual([c.index for c in self.classes], list(range(527)))
        self.assertEqual(len({c.label for c in self.classes}), 527)
        self.assertEqual(self.classes[0].label, "Speech")
        self.assertEqual(self.classes[23].label, "Baby cry, infant cry")
        self.assertEqual(self.classes[-1].label, "Field recording")

    def test_tiers_groups_and_reasons(self):
        by_label = {c.label: c for c in self.classes}
        self.assertEqual(
            (by_label["Baby cry, infant cry"].tier, by_label["Baby cry, infant cry"].group),
            (1, "baby_vocal"),
        )
        self.assertEqual(by_label["Gong"].tier, 2)
        self.assertEqual(by_label["Dog"].tier, 0)
        for c in self.classes:
            self.assertIn(c.tier, TIERS)
            self.assertEqual(bool(c.group), c.tier > 0, c.label)
            self.assertEqual(bool(c.reason), c.tier > 0, c.label)

    def test_priority_labels_nest_by_tier(self):
        tier1 = priority_labels(1, self.classes)
        tier2 = priority_labels(2, self.classes)
        self.assertTrue(set(tier1) < set(tier2))
        self.assertIn("Babbling", tier1)
        self.assertNotIn("Gong", tier1)
        self.assertIn("Gong", tier2)
        self.assertEqual(priority_labels(0, self.classes), ())
        self.assertEqual(list(tier2), sorted(tier2, key=[c.label for c in self.classes].index))

    def test_every_class_the_config_names_exists_and_has_priority(self):
        named = config_ast_labels(load_audio_config())
        self.assertIn("Gong", named)
        self.assertIn("Lullaby", named)
        validate_labels(named, self.classes)
        self.assertTrue(named <= set(priority_labels(2, self.classes)))

    def test_docs_table_lists_exactly_the_prioritised_classes(self):
        page = Path(__file__).parents[3] / "docs" / "audio_classes.md"
        rows = [line.split(" | ") for line in page.read_text().splitlines()]
        listed = [(int(r[0].strip("| ")), r[3]) for r in rows if r[0] in ("| 1", "| 2")]
        expected = [(c.tier, c.label) for t in (1, 2) for c in self.classes if c.tier == t]
        self.assertEqual(listed, expected)

    def test_unknown_label_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Baby scream"):
            validate_labels({"Baby scream", "Speech"}, self.classes)

    def test_bad_table_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.csv"
            path.write_text("index,label,tier,group,reason\n0,Speech,3,x,y\n")
            with self.assertRaisesRegex(ValueError, "tier"):
                load_classes(path)
            path.write_text("index,label\n0,Speech\n")
            with self.assertRaisesRegex(ValueError, "columns"):
                load_classes(path)


if __name__ == "__main__":
    unittest.main()
