import unittest

import numpy as np

from stillface.features.gaze.classes import (
    NK,
    GazeDecision,
    classify,
    gaze_point,
    mass_in_box,
)
from stillface.features.gaze.config import GazeConfig, Thresholds
from stillface.features.gaze.heads import HeadBox
from tests.features.gaze.fakes import box, peak_heatmap

CFG = GazeConfig(partner_zones={"baby": (0.0, 0.0, 0.4, 0.4), "mother": (0.6, 0.0, 1.0, 0.4)})
PARTNER_BOX = box(0.2, 0.5, width=0.1)  # x 0.15-0.25, y 0.44-0.56


class TestMassInBox(unittest.TestCase):
    def test_full_image_holds_all_mass(self) -> None:
        heatmap = np.random.default_rng(0).random((64, 64)).astype(np.float32)
        self.assertAlmostEqual(mass_in_box(heatmap, HeadBox(0, 0, 1, 1)), 1.0, places=5)

    def test_mass_grows_with_the_box(self) -> None:
        heatmap = np.random.default_rng(1).random((64, 64)).astype(np.float32)
        small = mass_in_box(heatmap, HeadBox(0.4, 0.4, 0.5, 0.5))
        large = mass_in_box(heatmap, HeadBox(0.3, 0.3, 0.6, 0.6))
        self.assertLess(small, large)

    def test_peak_inside_and_outside(self) -> None:
        heatmap = peak_heatmap(0.2, 0.5)
        self.assertGreater(mass_in_box(heatmap, PARTNER_BOX), 0.9)
        self.assertLess(mass_in_box(heatmap, HeadBox(0.6, 0.6, 0.9, 0.9)), 0.01)

    def test_empty_heatmap_has_no_mass(self) -> None:
        self.assertEqual(mass_in_box(np.zeros((64, 64), np.float32), HeadBox(0, 0, 1, 1)), 0.0)

    def test_partial_cells_are_weighted(self) -> None:
        heatmap = np.ones((64, 64), np.float32)
        self.assertAlmostEqual(mass_in_box(heatmap, HeadBox(0.0, 0.0, 0.5, 0.25)), 0.125)
        self.assertAlmostEqual(
            mass_in_box(heatmap, HeadBox(0.0, 0.0, 0.5 + 1 / 128, 1.0)), 0.5 + 1 / 128
        )


class TestGazePoint(unittest.TestCase):
    def test_argmax_in_normalized_coordinates(self) -> None:
        x, y = gaze_point(peak_heatmap(0.25, 0.75))
        self.assertAlmostEqual(x, 0.25, delta=1 / 64)
        self.assertAlmostEqual(y, 0.75, delta=1 / 64)


class TestSideCameras(unittest.TestCase):
    def decide(self, person, target, inout=0.9, partner=PARTNER_BOX, camera="window"):
        return classify(person, camera, peak_heatmap(*target), inout, partner, CFG)

    def test_mother_looking_at_the_baby(self) -> None:
        decision = self.decide("mother", (0.2, 0.5))
        self.assertEqual(decision.gaze_class, "BABARA")
        self.assertGreater(decision.partner_mass, 0.9)
        self.assertAlmostEqual(decision.gaze_x, 0.2, delta=1 / 64)

    def test_baby_looking_at_the_mother(self) -> None:
        self.assertEqual(self.decide("baby", (0.2, 0.5)).gaze_class, "ANYARA")

    def test_gaze_elsewhere_in_the_scene(self) -> None:
        self.assertEqual(self.decide("baby", (0.8, 0.8)).gaze_class, "TARGY_KORNYEZET")
        self.assertEqual(self.decide("mother", (0.8, 0.8)).gaze_class, "MASHOVA")

    def test_out_of_frame_gaze(self) -> None:
        self.assertEqual(self.decide("baby", (0.2, 0.5), inout=0.1).gaze_class, "ELFORDUL")
        self.assertEqual(self.decide("mother", (0.2, 0.5), inout=0.1).gaze_class, "MASHOVA")

    def test_margin_catches_gaze_just_beside_the_head(self) -> None:
        self.assertEqual(self.decide("mother", (0.28, 0.5)).gaze_class, "BABARA")
        strict = GazeConfig(thresholds=Thresholds(box_margin=0.0))
        decision = classify("mother", "window", peak_heatmap(0.28, 0.5), 0.9, PARTNER_BOX, strict)
        self.assertEqual(decision.gaze_class, "MASHOVA")

    def test_threshold_comes_from_the_config(self) -> None:
        heatmap = peak_heatmap(0.2, 0.5)
        strict = GazeConfig(thresholds=Thresholds(partner_mass=1.01))
        self.assertEqual(
            classify("mother", "window", heatmap, 0.9, PARTNER_BOX, strict).gaze_class, "MASHOVA"
        )

    def test_partner_head_not_found_is_not_codable(self) -> None:
        decision = self.decide("mother", (0.2, 0.5), partner=None)
        self.assertEqual((decision.gaze_class, decision.reason), (NK, "partner_not_found"))
        self.assertIsNone(decision.partner_mass)


class TestCloseCameras(unittest.TestCase):
    def test_mother_camera_zone(self) -> None:
        decision = classify("mother", "mother", peak_heatmap(0.8, 0.2), 0.3, None, CFG)
        self.assertEqual(decision.gaze_class, "BABARA")

    def test_baby_camera_zone(self) -> None:
        decision = classify("baby", "baby", peak_heatmap(0.2, 0.2), 0.3, None, CFG)
        self.assertEqual(decision.gaze_class, "ANYARA")

    def test_outside_zone_and_out_of_frame(self) -> None:
        self.assertEqual(
            classify("baby", "baby", peak_heatmap(0.8, 0.8), 0.1, None, CFG).gaze_class, "ELFORDUL"
        )
        self.assertEqual(
            classify("baby", "baby", peak_heatmap(0.8, 0.8), 0.9, None, CFG).gaze_class,
            "TARGY_KORNYEZET",
        )

    def test_uncalibrated_camera_is_not_codable(self) -> None:
        decision = classify("baby", "baby", peak_heatmap(0.2, 0.2), 0.9, None, GazeConfig())
        self.assertEqual((decision.gaze_class, decision.reason), (NK, "no_partner_zone"))


class TestDecision(unittest.TestCase):
    def test_not_codable_has_no_gaze_values(self) -> None:
        decision = GazeDecision.not_codable("occluded")
        self.assertEqual((decision.gaze_class, decision.reason), (NK, "occluded"))
        self.assertIsNone(decision.gaze_x)
        self.assertIsNone(decision.inout)

    def test_unknown_person_or_camera(self) -> None:
        with self.assertRaises(ValueError):
            classify("father", "window", peak_heatmap(0.5, 0.5), 0.9, PARTNER_BOX, CFG)
        with self.assertRaises(ValueError):
            classify("baby", "ceiling", peak_heatmap(0.5, 0.5), 0.9, PARTNER_BOX, CFG)


if __name__ == "__main__":
    unittest.main()
