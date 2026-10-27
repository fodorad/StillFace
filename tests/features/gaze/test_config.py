import unittest

from stillface.features.gaze.config import DEFAULT_CONFIG_PATH, GazeConfig, load_gaze_config


class TestGazeConfig(unittest.TestCase):
    def test_packaged_default_matches_the_dataclass_defaults(self) -> None:
        self.assertTrue(DEFAULT_CONFIG_PATH.exists())
        self.assertEqual(load_gaze_config(), GazeConfig())

    def test_native_side_camera_rate_and_priors(self) -> None:
        cfg = load_gaze_config()
        self.assertEqual(cfg.sample_rate_hz, 30.0)
        self.assertEqual((cfg.side_priors.window, cfg.side_priors.door), ("right", "left"))
        self.assertEqual(cfg.model_variant, "vitl14_inout_childplay")

    def test_seating_side_per_camera_and_phase(self) -> None:
        priors = load_gaze_config().side_priors
        self.assertEqual(priors.side("window", "play"), "right")
        self.assertEqual(priors.side("window", "baseline"), "left")
        self.assertEqual(priors.side("door", "reunion"), "left")
        self.assertEqual(priors.side("door", "baseline"), "left")


if __name__ == "__main__":
    unittest.main()
