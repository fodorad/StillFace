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


if __name__ == "__main__":
    unittest.main()
