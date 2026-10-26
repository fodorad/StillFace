import unittest

import numpy as np

from stillface.features.gaze.estimator import GazeEstimate, GazeEstimator
from tests.features.gaze.fakes import FakeGazeEstimator, box


class TestGazeEstimatorContract(unittest.TestCase):
    def test_fake_satisfies_the_protocol_and_returns_documented_shapes(self) -> None:
        estimator: GazeEstimator = FakeGazeEstimator({0: (0.2, 0.5), 1: (0.8, 0.5)})
        frames = np.random.default_rng(0).integers(0, 255, (3, 48, 64, 3), dtype=np.uint8)
        boxes = [[box(0.8), box(0.3)]] * 3

        estimate = estimator.estimate(frames, boxes)

        self.assertEqual(estimate.heatmap.shape, (3, 2, 64, 64))
        self.assertEqual(estimate.inout.shape, (3, 2))
        self.assertGreaterEqual(float(estimate.heatmap.min()), 0.0)
        self.assertLessEqual(float(estimate.heatmap.max()), 1.0)

    def test_estimate_without_inout_scores(self) -> None:
        estimate = GazeEstimate(np.zeros((1, 1, 64, 64), np.float32), None)
        self.assertIsNone(estimate.inout)


if __name__ == "__main__":
    unittest.main()
