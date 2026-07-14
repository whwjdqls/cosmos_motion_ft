"""Focused tests for aggregate BONES shape-awareness metrics."""
from __future__ import annotations

import unittest

import numpy as np

from bs_shape_metrics import (
    counterfactual_shape_response,
    farthest_shape_indices,
    population_shape_tracking,
)


class ShapeMetricsTest(unittest.TestCase):
    def setUp(self):
        self.target = np.array(
            [
                [0.10, 0.20, 0.30],
                [0.12, 0.18, 0.33],
                [0.08, 0.24, 0.27],
            ],
            dtype=np.float32,
        )

    def test_population_tracking_is_one_for_perfect_shape(self):
        metrics = population_shape_tracking(self.target, self.target)
        self.assertAlmostEqual(metrics["actor_centered_correlation"], 1.0)
        self.assertAlmostEqual(metrics["actor_centered_response_slope"], 1.0)
        self.assertAlmostEqual(metrics["actor_centered_variance_ratio"], 1.0)
        self.assertAlmostEqual(metrics["actor_centered_mae_cm"], 0.0)

    def test_population_tracking_detects_average_skeleton_collapse(self):
        collapsed = np.repeat(self.target.mean(axis=0, keepdims=True), len(self.target), axis=0)
        metrics = population_shape_tracking(collapsed, self.target)
        self.assertAlmostEqual(metrics["actor_centered_correlation"], 0.0)
        self.assertAlmostEqual(metrics["actor_centered_response_slope"], 0.0)
        self.assertAlmostEqual(metrics["actor_centered_variance_ratio"], 0.0)
        self.assertGreater(metrics["actor_centered_mae_cm"], 0.0)

    def test_counterfactual_response_distinguishes_following_from_ignoring(self):
        swapped = self.target[[2, 0, 1]]
        following = counterfactual_shape_response(
            self.target,
            swapped,
            self.target,
            swapped,
        )
        ignored = counterfactual_shape_response(
            self.target,
            self.target,
            self.target,
            swapped,
        )
        self.assertAlmostEqual(following["delta_cosine"], 1.0)
        self.assertAlmostEqual(following["delta_response_slope"], 1.0)
        self.assertAlmostEqual(following["delta_magnitude_ratio"], 1.0)
        self.assertGreater(following["counterfactual_target_advantage_cm"], 0.0)
        self.assertAlmostEqual(ignored["delta_response_slope"], 0.0)
        self.assertAlmostEqual(ignored["delta_magnitude_ratio"], 0.0)
        self.assertLess(ignored["counterfactual_target_advantage_cm"], 0.0)

    def test_farthest_pairing_never_selects_self(self):
        indices = farthest_shape_indices(self.target)
        np.testing.assert_array_equal(indices, np.array([2, 2, 1]))
        self.assertTrue(np.all(indices != np.arange(len(self.target))))

    def test_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            population_shape_tracking(self.target[:2], self.target)


if __name__ == "__main__":
    unittest.main()
