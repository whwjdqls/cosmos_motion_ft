"""Focused tests for model-only training continuation semantics."""
from __future__ import annotations

import unittest

from bs_train import continuation_lr, event_step_after_update


class ContinuationLearningRateTest(unittest.TestCase):
    def test_restart_schedule_uses_local_step(self):
        kwargs = {
            "start_step": 200_000,
            "total_steps": 500_000,
            "base_lr": 5e-5,
            "warmup_steps": 1_000,
        }
        self.assertAlmostEqual(continuation_lr(200_000, **kwargs), 5e-8)
        self.assertAlmostEqual(continuation_lr(200_999, **kwargs), 5e-5)
        self.assertAlmostEqual(continuation_lr(201_000, **kwargs), 5e-5)
        self.assertLess(continuation_lr(499_999, **kwargs), 1e-12)

    def test_rejects_steps_outside_invocation(self):
        kwargs = {
            "start_step": 10,
            "total_steps": 20,
            "base_lr": 1e-4,
            "warmup_steps": 2,
        }
        with self.assertRaises(ValueError):
            continuation_lr(9, **kwargs)
        with self.assertRaises(ValueError):
            continuation_lr(20, **kwargs)


class StepIndexingTest(unittest.TestCase):
    def test_completed_update_labels(self):
        self.assertEqual(event_step_after_update(1, "completed_updates"), 1)
        self.assertEqual(event_step_after_update(10_000, "completed_updates"), 10_000)

    def test_historical_preincrement_labels(self):
        self.assertEqual(event_step_after_update(1, "historical_preincrement"), 0)
        self.assertEqual(event_step_after_update(10_001, "historical_preincrement"), 10_000)


if __name__ == "__main__":
    unittest.main()
