"""Focused tests for BONES normalization provenance and checkpoint resolution."""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from bs_normalization import load_normalization, resolve_checkpoint_normalization


class NormalizationResolutionTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.mean = os.path.join(self.tempdir.name, "mean.npy")
        self.std = os.path.join(self.tempdir.name, "std.npy")
        self.other_mean = os.path.join(self.tempdir.name, "other_mean.npy")
        self.other_std = os.path.join(self.tempdir.name, "other_std.npy")
        np.save(self.mean, np.zeros(283, dtype=np.float32))
        np.save(self.std, np.ones(283, dtype=np.float32))
        np.save(self.other_mean, np.ones(283, dtype=np.float32))
        np.save(self.other_std, np.full(283, 2.0, dtype=np.float32))

    def tearDown(self):
        self.tempdir.cleanup()

    def test_resolves_and_verifies_new_checkpoint_metadata(self):
        _, _, metadata = load_normalization(self.mean, self.std, tag="test_stats")
        mean, std, resolved = resolve_checkpoint_normalization(
            {"normalization": metadata}
        )
        np.testing.assert_array_equal(mean, np.zeros(283, dtype=np.float32))
        np.testing.assert_array_equal(std, np.ones(283, dtype=np.float32))
        self.assertEqual(resolved["tag"], "test_stats")
        self.assertTrue(resolved["checkpoint_match"])

    def test_legacy_checkpoint_args_are_content_verified(self):
        checkpoint = {"args": {"mean": self.mean, "std": self.std}}
        _, _, resolved = resolve_checkpoint_normalization(checkpoint)
        self.assertTrue(resolved["tag"].startswith("custom_"))
        self.assertTrue(resolved["checkpoint_match"])

    def test_rejects_silent_override_and_records_explicit_override(self):
        _, _, metadata = load_normalization(self.mean, self.std, tag="test_stats")
        checkpoint = {"normalization": metadata}
        with self.assertRaisesRegex(ValueError, "normalization mismatch"):
            resolve_checkpoint_normalization(
                checkpoint,
                mean_override=self.other_mean,
                std_override=self.other_std,
            )
        _, _, resolved = resolve_checkpoint_normalization(
            checkpoint,
            mean_override=self.other_mean,
            std_override=self.other_std,
            allow_override=True,
        )
        self.assertFalse(resolved["checkpoint_match"])
        self.assertTrue(resolved["override_allowed"])

    def test_requires_mean_and_std_overrides_as_a_pair(self):
        with self.assertRaisesRegex(ValueError, "supplied together"):
            resolve_checkpoint_normalization({}, mean_override=self.mean)


if __name__ == "__main__":
    unittest.main()
