from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

try:
    from .estimate_camera_head_absolute_lever import (
        _clean_mask,
        _fit_one,
        _metric_one,
        _plot_qualitative_lever_comparison,
        _relative_translation,
        geometric_median,
    )
except ImportError:  # pragma: no cover - direct test-file execution
    from estimate_camera_head_absolute_lever import (
        _clean_mask,
        _fit_one,
        _metric_one,
        _plot_qualitative_lever_comparison,
        _relative_translation,
        geometric_median,
    )


class CameraHeadAbsoluteLeverTest(unittest.TestCase):
    @staticmethod
    def _synthetic_sequence() -> dict[str, np.ndarray]:
        frames = 4
        return {
            "head_rotation": np.broadcast_to(np.eye(3), (frames, 3, 3)).copy(),
            "head_position": np.zeros((frames, 3), dtype=np.float64),
            "camera_rotation": np.broadcast_to(np.eye(3), (frames, 3, 3)).copy(),
            "camera_position": np.broadcast_to(
                np.asarray([0.0, 0.1, 0.2]), (frames, 3)
            ).copy(),
            "camera_action": np.zeros((frames - 1, 9), dtype=np.float64),
            "delta_off_axis_max": np.asarray(0.0),
        }

    def test_clean_mask_unions_overlapping_windows(self) -> None:
        mask = _clean_mask(10, [(1, 5), (3, 7), (-2, 2), (9, 20)])
        np.testing.assert_array_equal(
            mask,
            np.asarray([True, True, True, True, True, True, True, False, False, True]),
        )

    def test_geometric_median_is_robust_to_one_large_outlier(self) -> None:
        points = np.asarray(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [-0.01, 0.0, 0.0], [10.0, 0.0, 0.0]]
        )
        estimate, metadata = geometric_median(points)
        self.assertTrue(metadata["converged"])
        self.assertLess(abs(float(estimate[0])), 0.011)
        self.assertLess(np.linalg.norm(estimate[1:]), 1e-12)

    def test_corrected_head_offsets_recover_known_lever(self) -> None:
        theta = np.linspace(-1.0, 1.0, 101)
        cosine, sine = np.cos(theta), np.sin(theta)
        rotation = np.zeros((len(theta), 3, 3), dtype=np.float64)
        rotation[:, 0, 0] = cosine
        rotation[:, 0, 2] = sine
        rotation[:, 1, 1] = 1.0
        rotation[:, 2, 0] = -sine
        rotation[:, 2, 2] = cosine
        head_position = np.stack((theta, theta * 0.2, theta * -0.3), axis=-1)
        expected = np.asarray([-0.015, 0.060, 0.125])
        camera_position = head_position + np.einsum(
            "tij,j->ti", rotation, expected
        )
        offsets = np.einsum(
            "tji,tj->ti", rotation, camera_position - head_position
        )
        estimate, _ = geometric_median(offsets)
        np.testing.assert_allclose(estimate, expected, atol=1e-12, rtol=0.0)

    def test_relative_translation_matches_rigid_pose_definition(self) -> None:
        rotation = np.broadcast_to(np.eye(3), (4, 3, 3)).copy()
        position = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [1.5, 1.0, 4.0], [1.5, 1.0, 4.0]]
        )
        np.testing.assert_allclose(
            _relative_translation(rotation, position), np.diff(position, axis=0)
        )

    def test_fit_and_metric_worker_payload_contracts(self) -> None:
        module = sys.modules[_fit_one.__module__]
        lever = [0.0, 0.1, 0.2]
        with mock.patch.object(
            module, "_load_sequence", return_value=self._synthetic_sequence()
        ):
            fit = _fit_one(
                (
                    "S01/synthetic",
                    "train",
                    [(0, 4)],
                    "/corrected",
                    "/camera",
                    np.eye(3).tolist(),
                    1,
                )
            )
            self.assertEqual(fit["clean_frames"], 4)
            metric = _metric_one(
                (
                    "S01/synthetic",
                    "test",
                    [(0, 4)],
                    "/corrected",
                    "/original",
                    "/camera",
                    np.eye(3).tolist(),
                    {"historical_relative_lever": lever},
                    lever,
                    lever,
                    1,
                )
            )
        self.assertIn(
            "original_representation_historical_lever", metric["metrics"]
        )
        self.assertEqual(
            metric["metrics"]["historical_relative_lever"]["all"][
                "absolute_translation_m"
            ]["maximum"],
            0.0,
        )

    def test_qualitative_plot_smoke(self) -> None:
        module = sys.modules[_fit_one.__module__]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "gallery_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "category": "synthetic",
                                "uuid": "S01/synthetic",
                                "start": 0,
                                "frames": 4,
                            }
                        ],
                        "artifacts": {},
                    }
                )
            )
            with mock.patch.object(
                module, "_load_sequence", return_value=self._synthetic_sequence()
            ):
                result = _plot_qualitative_lever_comparison(
                    root,
                    manifest,
                    root / "original",
                    root / "corrected",
                    root / "camera",
                    np.asarray([0.0, 0.1, 0.2]),
                    np.asarray([0.0, 0.1, 0.2]),
                )
            self.assertIsNotNone(result)
            self.assertEqual(result["cases"], 1)
            self.assertTrue(Path(result["trajectory_comparison"]).is_file())


if __name__ == "__main__":
    unittest.main()
