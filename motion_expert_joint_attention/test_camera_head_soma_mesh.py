#!/usr/bin/env python3
"""Focused contracts for the standard-SOMA camhead_v1 visualization path."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from visualize_camera_head_aligned_soma_mesh import (  # noqa: E402
    DEFAULT_SOMA_ASSETS,
    SOMA30_NAMES,
    load_standard_soma_skin,
    skin_soma30_motion,
)


def _yaw(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)),
        dtype=np.float64,
    )


class StandardSomaMeshContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.skin = load_standard_soma_skin(DEFAULT_SOMA_ASSETS, cluster_m=0.012)

    def test_compacted_skin_has_normalized_weights_and_head_surface(self) -> None:
        skin = self.skin
        self.assertEqual(skin.source_vertex_count, 18056)
        self.assertEqual(skin.source_face_count, 36108)
        self.assertLess(len(skin.bind_vertices), skin.source_vertex_count)
        self.assertLess(len(skin.faces), skin.source_face_count)
        np.testing.assert_allclose(skin.lbs_weights.sum(axis=1), 1.0, atol=1e-12)
        self.assertGreater(int((skin.head_influence > 0.0).sum()), 500)
        self.assertGreater(len(skin.head_face_indices), 1000)

    def test_head_rotation_moves_only_head_weighted_surface(self) -> None:
        frames = 3
        original = np.broadcast_to(np.eye(3), (frames, 30, 3, 3)).copy()
        aligned = original.copy()
        aligned[:, 6] = np.stack((_yaw(10.0), _yaw(25.0), _yaw(40.0)))
        roots = np.asarray(((0.0, 1.0, 0.0), (0.1, 1.0, 0.2), (0.2, 1.0, 0.4)))
        old_vertices, old_rotations, old_positions = skin_soma30_motion(
            original, roots, self.skin
        )
        new_vertices, new_rotations, new_positions = skin_soma30_motion(
            aligned, roots, self.skin
        )
        displacement = np.linalg.norm(new_vertices - old_vertices, axis=-1)
        affected = self.skin.head_influence > 1e-6
        self.assertGreater(float(displacement[:, affected].max()), 0.05)
        self.assertLess(float(displacement[:, ~affected].max()), 1e-10)
        np.testing.assert_allclose(old_positions[:, 6], new_positions[:, 6], atol=1e-12)
        head77 = int(self.skin.map30_to77[6])
        np.testing.assert_allclose(old_rotations[:, head77], original[:, 6], atol=1e-12)
        np.testing.assert_allclose(new_rotations[:, head77], aligned[:, 6], atol=1e-12)

    def test_core_soma30_global_rotations_survive_expansion(self) -> None:
        frames = 2
        rotations = np.broadcast_to(np.eye(3), (frames, 30, 3, 3)).copy()
        rotations[0, 0] = _yaw(15.0)
        rotations[1, 0] = _yaw(-20.0)
        # Construct a valid hierarchy by assigning every child its parent's world
        # orientation, then replace Head with an additional relative yaw.
        parents = (-1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 3, 10, 11, 12, 13, 13,
                   3, 16, 17, 18, 19, 19, 0, 22, 23, 24, 0, 26, 27, 28)
        for joint, parent in enumerate(parents):
            if parent >= 0:
                rotations[:, joint] = rotations[:, parent]
        rotations[:, 6] = rotations[:, 5] @ _yaw(30.0)
        for joint in (7, 8, 9):
            rotations[:, joint] = rotations[:, 6]
        _vertices, rotations77, _positions77 = skin_soma30_motion(
            rotations, np.zeros((frames, 3)), self.skin
        )
        # Four reduced hand endpoints have intermediate relaxed SOMA-77 chains;
        # all core/body/head joints map one-to-one exactly.
        reduced_hand_endpoints = {
            "LeftHandThumbEnd",
            "LeftHandMiddleEnd",
            "RightHandThumbEnd",
            "RightHandMiddleEnd",
        }
        for index, name in enumerate(SOMA30_NAMES):
            if name in reduced_hand_endpoints:
                continue
            np.testing.assert_allclose(
                rotations77[:, self.skin.map30_to77[index]],
                rotations[:, index],
                atol=1e-12,
                err_msg=name,
            )


if __name__ == "__main__":
    unittest.main()
