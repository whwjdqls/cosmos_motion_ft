"""Focused numerical contracts for camera-aligned Head re-canonicalization."""
from __future__ import annotations

import unittest

import numpy as np

try:
    from .camera_head_recanonicalization import (
        ARIA_Z_UP_TO_KIMODO_Y_UP,
        DELTA_END,
        HEAD_JOINT_IDX,
        N_FOOT,
        N_JOINTS,
        decode_uniego,
        encode_world_uniego,
        recanonicalize_camera_aligned_head,
        rotation_angle_deg,
        yaw_rotation_y,
    )
    from .uniego_layout import FOOT_JOINT_IDX
except ImportError:  # pragma: no cover - direct test-file execution
    from camera_head_recanonicalization import (
        ARIA_Z_UP_TO_KIMODO_Y_UP,
        DELTA_END,
        HEAD_JOINT_IDX,
        N_FOOT,
        N_JOINTS,
        decode_uniego,
        encode_world_uniego,
        recanonicalize_camera_aligned_head,
        rotation_angle_deg,
        yaw_rotation_y,
    )
    from uniego_layout import FOOT_JOINT_IDX


PARENTS = (-1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 3, 10, 11, 12, 13, 13, 3, 16, 17, 18,
           19, 19, 0, 22, 23, 24, 0, 26, 27, 28)


def _rotation_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Small batched XYZ rotation helper used only to synthesize test motion."""
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    out = np.empty((len(x), 3, 3), dtype=np.float64)
    out[:, 0, 0] = cy * cz
    out[:, 0, 1] = -cy * sz
    out[:, 0, 2] = sy
    out[:, 1, 0] = sx * sy * cz + cx * sz
    out[:, 1, 1] = -sx * sy * sz + cx * cz
    out[:, 1, 2] = -sx * cy
    out[:, 2, 0] = -cx * sy * cz + sx * sz
    out[:, 2, 1] = cx * sy * sz + sx * cz
    out[:, 2, 2] = cx * cy
    return out


class CameraHeadRecanonicalizationTest(unittest.TestCase):
    def setUp(self) -> None:
        frames = 41
        phase = np.linspace(0.0, 2.0 * np.pi, frames)
        base = _rotation_xyz(
            0.12 * np.sin(phase),
            0.8 * np.sin(phase * 0.37),
            0.09 * np.cos(phase * 0.71),
        )
        rotations = np.repeat(base[:, None], N_JOINTS, axis=1)
        for joint in range(N_JOINTS):
            offset = _rotation_xyz(
                np.full(frames, 0.002 * joint),
                np.full(frames, -0.003 * joint),
                np.full(frames, 0.001 * joint),
            )
            rotations[:, joint] = rotations[:, joint] @ offset

        positions = np.zeros((frames, N_JOINTS, 3), dtype=np.float64)
        positions[..., 0] = np.linspace(0.0, 3.0, frames)[:, None]
        positions[..., 2] = (0.4 * np.sin(phase))[:, None]
        positions[..., 1] = np.linspace(0.8, 1.8, N_JOINTS)[None]
        positions += np.arange(N_JOINTS, dtype=np.float64)[None, :, None] * np.array(
            [0.002, 0.0, -0.001]
        )

        true_head = base
        fitting_error = _rotation_xyz(
            0.18 + 0.05 * np.sin(phase),
            0.11 * np.cos(phase * 1.3),
            -0.09 + 0.04 * np.sin(phase * 0.8),
        )
        rotations[:, HEAD_JOINT_IDX] = true_head @ fitting_error
        contacts = (np.arange(frames * N_FOOT).reshape(frames, N_FOOT) % 3 == 0).astype(
            np.float32
        )
        self.old_features = encode_world_uniego(
            rotations, positions, contacts, output_dtype=np.float32
        )
        self.rotation_head_to_camera = _rotation_xyz(
            np.array([2.72]), np.array([-0.23]), np.array([3.04])
        )[0]
        camera_kimodo = true_head @ self.rotation_head_to_camera
        # Invert the fixed world-basis change to synthesize the sidecar convention.
        self.camera_aria = ARIA_Z_UP_TO_KIMODO_Y_UP.T @ camera_kimodo
        self.true_head = true_head

    def test_changes_only_head_world_rotation(self) -> None:
        old = decode_uniego(self.old_features)
        result = recanonicalize_camera_aligned_head(
            self.old_features, self.camera_aria, self.rotation_head_to_camera
        )
        new = decode_uniego(result.features)

        self.assertLess(float(np.max(rotation_angle_deg(
            np.swapaxes(new.world_rotations[:, HEAD_JOINT_IDX], -1, -2) @ self.true_head
        ))), 0.02)
        non_head = np.arange(N_JOINTS) != HEAD_JOINT_IDX
        self.assertLess(
            float(np.max(rotation_angle_deg(
                np.swapaxes(new.world_rotations[:, non_head], -1, -2)
                @ old.world_rotations[:, non_head]
            ))),
            0.02,
        )
        self.assertLess(float(np.max(np.abs(new.world_positions - old.world_positions))), 2e-5)
        self.assertTrue(np.array_equal(
            result.features[:, DELTA_END:], self.old_features[:, DELTA_END:]
        ))

    def test_rebuilds_camera_consistent_canonical_frame(self) -> None:
        result = recanonicalize_camera_aligned_head(
            self.old_features, self.camera_aria, self.rotation_head_to_camera
        )
        new = decode_uniego(result.features)
        forward = new.world_rotations[:, HEAD_JOINT_IDX, :, 2]
        expected_yaw = np.arctan2(forward[:, 0], forward[:, 2])
        expected = yaw_rotation_y(expected_yaw)
        self.assertLess(float(np.max(np.abs(new.canonical_rotations - expected))), 2e-5)

        # The floor-projected head anchor must be the canonical origin in local X/Z.
        head_block = result.features[:, HEAD_JOINT_IDX * 9:(HEAD_JOINT_IDX + 1) * 9]
        self.assertLess(float(np.max(np.abs(head_block[:, (6, 8)]))), 2e-5)

    def test_float32_roundtrip_keeps_all_values_finite(self) -> None:
        result = recanonicalize_camera_aligned_head(
            self.old_features, self.camera_aria, self.rotation_head_to_camera
        )
        self.assertEqual(result.features.shape, self.old_features.shape)
        self.assertEqual(result.features.dtype, np.float32)
        self.assertTrue(np.isfinite(result.features).all())

    def test_foot_skating_floating_and_bone_lengths_are_preserved(self) -> None:
        old = decode_uniego(self.old_features)
        result = recanonicalize_camera_aligned_head(
            self.old_features, self.camera_aria, self.rotation_head_to_camera
        )
        new = decode_uniego(result.features)
        foot_indices = np.asarray(FOOT_JOINT_IDX)
        old_feet = old.world_positions[:, foot_indices]
        new_feet = new.world_positions[:, foot_indices]
        old_speed = np.linalg.norm(np.diff(old_feet[..., (0, 2)], axis=0), axis=-1) * 20.0
        new_speed = np.linalg.norm(np.diff(new_feet[..., (0, 2)], axis=0), axis=-1) * 20.0
        self.assertLess(float(np.max(np.abs(new_speed - old_speed))), 1e-4)

        # Use a floor offset away from decision thresholds and prove classifications
        # for floating/penetration cannot change under the coordinate rewrite.
        floor_offset = float(np.min(old_feet[..., 1]) - 0.02)
        old_height = old_feet[..., 1] - floor_offset
        new_height = new_feet[..., 1] - floor_offset
        self.assertLess(float(np.max(np.abs(new_height - old_height))), 2e-5)
        self.assertTrue(np.array_equal(old_height > 0.10, new_height > 0.10))
        self.assertTrue(np.array_equal(old_height < -0.05, new_height < -0.05))

        children = np.asarray([joint for joint, parent in enumerate(PARENTS) if parent >= 0])
        parents = np.asarray([parent for parent in PARENTS if parent >= 0])
        old_bones = np.linalg.norm(
            old.world_positions[:, children] - old.world_positions[:, parents], axis=-1
        )
        new_bones = np.linalg.norm(
            new.world_positions[:, children] - new.world_positions[:, parents], axis=-1
        )
        self.assertLess(float(np.max(np.abs(new_bones - old_bones))), 2e-5)


if __name__ == "__main__":
    unittest.main()
