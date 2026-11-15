"""Explicit gauge/conditioning adapters for predicted metric CV camera poses.

The initial gravity orientation and height are supplied by a TRAIN-ONLY prior.
They are not observable from camera0-relative SE(3) alone. No test-camera,
target-body, test-floor, scale fitting, or prediction grounding is used here.
"""
import numpy as np

from nymeria_baselines.prepare import camera_trajectory18, IDENTITY_DELTA, Y_FROM_Z


def validate_poses(poses):
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape != (97, 4, 4) or not np.isfinite(poses).all():
        raise ValueError('Finite 97x4x4 metric camera-to-world poses required')
    np.testing.assert_allclose(poses[:, 3], np.tile([0, 0, 0, 1], (97, 1)), atol=2e-5)
    rotation = poses[:, :3, :3]
    np.testing.assert_allclose(rotation.swapaxes(-1, -2) @ rotation,
                               np.tile(np.eye(3), (97, 1, 1)), atol=2e-4)
    np.testing.assert_allclose(np.linalg.det(rotation), 1., atol=2e-4)
    return poses


def yaw_rotation(rotation):
    forward = np.asarray(rotation)[:, 2]
    if np.linalg.norm(forward[[0, 2]]) < 1e-4:
        raise ValueError('Undefined heading for vertically pointing camera')
    yaw = np.arctan2(forward[0], forward[2])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def remove_initial_gauge(poses, initial_rotation, height):
    """Keep all predicted relative transforms/metric scale; supply one gauge."""
    poses = validate_poses(poses)
    relative = np.linalg.inv(poses[0]) @ poses
    gauge = np.eye(4)
    gauge[:3, :3] = initial_rotation
    gauge[1, 3] = height
    result = gauge @ relative
    # Check actual relative SE(3), not just translation norms.
    np.testing.assert_allclose(np.linalg.inv(result[0]) @ result, relative, atol=1e-10)
    return result


def egoego_condition(poses):
    poses = validate_poses(poses)
    anchor = yaw_rotation(poses[0, :3, :3])
    origin = poses[0, :3, 3] * [1, 0, 1]
    positions = (poses[:, :3, 3] - origin) @ anchor
    rotations = anchor.T @ poses[:, :3, :3]
    return np.concatenate([positions, rotations[:, :2, :].reshape(97, 6)], -1).astype(np.float32)


def uem_condition(poses):
    value = camera_trajectory18(validate_poses(poses))
    value[0, 9:18] = IDENTITY_DELTA
    return value


def egoallo_poses(poses):
    """Native Z-up world / unchanged CV camera axes; duplicate frame0 previous."""
    poses = validate_poses(poses)
    zup = poses.copy()
    zup[:, :3, :3] = Y_FROM_Z.T @ poses[:, :3, :3]
    zup[:, :3, 3] = poses[:, :3, 3] @ Y_FROM_Z
    return np.concatenate([zup[:1], zup], axis=0).astype(np.float32)


def common_from_world(poses, calibration):
    """Match existing head-yaw/horizontal-origin benchmark using camera only."""
    poses = validate_poses(poses)
    rotation = np.asarray(calibration['head_rotation_in_camera'])
    offset = np.asarray(calibration['head_offset_camera'])
    yaw = yaw_rotation(poses[0, :3, :3] @ rotation)
    origin = (poses[0, :3, 3] + poses[0, :3, :3] @ offset) * [1, 0, 1]
    transform = np.eye(4)
    transform[:3, :3] = yaw.T
    transform[:3, 3] = -yaw.T @ origin
    return transform


def egoego_to_common(normalized, mean, std, poses, calibration):
    """Decode the actual predicted positions, without replacing them by FK."""
    poses = validate_poses(poses)
    raw = normalized * std + mean
    positions = raw[:, :69].reshape(97, 23, 3)
    row6 = raw[:, 69:].reshape(97, 23, 6).astype(np.float64)
    first = row6[..., :3]
    first /= np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    second = row6[..., 3:] - (first * row6[..., 3:]).sum(-1, keepdims=True) * first
    second /= np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12)
    rotations = np.stack([first, second, np.cross(first, second)], axis=-2)
    anchor = np.eye(4)
    anchor[:3, :3] = yaw_rotation(poses[0, :3, :3])
    anchor[:3, 3] = poses[0, :3, 3] * [1, 0, 1]
    common = common_from_world(poses, calibration) @ anchor
    joints = positions @ common[:3, :3].T + common[:3, 3]
    rotations = common[:3, :3] @ rotations
    return joints.astype(np.float32), rotations.astype(np.float32)
