"""EgoControl body23 features from the exact Phase3 SOMA motion windows.

Reference: CVG-Bonn/EgoControl, commit
5dcfb82ef898315d415f8f04c94337c83d9a099d, data/pose_conditioned/dataset_utils.py.
Angles are radians with R = Rz(z) Ry(y) Rx(x), not Uniego's 6D rotations.
"""
from __future__ import annotations

import numpy as np

# Match the upstream right-before-left slots, explicitly mapping SOMA30's
# left-before-right storage order. Spine slots retain the SOMA chain; they
# are not a claim that SOMA and XSens have identical anatomical landmarks.
BODY23 = (0, 1, 2, 3, 4, 5, 6, 16, 17, 18, 19, 10, 11, 12, 13, 26, 27, 28, 29, 22, 23, 24, 25)
SOMA_BODY_NAMES = ('Hips', 'Spine1', 'Spine2', 'Chest', 'Neck1', 'Neck2', 'Head',
                   'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
                   'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
                   'RightLeg', 'RightShin', 'RightFoot', 'RightToeBase',
                   'LeftLeg', 'LeftShin', 'LeftFoot', 'LeftToeBase')
BODY_NAMES = ('Pelvis', 'L5', 'L3', 'T12', 'T8', 'Neck', 'Head', 'R_Shoulder',
              'R_UpperArm', 'R_Forearm', 'R_Hand', 'L_Shoulder', 'L_UpperArm',
              'L_Forearm', 'L_Hand', 'R_UpperLeg', 'R_LowerLeg', 'R_Foot', 'R_Toe',
              'L_UpperLeg', 'L_LowerLeg', 'L_Foot', 'L_Toe')


def matrix_to_euler(rotation):
    r = np.asarray(rotation, dtype=np.float64)
    sy = np.sqrt(r[..., 0, 0] ** 2 + r[..., 1, 0] ** 2)
    singular = sy < 1e-6
    x = np.where(singular, np.arctan2(-r[..., 1, 2], r[..., 1, 1]),
                 np.arctan2(r[..., 2, 1], r[..., 2, 2]))
    y = np.arctan2(-r[..., 2, 0], sy)
    z = np.where(singular, 0.0, np.arctan2(r[..., 1, 0], r[..., 0, 0]))
    value = np.stack((x, y, z), axis=-1)
    return np.where(value <= -np.pi, value + 2 * np.pi, value)


def body23_to_pose(joints, rotations):
    p, r = np.asarray(joints, dtype=np.float64), np.asarray(rotations, dtype=np.float64)
    if p.shape != (97, 23, 3) or r.shape != (97, 23, 3, 3):
        raise ValueError(f'Expected T97 Body23, got {p.shape}/{r.shape}')
    pieces = []
    for joint in (6, 0):
        previous_inverse = r[:-1, joint].swapaxes(-1, -2)
        delta_p = np.einsum('tij,tj->ti', previous_inverse, p[1:, joint] - p[:-1, joint])
        delta_r = previous_inverse @ r[1:, joint]
        pieces.append(np.concatenate((delta_p, matrix_to_euler(delta_r)), axis=-1)[:, None])
    rest = [j for j in range(23) if j not in (0, 6)]
    inverse_root = r[1:, 0].swapaxes(-1, -2)
    relative_p = np.einsum('tij,tkj->tki', inverse_root, p[1:, rest] - p[1:, :1])
    relative_r = inverse_root[:, None] @ r[1:, rest]
    pieces.append(np.concatenate((relative_p, matrix_to_euler(relative_r)), axis=-1))
    pose = np.concatenate(pieces, axis=1).reshape(96, 138).astype(np.float32)
    if not np.isfinite(pose).all():
        raise ValueError('Non-finite EgoControl pose')
    return pose


def from_normalized_motion(motion, mean, std):
    from nymeria_baselines.visualize_official_baselines import decode
    raw = np.asarray(motion, dtype=np.float32) * std + mean
    p, r = decode(raw, 30)
    return body23_to_pose(p[:, BODY23], r[:, BODY23])
