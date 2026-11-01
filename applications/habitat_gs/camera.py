"""Explicit metric SE(3) and camera-ray conventions; no model dependencies."""
from pathlib import Path
import hashlib
import numpy as np

CV_FROM_GL = np.diag([1., -1., -1., 1.])


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data := stream.read(4 * 1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def validate_pose(pose):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError('Expected finite camera-to-world [4,4]')
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError('Invalid homogeneous pose')
    r = pose[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or abs(np.linalg.det(r) - 1) > 1e-5:
        raise ValueError('Camera rotation must be in SO(3)')
    return pose


def integrate_actions(actions, start_pose, scene_units_per_meter=1.):
    """Cosmos backward_framewise: T_next = T_current @ delta in optical CV axes.

    Rotation6D is the first TWO COLUMNS, not rows. Translation stays metric;
    only an explicit scene-unit conversion is applied, never GT/Sim3 fitting.
    """
    actions = np.asarray(actions, dtype=np.float64)
    if actions.shape != (96, 9) or not np.isfinite(actions).all():
        raise ValueError('Expected finite predicted camera actions [96,9]')
    if not np.isfinite(scene_units_per_meter) or scene_units_per_meter <= 0:
        raise ValueError('Scene units per meter must be positive')
    current = validate_pose(start_pose).copy()
    poses = [current.copy()]
    for action in actions:
        x, y = action[3:6].copy(), action[6:9].copy()
        if np.linalg.norm(x) < 1e-8:
            raise ValueError('Degenerate predicted rotation6D first axis')
        x /= np.linalg.norm(x)
        y -= x * np.dot(x, y)
        if np.linalg.norm(y) < 1e-8:
            raise ValueError('Degenerate predicted rotation6D second axis')
        y /= np.linalg.norm(y)
        delta = np.eye(4)
        delta[:3, :3] = np.stack([x, y, np.cross(x, y)], axis=1)
        delta[:3, 3] = action[:3] * scene_units_per_meter
        current = current @ delta
        poses.append(current.copy())
    return np.stack(poses)


def scaled_intrinsics(k, source_hw, target_hw=(256, 256)):
    """Pillow half-pixel resize convention, after upright image rotation."""
    k = np.array(k, dtype=np.float64, copy=True)
    sy, sx = np.asarray(target_hw) / np.asarray(source_hw)
    k[0, 0] *= sx
    k[1, 1] *= sy
    k[0, 2] = (k[0, 2] + .5) * sx - .5
    k[1, 2] = (k[1, 2] + .5) * sy - .5
    return k


def pinhole_rays(k, hw=(256, 256)):
    y, x = np.mgrid[:hw[0], :hw[1]]
    return np.stack([(x - k[0, 2]) / k[0, 0], (y - k[1, 2]) / k[1, 1], np.ones(hw)], axis=-1)


def cube_rotations():
    """R_target_optical_from_face_optical, right/down/forward column axes."""
    result = []
    for f in np.array([[0,0,1], [1,0,0], [0,0,-1], [-1,0,0], [0,-1,0], [0,1,0]], float):
        up = np.array([0., -1., 0.]) if abs(f[1]) < .5 else np.array([0., 0., 1.])
        right = np.cross(f, up)
        right /= np.linalg.norm(right)
        result.append(np.stack([right, np.cross(f, right), f], axis=1))
    return np.stack(result)


def cube_lookup(rays, face_size):
    rotations = cube_rotations()
    rays = np.asarray(rays, float)
    valid = np.isfinite(rays).all(-1) & (np.linalg.norm(rays, axis=-1) > 1e-8)
    safe = np.where(valid[..., None], rays, [0., 0., 1.])
    faces = np.argmax(safe @ rotations[:, :, 2].T, axis=-1)
    local = np.einsum('...i,...ij->...j', safe, rotations[faces])
    uv = (local[..., :2] / local[..., 2:] + 1.) * face_size / 2. - .5
    return faces, uv.astype(np.float32), valid
