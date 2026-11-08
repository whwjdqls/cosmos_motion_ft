"""Map decoded SOMA head/body poses to one explicit Habitat optical anchor."""
import numpy as np
from .camera import validate_pose


def map_motion_to_scene(transforms, camera_start, head_to_camera, scene_units_per_meter=1.):
    transforms = np.asarray(transforms, dtype=np.float64)
    if transforms.shape != (97, 30, 4, 4) or not np.isfinite(transforms).all():
        raise ValueError('Expected finite decoded transforms [97,30,4,4]')
    if not np.isfinite(scene_units_per_meter) or scene_units_per_meter <= 0:
        raise ValueError('Scene scale must be positive')
    head_to_camera = validate_pose(head_to_camera).copy()
    head_to_camera[:3, 3] *= scene_units_per_meter
    # The evaluated decoder uses epsilon-normalized rotations. Project only
    # replay rotations to SO(3); preserve every decoded joint position.
    u, _, vt = np.linalg.svd(transforms[..., :3, :3])
    sign = np.ones((*u.shape[:-2], 3))
    sign[..., -1] = np.linalg.det(u @ vt)
    rigid = transforms.copy()
    rigid[..., :3, :3] = (u * sign[..., None, :]) @ vt
    rigid[..., :3, 3] *= scene_units_per_meter
    camera_local = rigid[:, 6] @ head_to_camera
    anchor = validate_pose(camera_start) @ np.linalg.inv(camera_local[0])
    world = anchor @ rigid
    camera_world = world[:, 6] @ head_to_camera
    for pose in camera_world:
        validate_pose(pose)
    if not np.allclose(camera_world[0], camera_start, atol=1e-9):
        raise ValueError('Head-derived camera anchor failed')
    return dict(joint_poses_scene=world, joints_scene=world[..., :3, 3],
                camera_to_world_optical=camera_world, scene_from_decoded=anchor,
                head_to_camera_scene=head_to_camera,
                rotation_projection_max_abs=float(np.abs(rigid[..., :3, :3] - transforms[..., :3, :3]).max()),
                world_up_tilt_deg=float(np.degrees(np.arccos(np.clip(anchor[1, 1], -1, 1)))))
