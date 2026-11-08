import numpy as np
import pytest
from .camera import CV_FROM_GL
from .human_geometry import map_motion_to_scene


def test_noncommuting_head_offset_full_body_anchor_and_metric_scale():
    transforms = np.broadcast_to(np.eye(4), (97,30,4,4)).copy()
    for t in range(97):
        a = t*.002
        r = np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
        transforms[t,:,:3,:3] = r
        transforms[t,:,0,3] = t*.01
    transforms[:,:,1,3] = np.arange(30)[None]*.05
    head = np.eye(4)
    head[:3,:3] = CV_FROM_GL[:3,:3]
    head[:3,3] = [.03,.05,.1]
    start = CV_FROM_GL.copy()
    start[:3,3] = [2.,1.7,3.]
    result = map_motion_to_scene(transforms,start,head,2.)
    camera = result['camera_to_world_optical']
    np.testing.assert_allclose(camera[0],start,atol=1e-12)
    np.testing.assert_allclose(camera,result['joint_poses_scene'][:,6]@result['head_to_camera_scene'],atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(np.diff(camera[:,:3,3],axis=0),axis=-1),
        np.linalg.norm(np.diff((transforms[:,6]@head)[:,:3,3],axis=0),axis=-1)*2,atol=1e-12)
    # A second segment starts exactly at the previous endpoint.
    second = map_motion_to_scene(transforms,camera[-1],head,2.)
    np.testing.assert_allclose(second['camera_to_world_optical'][0],camera[-1],atol=1e-12)


def test_replay_rotation_projection_preserves_all_joint_positions():
    transforms = np.broadcast_to(np.eye(4), (97,30,4,4)).copy()
    transforms[:,:,:3,:3] *= .99999
    transforms[:,:,1,3] = np.arange(30)[None]*.05
    start = np.eye(4)
    result = map_motion_to_scene(transforms,start,np.eye(4))
    expected = np.einsum('ij,tkj->tki',result['scene_from_decoded'][:3,:3],transforms[:,:,:3,3])+result['scene_from_decoded'][:3,3]
    np.testing.assert_allclose(result['joints_scene'],expected,atol=1e-12)
    assert result['rotation_projection_max_abs'] > 0
    with pytest.raises(ValueError):
        map_motion_to_scene(transforms,start,np.eye(4),-1)
