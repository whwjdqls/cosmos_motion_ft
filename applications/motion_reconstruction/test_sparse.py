"""Camera-frame invariants, independent of render appearance or dataset files."""
import numpy as np
from .sparse import camera_in_common_frame


def yaw_rotation_y(angle):
    c,s=np.cos(angle),np.sin(angle)
    values=np.zeros(np.shape(angle)+(3,3));values[...,0,0]=values[...,2,2]=c
    values[...,1,1]=1;values[...,0,2]=s;values[...,2,0]=-s
    return values


def test_reference_camera_frame_is_yaw_translation_invariant():
    poses=np.repeat(np.eye(4)[None],3,axis=0)
    poses[:,:3,:3]=yaw_rotation_y(np.array([.2,.3,-.1]))
    poses[:,:3,3]=[[2.,1.7,4.],[2.1,1.6,4.3],[2.3,1.2,4.5]]
    calibration=dict(head_rotation_in_camera=np.eye(3).tolist(),head_offset_camera=[0.,-.1,-.1])
    before=poses.copy();common,yaw,origin=camera_in_common_frame(poses,calibration)
    np.testing.assert_array_equal(before,poses)
    np.testing.assert_allclose(common[:,1,3],poses[:,1,3],atol=1e-14)
    initial_head=common[0,:3,3]+common[0,:3,:3]@calibration['head_offset_camera']
    np.testing.assert_allclose(initial_head[[0,2]],0,atol=1e-14)
    np.testing.assert_allclose(yaw@common[:,:3,:3],poses[:,:3,:3],atol=1e-14)
    np.testing.assert_allclose(common[:,:3,3]@yaw.T+origin,poses[:,:3,3],atol=1e-14)
    moved=poses.copy();rot=yaw_rotation_y(np.array(1.1));shift=np.array([8.,0.,-3.])
    moved[:,:3,:3]=rot@poses[:,:3,:3];moved[:,:3,3]=poses[:,:3,3]@rot.T+shift
    invariant,_,_=camera_in_common_frame(moved,calibration)
    np.testing.assert_allclose(invariant,common,atol=2e-14)


def test_camera_axes_use_columns_and_keep_optical_handedness():
    from .camera_geometry import camera_axis_segments
    poses=np.repeat(np.eye(4)[None],2,axis=0)
    poses[:,:3,:3]=yaw_rotation_y(np.array([0.,.7]));poses[:,:3,3]=[[1.,2.,3.],[-4.,1.,.5]]
    original=poses.copy();segments=camera_axis_segments(poses,.18)
    np.testing.assert_array_equal(poses,original)
    for axis in range(3):
        np.testing.assert_allclose(segments[:,axis,0],poses[:,:3,3],atol=1e-14)
        np.testing.assert_allclose(segments[:,axis,1]-segments[:,axis,0],.18*poses[:,:3,axis],atol=1e-14)
    directions=(segments[:,: ,1]-segments[:,:,0])/.18
    np.testing.assert_allclose(np.cross(directions[:,0],directions[:,1]),directions[:,2],atol=1e-14)
