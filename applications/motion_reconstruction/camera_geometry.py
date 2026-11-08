"""Camera-local coordinate axes in a common world frame, optical CV convention."""
import numpy as np
AXIS_COLORS=('#D94646','#2E9F61','#356FE0')


def camera_axis_segments(poses,length=.18):
    poses=np.asarray(poses,dtype=np.float64)
    if poses.shape[-2:]!=(4,4) or not np.isfinite(poses).all() or not np.isfinite(length) or length<=0:
        raise ValueError('Expected finite camera-to-world poses and positive axis length')
    origins=poses[...,:3,3]
    # Columns of camera-to-world rotation are the world directions of X/Y/Z.
    ends=origins[...,None,:]+length*poses[...,:3,:3].swapaxes(-1,-2)
    starts=np.broadcast_to(origins[...,None,:],ends.shape)
    return np.stack((starts,ends),axis=-2)
