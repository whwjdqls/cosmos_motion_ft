import numpy as np

def trajectory18(positions, rotations, ground_offset):
    from nymeria_baselines.prepare import camera_trajectory18, Y_FROM_Z, IDENTITY_DELTA
    if positions.shape != (97, 3) or rotations.shape != (97, 3, 3):
        raise ValueError("Trajectory must contain exactly 97 aligned camera poses")
    if ground_offset is None or not np.isfinite(ground_offset):
        raise ValueError("Floor-aware conditioning requires a known calibrated floor offset")
    poses = np.tile(np.eye(4, dtype=np.float64), (97, 1, 1))
    poses[:, :3, :3] = Y_FROM_Z @ rotations.astype(np.float64)
    poses[:, :3, 3] = positions.astype(np.float64) @ Y_FROM_Z.T
    poses[:, 1, 3] -= float(ground_offset)
    features = camera_trajectory18(poses)
    features[0, 9:18] = IDENTITY_DELTA
    if features.shape != (97, 18) or not np.isfinite(features).all():
        raise ValueError("Invalid floor-aware camera features")
    return features.astype(np.float32)
