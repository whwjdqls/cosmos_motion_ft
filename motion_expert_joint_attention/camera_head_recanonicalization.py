"""Pure-NumPy correction of Nymeria UniEgo Head orientation from the RGB camera.

The stored 283-D representation is head-canonical.  Correcting only the six Head
rotation channels would leave every other joint expressed in the old, noisy
canonical frame.  This module therefore performs a lossless change of coordinates:

1. decode every joint into the Kimodo Y-up world frame;
2. replace only the SOMA ``Head`` world rotation using the measured upright RGB
   camera rotation and the train-split rigid Head-to-camera calibration;
3. rebuild the yaw-only canonical frame from the corrected Head; and
4. express all world transforms in that new canonical frame.

World positions and all non-Head world rotations are inputs to the re-encoder and
are not modified.  Saving float32 residual transforms introduces the same small
long-sequence accumulation error as the original UniEgo representation; windowed
decoding remains float32-accurate.

Coordinate contract
-------------------
``T_WC = T_WH @ T_HC``.  The camera sidecar stores upright-camera rotations in
Aria's Z-up world basis, so

``R_WH_corrected = (R_aria_to_kimodo @ R_WC_aria) @ R_HC.T``.

The calibrated Head and RGB-camera frames have different local axes.  Directly
copying the camera matrix into the Head joint would therefore be incorrect.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


N_JOINTS = 30
N_FOOT = 4
FEATURE_DIM = N_JOINTS * 9 + 9 + N_FOOT
HEAD_JOINT_IDX = 6
ROOT_JOINT_IDX = 0
LOCAL_END = N_JOINTS * 9
DELTA_END = LOCAL_END + 9

# Maps a vector expressed in the Aria/MPS Z-up world basis into Kimodo's Y-up
# world basis: (x, y, z)_kimodo = (x, z, -y)_aria.
ARIA_Z_UP_TO_KIMODO_Y_UP = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)


@dataclass(frozen=True)
class DecodedUniEgo:
    """World and canonical transforms decoded from one UniEgo sequence."""

    world_rotations: np.ndarray  # [T,J,3,3]
    world_positions: np.ndarray  # [T,J,3]
    canonical_rotations: np.ndarray  # [T,3,3]
    canonical_positions: np.ndarray  # [T,3]


@dataclass(frozen=True)
class RecanonicalizationResult:
    """Corrected features plus the exact world transforms passed to the encoder."""

    features: np.ndarray  # [T,283]
    old_world_rotations: np.ndarray  # [T,J,3,3]
    world_positions: np.ndarray  # [T,J,3]
    corrected_head_rotations: np.ndarray  # [T,3,3]


def _require_shape(name: str, value: np.ndarray, shape: tuple[int | None, ...]) -> None:
    if value.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(value.shape, shape)
    ):
        expected_text = "x".join("*" if item is None else str(item) for item in shape)
        raise ValueError(f"{name} must have shape {expected_text}; got {value.shape}")


def _validate_so3(name: str, rotations: np.ndarray, tolerance: float = 2e-4) -> None:
    rotations = np.asarray(rotations, dtype=np.float64)
    _require_shape(name, rotations, (None, 3, 3))
    if not np.isfinite(rotations).all():
        raise ValueError(f"{name} contains non-finite values")
    identity = np.eye(3, dtype=np.float64)
    gram = np.swapaxes(rotations, -1, -2) @ rotations
    ortho_error = float(np.max(np.abs(gram - identity)))
    determinant_error = float(np.max(np.abs(np.linalg.det(rotations) - 1.0)))
    if ortho_error > tolerance or determinant_error > tolerance:
        raise ValueError(
            f"{name} is not SO(3): max orthogonality error={ortho_error:.3e}, "
            f"max determinant error={determinant_error:.3e}"
        )


def load_rotation_head_to_camera(path: str | Path) -> tuple[np.ndarray, dict]:
    """Load and validate the train-only rigid Head-to-upright-camera rotation."""
    path = Path(path)
    with path.open() as input_file:
        payload = json.load(input_file)
    rotation = np.asarray(payload["rotation_head_to_upright_camera"], dtype=np.float64)
    _require_shape("rotation_head_to_upright_camera", rotation, (3, 3))
    _validate_so3("rotation_head_to_upright_camera", rotation[None])
    if payload.get("split") != "train":
        raise ValueError(f"{path}: correction calibration must declare split='train'")
    return rotation, payload


def cont6d_to_matrix(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Column-convention continuous 6D rotations to matrices."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] != 6:
        raise ValueError(f"continuous rotations must end in 6; got {values.shape}")
    x_axis = values[..., :3]
    y_raw = values[..., 3:6]
    x_norm = np.linalg.norm(x_axis, axis=-1, keepdims=True)
    if np.any(x_norm <= eps):
        raise ValueError("degenerate continuous-6D x axis")
    x_axis = x_axis / x_norm
    z_axis = np.cross(x_axis, y_raw)
    z_norm = np.linalg.norm(z_axis, axis=-1, keepdims=True)
    if np.any(z_norm <= eps):
        raise ValueError("degenerate continuous-6D y axis")
    z_axis = z_axis / z_norm
    y_axis = np.cross(z_axis, x_axis)
    return np.stack((x_axis, y_axis, z_axis), axis=-1)


def matrix_to_cont6d(rotations: np.ndarray) -> np.ndarray:
    """Rotation matrices to column-convention continuous 6D rotations."""
    rotations = np.asarray(rotations)
    if rotations.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrices must end in 3x3; got {rotations.shape}")
    return np.concatenate((rotations[..., :, 0], rotations[..., :, 1]), axis=-1)


def yaw_rotation_y(yaw: np.ndarray) -> np.ndarray:
    """Return matrices mapping canonical +Z to heading ``yaw`` about world +Y."""
    yaw = np.asarray(yaw, dtype=np.float64)
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    out = np.zeros(yaw.shape + (3, 3), dtype=np.float64)
    out[..., 0, 0] = cosine
    out[..., 0, 2] = sine
    out[..., 1, 1] = 1.0
    out[..., 2, 0] = -sine
    out[..., 2, 2] = cosine
    return out


def decode_uniego(features: np.ndarray) -> DecodedUniEgo:
    """Decode raw, unnormalized 283-D features into world transforms.

    Computation is float64 to keep whole-sequence residual accumulation error small.
    The recurrence is the exact UniEgo contract: ``cM[t] = cM[t-1] @ delta[t]``.
    """
    features = np.asarray(features)
    _require_shape("features", features, (None, FEATURE_DIM))
    if len(features) == 0:
        raise ValueError("cannot decode an empty UniEgo sequence")
    if not np.isfinite(features).all():
        raise ValueError("features contain non-finite values")

    frames = len(features)
    joints = features[:, :LOCAL_END].astype(np.float64).reshape(frames, N_JOINTS, 9)
    local_rotations = cont6d_to_matrix(joints[..., :6])
    local_positions = joints[..., 6:9]
    delta_features = features[:, LOCAL_END:DELTA_END].astype(np.float64)
    delta_rotations = cont6d_to_matrix(delta_features[:, :6])
    delta_positions = delta_features[:, 6:9]

    canonical_rotations = np.empty((frames, 3, 3), dtype=np.float64)
    canonical_positions = np.empty((frames, 3), dtype=np.float64)
    canonical_rotations[0] = delta_rotations[0]
    canonical_positions[0] = delta_positions[0]
    for frame in range(1, frames):
        previous_rotation = canonical_rotations[frame - 1]
        canonical_positions[frame] = (
            canonical_positions[frame - 1]
            + previous_rotation @ delta_positions[frame]
        )
        canonical_rotations[frame] = previous_rotation @ delta_rotations[frame]

    world_rotations = np.einsum(
        "tij,tkjl->tkil", canonical_rotations, local_rotations, optimize=True
    )
    world_positions = canonical_positions[:, None, :] + np.einsum(
        "tij,tkj->tki", canonical_rotations, local_positions, optimize=True
    )
    return DecodedUniEgo(
        world_rotations=world_rotations,
        world_positions=world_positions,
        canonical_rotations=canonical_rotations,
        canonical_positions=canonical_positions,
    )


def canonical_frame_from_head(
    world_rotations: np.ndarray,
    world_positions: np.ndarray,
    *,
    head_idx: int = HEAD_JOINT_IDX,
    root_idx: int = ROOT_JOINT_IDX,
    forward_eps: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the yaw-only, floor-projected UniEgo canonical frame."""
    world_rotations = np.asarray(world_rotations, dtype=np.float64)
    world_positions = np.asarray(world_positions, dtype=np.float64)
    _require_shape("world_rotations", world_rotations, (None, N_JOINTS, 3, 3))
    _require_shape("world_positions", world_positions, (len(world_rotations), N_JOINTS, 3))

    forward = world_rotations[:, head_idx, :, 2].copy()
    degenerate = np.linalg.norm(forward[:, (0, 2)], axis=-1) < forward_eps
    if np.any(degenerate):
        forward[degenerate] = world_rotations[degenerate, root_idx, :, 2]
    yaw = np.arctan2(forward[:, 0], forward[:, 2])
    canonical_rotations = yaw_rotation_y(yaw)
    head_positions = world_positions[:, head_idx]
    canonical_positions = np.stack(
        (head_positions[:, 0], np.zeros(len(head_positions)), head_positions[:, 2]),
        axis=-1,
    )
    return canonical_rotations, canonical_positions


def encode_world_uniego(
    world_rotations: np.ndarray,
    world_positions: np.ndarray,
    foot_contacts: np.ndarray,
    *,
    output_dtype: np.dtype | type = np.float32,
    head_idx: int = HEAD_JOINT_IDX,
) -> np.ndarray:
    """Encode world joint transforms with a canonical frame derived from ``head_idx``."""
    world_rotations = np.asarray(world_rotations, dtype=np.float64)
    world_positions = np.asarray(world_positions, dtype=np.float64)
    foot_contacts = np.asarray(foot_contacts)
    _require_shape("world_rotations", world_rotations, (None, N_JOINTS, 3, 3))
    frames = len(world_rotations)
    _require_shape("world_positions", world_positions, (frames, N_JOINTS, 3))
    _require_shape("foot_contacts", foot_contacts, (frames, N_FOOT))
    if not (
        np.isfinite(world_rotations).all()
        and np.isfinite(world_positions).all()
        and np.isfinite(foot_contacts).all()
    ):
        raise ValueError("world transforms or contacts contain non-finite values")

    canonical_rotations, canonical_positions = canonical_frame_from_head(
        world_rotations, world_positions, head_idx=head_idx
    )
    inverse_rotations = np.swapaxes(canonical_rotations, -1, -2)
    local_rotations = np.einsum(
        "tij,tkjl->tkil", inverse_rotations, world_rotations, optimize=True
    )
    centered_positions = world_positions - canonical_positions[:, None, :]
    local_positions = np.einsum(
        "tij,tkj->tki", inverse_rotations, centered_positions, optimize=True
    )
    joint_features = np.concatenate(
        (matrix_to_cont6d(local_rotations), local_positions), axis=-1
    ).reshape(frames, LOCAL_END)

    delta_rotations = np.empty_like(canonical_rotations)
    delta_positions = np.empty_like(canonical_positions)
    delta_rotations[0] = canonical_rotations[0]
    delta_positions[0] = canonical_positions[0]
    if frames > 1:
        previous_inverse = np.swapaxes(canonical_rotations[:-1], -1, -2)
        delta_rotations[1:] = previous_inverse @ canonical_rotations[1:]
        delta_positions[1:] = np.einsum(
            "tij,tj->ti",
            previous_inverse,
            canonical_positions[1:] - canonical_positions[:-1],
            optimize=True,
        )
    delta_features = np.concatenate(
        (matrix_to_cont6d(delta_rotations), delta_positions), axis=-1
    )
    features = np.concatenate((joint_features, delta_features, foot_contacts), axis=-1)
    if features.shape != (frames, FEATURE_DIM):
        raise AssertionError(f"internal feature shape error: {features.shape}")
    return features.astype(output_dtype, copy=False)


def camera_rotations_to_kimodo(camera_world_rotations_upright: np.ndarray) -> np.ndarray:
    """Convert upright RGB-camera world rotations from Aria Z-up to Kimodo Y-up."""
    rotations = np.asarray(camera_world_rotations_upright, dtype=np.float64)
    _require_shape("camera_world_rotations_upright", rotations, (None, 3, 3))
    _validate_so3("camera_world_rotations_upright", rotations)
    return ARIA_Z_UP_TO_KIMODO_Y_UP @ rotations


def corrected_head_rotations(
    camera_world_rotations_upright: np.ndarray,
    rotation_head_to_camera: np.ndarray,
) -> np.ndarray:
    """Compute ``R_WH`` from measured ``R_WC`` under ``R_WC = R_WH R_HC``."""
    camera_kimodo = camera_rotations_to_kimodo(camera_world_rotations_upright)
    rotation_head_to_camera = np.asarray(rotation_head_to_camera, dtype=np.float64)
    _require_shape("rotation_head_to_camera", rotation_head_to_camera, (3, 3))
    _validate_so3("rotation_head_to_camera", rotation_head_to_camera[None])
    corrected = camera_kimodo @ rotation_head_to_camera.T
    _validate_so3("corrected_head_rotations", corrected)
    return corrected


def recanonicalize_camera_aligned_head(
    features: np.ndarray,
    camera_world_rotations_upright: np.ndarray,
    rotation_head_to_camera: np.ndarray,
    *,
    output_dtype: np.dtype | type | None = None,
) -> RecanonicalizationResult:
    """Correct Head world orientation and fully rebuild one UniEgo sequence."""
    features = np.asarray(features)
    _require_shape("features", features, (None, FEATURE_DIM))
    camera_world_rotations_upright = np.asarray(camera_world_rotations_upright)
    _require_shape(
        "camera_world_rotations_upright",
        camera_world_rotations_upright,
        (len(features), 3, 3),
    )
    decoded = decode_uniego(features)
    new_head = corrected_head_rotations(
        camera_world_rotations_upright, rotation_head_to_camera
    )
    new_world_rotations = decoded.world_rotations.copy()
    new_world_rotations[:, HEAD_JOINT_IDX] = new_head
    dtype = features.dtype if output_dtype is None else output_dtype
    corrected_features = encode_world_uniego(
        new_world_rotations,
        decoded.world_positions,
        features[:, DELTA_END:FEATURE_DIM],
        output_dtype=dtype,
    )
    # Contacts are categorical passthrough data; make bitwise preservation explicit.
    corrected_features[:, DELTA_END:FEATURE_DIM] = features[:, DELTA_END:FEATURE_DIM]
    return RecanonicalizationResult(
        features=corrected_features,
        old_world_rotations=decoded.world_rotations,
        world_positions=decoded.world_positions,
        corrected_head_rotations=new_head,
    )


def rotation_angle_deg(rotations: np.ndarray) -> np.ndarray:
    """Geodesic SO(3) angle in degrees for one or more rotation matrices."""
    rotations = np.asarray(rotations, dtype=np.float64)
    if rotations.shape[-2:] != (3, 3):
        raise ValueError(f"rotations must end in 3x3; got {rotations.shape}")
    trace = np.trace(rotations, axis1=-2, axis2=-1)
    cosine = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine))


__all__ = [
    "ARIA_Z_UP_TO_KIMODO_Y_UP",
    "DELTA_END",
    "DecodedUniEgo",
    "FEATURE_DIM",
    "HEAD_JOINT_IDX",
    "LOCAL_END",
    "N_FOOT",
    "N_JOINTS",
    "RecanonicalizationResult",
    "camera_rotations_to_kimodo",
    "canonical_frame_from_head",
    "cont6d_to_matrix",
    "corrected_head_rotations",
    "decode_uniego",
    "encode_world_uniego",
    "load_rotation_head_to_camera",
    "matrix_to_cont6d",
    "recanonicalize_camera_aligned_head",
    "rotation_angle_deg",
    "yaw_rotation_y",
]
