"""SOMA30 Kimodo369 encoding of the SAME calibrated UniEgo physical motion.

Uses the pinned Kimodo smoother, heading and velocity helpers. Positions are
decoded directly (the existing Kimodo inverse's positions mode), avoiding a
second FK fit or canonicalization. Original four contact labels are retained.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import os
from pathlib import Path
import tempfile
import inspect

import numpy as np
import torch


@lru_cache(maxsize=1)
def _dependencies():
    from kimodo.skeleton import SOMASkeleton30
    from kimodo.motion_rep.feature_utils import compute_heading_angle, compute_vel_xyz
    from kimodo.motion_rep.smooth_root import get_smooth_root_pos
    return SOMASkeleton30(), compute_heading_angle, compute_vel_xyz, get_smooth_root_pos


@lru_cache(maxsize=1)
def dependency_identity():
    """Pin helper implementations, skeleton and their geometry dependency."""
    import kimodo.geometry
    import kimodo.tools
    skeleton, heading, velocity, smooth = _dependencies()
    return {name: hashlib.sha256(Path(inspect.getfile(inspect.unwrap(value))).read_bytes()).hexdigest()
            for name, value in dict(skeleton=type(skeleton), heading=heading,
                                    velocity=velocity, smooth=smooth,
                                    geometry=kimodo.geometry, tools=kimodo.tools).items()}


def decode_joints(features):
    if features.shape[-1] != 369:
        raise ValueError("Kimodo features must be 369-D")
    local = features[..., 5:95].reshape(*features.shape[:-1], 30, 3)
    root_xz = features[..., :3] * features.new_tensor([1., 0., 1.])
    return local + root_xz.unsqueeze(-2)


def head_rotation(features):
    from decode_uniego_torch import cont6d_to_matrix
    rotations = features[..., 95:275].reshape(*features.shape[:-1], 30, 6)
    return cont6d_to_matrix(rotations[..., 6, :])


@torch.no_grad()
def encode_uniego(features, fps=20.):
    from decode_uniego_torch import decode_transforms
    if features.ndim != 2 or features.shape[-1] != 283 or len(features) < 1:
        raise ValueError("Expected a valid [T>=1,283] calibrated unpadded window")
    skeleton, heading, velocity, smooth = _dependencies()
    transforms = decode_transforms(features.unsqueeze(0))
    positions = transforms[..., :3, 3]
    rotations = transforms[..., :3, :3]
    root = positions[..., skeleton.root_idx, :]
    # The pinned multigrid smoother needs at least two coarse samples.
    # Keep tiny windows instead of changing the retained source population.
    smoothed = smooth(root).to(positions) if len(features) >= 4 else root
    offset = smoothed * smoothed.new_tensor([1., 0., 1.])
    local_positions = positions - offset.unsqueeze(-2)
    angle = heading(positions, skeleton)
    # Same column convention as Kimodo matrix_to_cont6d; retain all global axes.
    rot6 = torch.cat((rotations[..., :, 0], rotations[..., :, 1]), dim=-1)
    velocities = (velocity(positions, fps, lengths=torch.tensor([len(features)], device=positions.device))
                  if len(features) > 1 else torch.zeros_like(positions))
    result = torch.cat((smoothed, torch.stack((angle.cos(), angle.sin()), dim=-1),
                        local_positions.flatten(-2), rot6.flatten(-2),
                        velocities.flatten(-2), features[None, :, -4:]), dim=-1)[0]
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite converted Kimodo window")
    return result


def cached_encode(features, fps=20.):
    """Cache exact unnormalized windows; key includes codec and dependency source."""
    root = os.environ.get("PHASE2_KIMODO_CACHE_ROOT")
    if not root:
        return encode_uniego(features, fps)
    identity = hashlib.sha256(Path(__file__).read_bytes())
    for name, digest in sorted(dependency_identity().items()):
        identity.update((name+digest).encode())
    identity.update(str(fps).encode());identity.update(features.contiguous().numpy().tobytes())
    key = identity.hexdigest();path = Path(root)/key[:2]/(key+'.npy')
    if path.exists():
        array = np.load(path, allow_pickle=False)
        if array.shape != (len(features),369) or not np.isfinite(array).all():
            raise ValueError(f"Invalid Kimodo cache entry: {path}")
        return torch.from_numpy(array)
    result = encode_uniego(features, fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npy', delete=False) as f:
        temporary = Path(f.name);np.save(f, result.numpy())
    try:
        try: os.link(temporary, path)
        except FileExistsError: pass
    finally: temporary.unlink(missing_ok=True)
    return result
