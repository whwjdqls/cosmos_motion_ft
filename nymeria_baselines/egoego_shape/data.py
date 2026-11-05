"""Same Nymeria train/val/test cohort as UEM; camera-only coordinate anchoring.

Targets are global 23-joint positions followed by global 6D rotations (the
PyTorch3D/EgoEgo row convention). Global means a common floor-aligned world
frame, with the initial camera horizontal translation/yaw removed. It does
not mean parent-local rotations or root-relative joint positions.
"""
import argparse
from collections import OrderedDict, defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from nymeria_baselines.body_shape import BODY_JOINTS
from motion_expert_joint_attention.camera_head_recanonicalization import (
    cont6d_to_matrix, yaw_rotation_y, ARIA_Z_UP_TO_KIMODO_Y_UP as Y_FROM_Z)
from .upstream import sha

CACHE = Path('/scratch2/whwjdqls99/cosmos/caches/nymeria_soma_official_baselines_T97_20fps_floorclean_20261016')


def require_compute():
    if not os.environ.get('SLURM_JOB_ID') or 'login' in os.uname().nodename:
        raise RuntimeError('Data/model work requires an owned compute allocation')


def raw_window(row, arrays):
    s = row['start']; stop = s+97
    motion, pose, neutral, root, cpf = arrays
    motion = np.asarray(motion[s:stop], dtype=np.float64)
    camera = np.asarray(pose[s:stop], dtype=np.float64)
    root = np.asarray(root[s:stop], dtype=np.float64)
    cpf = np.asarray(cpf[s:stop], dtype=np.float64)
    assert motion.shape == (97, 283) and camera.shape == (97, 4, 4)
    # Independently prepared EgoAllo metadata preserves the source world frame.
    world_positions = np.concatenate((root[:, None, 4:7],
        camera[:, None, :3, 3] + np.einsum('tij,tkj->tki', camera[:, :3, :3], cpf)), axis=1)
    world_positions = np.einsum('ij,tkj->tki', Y_FROM_Z, world_positions)
    canonical = cont6d_to_matrix(motion[:, :270].reshape(97, 30, 9)[..., :6])
    root_rot = Rotation.from_quat(root[:, [1, 2, 3, 0]]).as_matrix()
    world_rot = (Y_FROM_Z @ root_rot @ canonical[:, 0].swapaxes(-1, -2))[:, None] @ canonical[:, BODY_JOINTS]
    cam_rot = Y_FROM_Z @ camera[:, :3, :3]
    cam_pos = np.einsum('ij,tj->ti', Y_FROM_Z, camera[:, :3, 3])
    floor = 0. if row['ground_offset_y'] is None else float(row['ground_offset_y'])
    world_positions[..., 1] -= floor
    cam_pos[:, 1] -= floor
    # This anchor uses only the supplied camera, never target body/head pose.
    forward = cam_rot[0, :, 2]
    anchor_rot = yaw_rotation_y(np.arctan2(forward[0], forward[2]))
    origin = cam_pos[0] * [1., 0., 1.]
    inverse = anchor_rot.T
    positions = np.einsum('ij,tkj->tki', inverse, world_positions-origin)
    rotations = inverse @ world_rot
    camera_position = np.einsum('ij,tj->ti', inverse, cam_pos-origin)
    camera_rotation = inverse @ cam_rot
    # Original EgoEgo uses PyTorch3D's first-two-rows 6D convention.
    target = np.concatenate((positions.reshape(97, 69), rotations[..., :2, :].reshape(97, 138)), axis=-1)
    condition = np.concatenate((camera_position, camera_rotation[:, :2, :].reshape(97, 6)), axis=-1)
    shape = np.array(neutral, dtype=np.float32, copy=True)
    shape -= shape.mean(axis=0, keepdims=True)
    if not all(np.isfinite(x).all() for x in (target, condition, shape)):
        raise ValueError('Non-finite source window: '+str(row))
    return target.astype(np.float32), condition.astype(np.float32), shape, (anchor_rot, origin)


def open_arrays(cache, key):
    cache = Path(cache)
    files = [cache/'records'/f'{key}.{suffix}.npy' for suffix in ('motion', 'pose_zup', 'neutral')]
    files += [cache/'egoallo_metadata_v1'/f'{key}.{suffix}.npy' for suffix in ('world_root_zup', 'joints_wrt_cpf')]
    return tuple(np.load(p, mmap_mode='r', allow_pickle=False) for p in files)


class MotionDataset(Dataset):
    def __init__(self, stats, split, cache=CACHE):
        self.cache = Path(cache)
        index = json.loads((self.cache/'index.json').read_text())
        self.rows = [r for r in index['rows'] if r['split'] == split]
        self.arrays = OrderedDict()
        with np.load(stats, allow_pickle=False) as z:
            self.mean, self.std = z['mean'].copy(), z['std'].copy()
            self.camera_mean, self.camera_std = z['camera_mean'].copy(), z['camera_std'].copy()

    def __len__(self):
        return len(self.rows)

    def raw(self, idx):
        row = self.rows[idx]; key = row['array']
        if key not in self.arrays:
            self.arrays[key] = open_arrays(self.cache, key)
        self.arrays.move_to_end(key)
        while len(self.arrays) > 8:
            self.arrays.popitem(last=False)
        return raw_window(row, self.arrays[key])

    def __getitem__(self, idx):
        target, camera, shape, _ = self.raw(idx)
        return {'motion': (target-self.mean)/self.std,
                'camera': (camera-self.camera_mean)/self.camera_std,
                'shape': shape, 'index': idx}


def _statistics(task):
    cache, rows = task
    arrays = open_arrays(cache, rows[0]['array'])
    sums = np.zeros(216, dtype=np.float64); squares = sums.copy(); count = 0
    for row in rows:
        motion, camera, _, _ = raw_window(row, arrays)
        value = np.concatenate((motion, camera), axis=-1).astype(np.float64)
        sums += value.sum(0); squares += np.square(value).sum(0); count += len(value)
    return sums, squares, count


def prepare(output, cache=CACHE, workers=8):
    require_compute(); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    index = json.loads((cache/'index.json').read_text())
    assert index['window'] == 97 and index['fps'] == 20
    counts = Counter(r['split'] for r in index['rows'])
    assert counts == {'train': 101576, 'val': 10927, 'test': 11911}, counts
    cohorts = {s: {r['uuid'] for r in index['rows'] if r['split']==s} for s in counts}
    assert not any(cohorts[a]&cohorts[b] for a,b in [('train','val'),('train','test'),('val','test')])
    sources = {'cache_index_sha256': sha(cache/'index.json'),
               'metadata_index_sha256': sha(cache/'egoallo_metadata_v1/index.json'),
               'implementation_sha256': sha(__file__), 'counts': dict(counts),
               'statistics_split': 'train_only_window_weighted', 'joints': list(BODY_JOINTS),
               'representation': 'global_position69_then_global_rotation138_row6D_camera_anchored_Yup',
               'camera': 'position3_rotation6_row6D_no_anatomical_head_extrinsic',
               'shape': 'full_SOMA30_centroid_centered_meters_no_PCA'}
    if (output/'READY.json').exists():
        assert json.loads((output/'READY.json').read_text()) == sources
        assert (output/'stats.npz').is_file()
        return
    groups = defaultdict(list)
    for row in index['rows']:
        if row['split'] == 'train': groups[row['array']].append(row)
    sums = np.zeros(216); squares = sums.copy(); count = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks = [(str(cache), groups[key]) for key in sorted(groups)]
        for i,(total,square,n) in enumerate(pool.map(_statistics,tasks),1):
            sums += total; squares += square; count += n
            if i % 50 == 0 or i == len(tasks): print('statistics',i,len(tasks),count,flush=True)
    assert count == 101576*97
    mean = sums/count; std = np.sqrt(np.maximum(squares/count-mean**2, 0)).clip(min=1e-6)
    np.savez(output/'stats.tmp.npz', mean=mean[:207].astype(np.float32), std=std[:207].astype(np.float32),
             camera_mean=mean[207:].astype(np.float32), camera_std=std[207:].astype(np.float32), frames=count)
    (output/'stats.tmp.npz').replace(output/'stats.npz')
    (output/'READY.json').write_text(json.dumps(sources,indent=2)+'\n')
    print(json.dumps(sources),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();prepare(a.output,workers=a.workers)
