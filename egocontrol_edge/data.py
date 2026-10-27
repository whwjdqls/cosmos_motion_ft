"""Reuse Phase3's immutable floor-clean windows and Phase1's native packing."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from native_phase_training.latent_nymeria_dataset import NymeriaCameraLatentDataset
from motion_expert_phase3_edge.settings import sha256
from egocontrol_edge.representation import from_normalized_motion


class EgoControlDataset(NymeriaCameraLatentDataset):
    def __init__(self, cache, motion_mean, motion_std, **kwargs):
        kwargs.pop('caption_conditioning', None)
        kwargs['mode'] = 'image2video'
        kwargs['prefix_lengths'] = [1]
        kwargs['prefix_sampling_weights'] = None
        self.root = Path(cache)
        index = json.loads((self.root / 'index.json').read_text())
        if not index['full_training_index']:
            raise ValueError('Production requires the full validated Phase3 cache')
        if sha256(motion_mean) != index['source']['mean'] or sha256(motion_std) != index['source']['std']:
            raise ValueError('Phase3 normalization provenance mismatch')
        self.mean, self.std = np.load(motion_mean), np.load(motion_std)
        self.rows = index['rows']
        if kwargs.get('max_samples') is not None:
            self.rows = self.rows[:int(kwargs['max_samples'])]
        super().__init__(**kwargs)
        self.caption_conditioning = False
        self.arrays = OrderedDict()
        print(f'[egocontrol/data] {len(self.rows)} Phase3 windows; fingerprint={index["fingerprint"]}; '
              'first-frame+body-pose only; captions/actions absent', flush=True)

    def _build_cached_index(self, *args, **kwargs):
        return [{'latent_path': row['latent'], 'cap': ''} for row in self.rows]

    def __getitem__(self, idx):
        row = self.rows[idx % len(self.rows)]
        stat = Path(row['latent']).stat()
        if {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns} != row['latent_identity']:
            raise RuntimeError('Validated Phase3 latent changed: ' + row['latent'])
        sample = super().__getitem__(idx)
        plan = sample['sequence_plan']
        if plan.condition_frame_indexes_vision != [0] or plan.has_action or 'action' in sample:
            raise RuntimeError('M2V input contract changed: only RGB frame0 may be observed')
        key = row['array']
        if key not in self.arrays:
            self.arrays[key] = np.load(self.root / (key + '.motion.npy'), mmap_mode='r', allow_pickle=False)
        self.arrays.move_to_end(key)
        while len(self.arrays) > 16:
            self.arrays.popitem(last=False)
        sample['egocontrol_pose'] = torch.from_numpy(from_normalized_motion(
            self.arrays[key][row['row']], self.mean, self.std))
        sample['egocontrol_case'] = f"{row['uuid']}@{row['start']}"
        return sample


def get_dataset(**kwargs):
    return EgoControlDataset(**kwargs)
