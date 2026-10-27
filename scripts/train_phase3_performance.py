"""Run an existing Phase-3 trainer with the validated FP32 bridge-mask cache.

Example, inside the normal site wrapper / torchrun launch:
  scripts/train_phase3_performance.py --training-module
  motion_expert_phase3_edge_fourtask.train --out RUN [original trainer args]

Original recipe/source fingerprint checks remain enabled. The execution-only
change has a separate manifest and is recorded in every subsequent checkpoint.
It does not change parameters, optimizer, dtype, TF32, or checkpoint policy.
"""
import argparse
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

import torch

from scripts.phase3_performance import execution_policy


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--training-module', required=True, choices=[
        'motion_expert_phase3_edge_fourtask.train',
        'motion_expert_phase3_edge_camfilm.train',
        'motion_expert_phase3_edge_camtrajfilm.train',
    ])
    options, trainer_args = parser.parse_known_args()
    if '--out' not in trainer_args:
        raise ValueError('An explicit existing --out is required for an auditable resume')
    out = Path(trainer_args[trainer_args.index('--out') + 1])
    train = importlib.import_module(options.training_module)
    train.require_compute_node()
    implementation = Path(__file__).with_name('phase3_performance.py')
    provenance = {
        'schema': 1, 'optimization': 'forward_local_fp32_bridge_mask_cache_v1',
        'started_at': datetime.datetime.now().astimezone().isoformat(),
        'job': os.environ.get('SLURM_JOB_ID'), 'trainer': options.training_module,
        'implementation_sha256': hashlib.sha256(implementation.read_bytes()).hexdigest(),
        'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'cache_bridge_masks': True, 'motion_activation_checkpointing': True,
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'torch': torch.__version__,
        'validation_reference': os.environ.get('PHASE3_PERFORMANCE_VALIDATION', ''),
        'scientific_contract': 'unchanged; original checkpoint fingerprint is still checked',
    }
    if int(os.environ.get('RANK', '0')) == 0:
        directory = out / 'performance_runtime'
        directory.mkdir(parents=True, exist_ok=True)
        train.atomic_json(directory / (str(os.environ.get('SLURM_JOB_ID', 'local')) + '.json'), provenance)
        print('[phase3/performance] ' + json.dumps(provenance), flush=True)
    original_save = train.atomic_checkpoint
    def save_with_provenance(path, payload):
        return original_save(path, {**payload, 'runtime_performance': provenance})
    train.atomic_checkpoint = save_with_provenance
    sys.argv = [train.__file__, *trainer_args]
    with execution_policy(cache_bridge_masks=True, checkpoint_motion=True):
        train.main()


if __name__ == '__main__':
    main()
