"""Require a scheduler allocation or explicitly selected Foundary GPUs."""
from __future__ import annotations

import getpass
import os
from pathlib import Path
import re
import socket


def require_compute_access() -> None:
    host = socket.gethostname().split('.')[0]
    if 'login' in host.lower() or host in {'cs-gpu-01', 'node37'}:
        raise RuntimeError(f'Phase 1 compute is forbidden on {host}')
    if os.environ.get('SLURM_JOB_ID'):
        if os.environ.get('SLURM_JOB_USER', getpass.getuser()) != getpass.getuser():
            raise RuntimeError('The allocation must belong to the current user')
        return
    if os.environ.get('NATIVEP1_FOUNDRY_ADMISSION'):
        from native_phase_training.compute_admission import require_owned_compute
        require_owned_compute()
        return
    gpu_selection = os.environ.get('FOUNDRY_GPU_UUIDS', os.environ.get('FOUNDRY_GPU_UUID', ''))
    gpus = gpu_selection.split(',')
    storage = Path(os.environ.get('COSMOS_STORAGE_ROOT', '/'))
    foundary = (
        host in {'lg-gwangmo-4xh200-sep-1', 'lg-gwangmo-4xh200-sep-2'}
        and os.environ.get('COSMOS_SITE') == 'custom'
        and 1 <= len(gpus) <= 2
        and len(set(gpus)) == len(gpus)
        and all(re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', gpu) is not None for gpu in gpus)
        and os.environ.get('CUDA_VISIBLE_DEVICES') == gpu_selection
        and storage.resolve().is_relative_to('/mnt/local')
    )
    if not foundary:
        raise RuntimeError('Use an owned Slurm allocation or explicitly assigned Foundary GPU UUIDs and local storage')


if __name__ == '__main__':
    require_compute_access()
