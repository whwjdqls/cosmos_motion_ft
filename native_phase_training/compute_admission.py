"""Validate an owned Slurm job or an explicit Foundary launcher admission."""
from __future__ import annotations

import getpass
import json
import os
from pathlib import Path
import socket


def require_owned_compute() -> None:
    host = socket.gethostname()
    if any(value in host for value in ('login', 'cs-gpu-01', 'node37')):
        raise RuntimeError('Training requires an owned compute allocation')
    if os.environ.get('SLURM_JOB_ID'):
        if os.environ.get('SLURM_JOB_USER', getpass.getuser()) != getpass.getuser():
            raise RuntimeError('The Slurm allocation belongs to another user')
        return
    if os.environ.get('COSMOS_SITE') != 'custom' or host not in (
        'lg-gwangmo-4xh200-sep-1', 'lg-gwangmo-4xh200-sep-2'
    ):
        raise RuntimeError('A Slurm allocation or explicit Foundary admission is required')
    path = Path(os.environ.get('NATIVEP1_FOUNDRY_ADMISSION', '/nonexistent'))
    if not path.is_file() or path.stat().st_uid != os.getuid():
        raise RuntimeError('Missing user-owned Foundary admission')
    admission = json.loads(path.read_text())
    if admission['host'] != host or admission['uid'] != os.getuid():
        raise RuntimeError('Foundary admission host/user mismatch')
    if admission['cuda_visible_devices'] != os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('Visible GPUs differ from the admitted GPU set')
    # The launcher must still be alive and be an ancestor of this worker.
    ancestor = os.getpid()
    while ancestor > 1 and ancestor != admission['launcher_pid']:
        stat = Path(f'/proc/{ancestor}/stat').read_text()
        ancestor = int(stat.rsplit(')', 1)[1].split()[1])
    if ancestor != admission['launcher_pid']:
        raise RuntimeError('The Foundary admission launcher is not an active ancestor')


if __name__ == '__main__':
    require_owned_compute()
