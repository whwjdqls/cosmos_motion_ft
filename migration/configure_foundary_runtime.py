"""Make the pinned TorchCodec wheel find this Conda environment's CUDA/FFmpeg libs.

TorchCodec 0.10 cu128 has no embedded library search path. The project launchers
clear LD_LIBRARY_PATH, and Foundary's system CUDA is 13.2. Add relative RPATHs
only to the new environment's TorchCodec libraries; no system files change.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import sysconfig


def main() -> None:
    prefix = Path(sys.prefix).resolve()
    if not prefix.is_relative_to('/mnt/local') or prefix.stat().st_uid != os.getuid():
        raise SystemExit('Run with Python from your own /mnt/local Conda environment')
    package = Path(sysconfig.get_path('purelib')) / 'torchcodec'
    libraries = sorted(package.glob('libtorchcodec_*.so'))
    if not libraries:
        raise SystemExit('Install the pinned TorchCodec wheel first')
    patchelf = prefix / 'bin/patchelf'
    additions = [
        '$ORIGIN', '$ORIGIN/../torch/lib', '$ORIGIN/../nvidia/npp/lib',
        '$ORIGIN/../nvidia/cuda_runtime/lib', '$ORIGIN/../nvidia/cuda_nvrtc/lib',
        '$ORIGIN/../../..',  # Conda prefix/lib, containing shared FFmpeg.
    ]
    for library in libraries:
        old = subprocess.check_output([patchelf, '--print-rpath', library], text=True).strip()
        paths = list(dict.fromkeys(additions + [p for p in old.split(':') if p]))
        subprocess.run([patchelf, '--force-rpath', '--set-rpath', ':'.join(paths), library], check=True)
    print(f'Configured relative runtime library paths for {len(libraries)} TorchCodec libraries')


if __name__ == '__main__':
    main()
