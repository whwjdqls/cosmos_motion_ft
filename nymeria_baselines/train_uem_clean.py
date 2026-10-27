"""Acceptance gate followed by the unchanged native UniEgoMotion entrypoint."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    audit = json.loads((a.cache / 'acceptance.json').read_text())
    assert audit['phase3_matched_windows'] == 112503
    assert audit['phase3_max_normalized_difference'] <= 1e-4
    if a.out.exists():
        raise FileExistsError('Fresh-run output already exists: ' + str(a.out))
    sys.path.insert(0, str(a.upstream))
    import torch
    from dataset.nymeria_soma_dataset import NymeriaSOMADataset
    for split in ('train', 'val', 'test'):
        data = NymeriaSOMADataset(str(a.cache), split)
        for index in (0, len(data)//2, len(data)-1):
            sample = data[index]
            motion, traj = sample['misc']['motion'], sample['y']['traj']
            assert motion.shape == (97, 230) and traj.shape == (97, 18)
            assert torch.isfinite(motion).all() and torch.isfinite(traj).all()
        print(f'Clean UniEgoMotion {split} loader acceptance passed: {len(data)} windows', flush=True)
    os.chdir(a.upstream)
    args = [sys.executable, '-u', 'run/train_uem.py', 'CONFIG', 'config/nymeria_soma.yaml',
            'DATA.DATA_DIR', str(a.cache), 'TRAIN.EXP_PATH', str(a.out)]
    os.execv(sys.executable, args)


if __name__ == '__main__':
    main()
