"""Train-only, caption-window-weighted Kimodo369 statistics for A2.

Run through run_ablation.sh A2 prepare on an allocated compute node.
Smoke/limited results are deliberately rejected by the training contract.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
import torch

import config
from ablation_settings import sha
from audit_motion_stats import _load_kept_windows
from kimodo_codec import cached_encode, dependency_identity
from uniego_layout import canonicalize_frame0, ground_features


def convert_sequence(item):
    uuid, windows = item
    torch.set_num_threads(1)
    total = np.zeros(369, dtype=np.float64)
    square = np.zeros(369, dtype=np.float64)
    count = 0
    path = Path(config.NYMERIA_UNIEGO_ROOT)/(uuid+'.npz')
    with np.load(path, allow_pickle=False) as archive:
        source = archive['features']
        source_sha = __import__('hashlib').sha256(source.tobytes()).hexdigest()
        for start, end, offset in windows:
            values = source[start:end].astype(np.float32)
            if len(values) != end-start:
                raise RuntimeError(f'Truncated training window: {uuid}@{start}:{end}')
            values = canonicalize_frame0(ground_features(values, offset))
            converted = cached_encode(torch.from_numpy(values), config.FPS).numpy().astype(np.float64)
            total += converted.sum(axis=0)
            square += np.square(converted).sum(axis=0)
            count += len(converted)
    return total, square, count, len(windows), uuid, source_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-sequences', type=int, default=None)
    parser.add_argument('--progress-every', type=int, default=5)
    args = parser.parse_args()
    if args.workers < 1 or args.progress_every < 1 or (args.max_sequences is not None and args.max_sequences < 1):
        parser.error('workers/progress/max-sequences must be positive')
    if not os.environ.get('SLURM_JOB_ID') or 'login' in os.uname().nodename or os.uname().nodename.split('.')[0] == 'cs-gpu-01':
        raise RuntimeError('Use an allocated compute node')
    # Validate the original data/stat/model identities before changing representation.
    config.validate_artifacts()
    args.out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (args.out/'.preparation.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if any((args.out/name).exists() for name in ('mean.npy','std.npy','summary.json')):
        raise FileExistsError('Use a fresh output directory; prepared statistics are immutable')
    population_args = argparse.Namespace(split='train', manifest=config.NYMERIA_MANIFEST,
        split_file=config.NYMERIA_SPLIT_FILE, calibration=config.FLOOR_CALIBRATION_JSON)
    sequences, raw, dropped = _load_kept_windows(population_args)
    items = sorted(sequences.items())
    source_count = sum(len(windows) for _, windows in items)
    source_frames = sum(end-start for _, windows in items for start,end,_ in windows)
    expected = dict(config.EXPECTED_STATS_POPULATION)
    population = dict(split='train', raw_usable_captioned_windows=raw,
                      dropped_windows=sum(dropped.values()), kept_windows_after_floor_filter=source_count,
                      runtime_guard_rejected=0, stats_windows=source_count, frames=source_frames)
    if population != expected:
        raise RuntimeError(f'Source train population changed: {population} != {expected}')
    limited = args.max_sequences is not None
    if limited:
        items = items[:args.max_sequences]
    summary = dict(codec='kimodo369_v1', split='train', limited=limited, complete=False,
        fps=config.FPS, contacts='original_source_labels',
        canonicalization='source calibrated camhead; no second canonicalization',
        population=population, codec_sha256=sha(Path(__file__).with_name('kimodo_codec.py')),
        dependencies=dependency_identity(),
        source_sha256={key:sha(path) for key,path in dict(mean=config.MOTION_STATS_MEAN,
            std=config.MOTION_STATS_STD, manifest=config.NYMERIA_MANIFEST,
            split=config.NYMERIA_SPLIT_FILE, calibration=config.FLOOR_CALIBRATION_JSON).items()},
        source_summary_sha256=sha(config.MOTION_STATS_SUMMARY),
        source_representation=config.MOTION_REPRESENTATION,
        weighting='retained Nymeria train caption windows; repeated frames counted per window; no runtime z guard in stats')
    def write(name, value):
        path=args.out/name
        temporary=path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
        temporary.replace(path)
    write('progress.json',summary)
    total=np.zeros(369,np.float64); square=total.copy(); count=windows=0; fingerprints={}
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for index,(s1,s2,n,nw,uuid,digest) in enumerate(pool.map(convert_sequence,items),1):
            total+=s1; square+=s2; count+=n; windows+=nw; fingerprints[uuid]=digest
            if index%args.progress_every==0 or index==len(items):
                status=dict(sequences_done=index,sequences_total=len(items),frames_done=count,
                            windows_done=windows,elapsed_seconds=time.monotonic()-started)
                write('progress.json',status)
                print(json.dumps(status,sort_keys=True),flush=True)
    if not count or (not limited and (count != source_frames or windows != source_count)):
        raise RuntimeError('Incomplete statistics population')
    mean=total/count
    std=np.sqrt(np.maximum(square/count-np.square(mean),0))
    std[std<1e-6]=1
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise RuntimeError('Nonfinite statistics')
    for name,value in [('mean',mean),('std',std)]:
        path=args.out/(name+'.npy')
        with path.with_suffix('.npy.tmp').open('wb') as handle:
            np.save(handle,value.astype(np.float32))
        path.with_suffix('.npy.tmp').replace(path)
        summary[name+'_sha256']=sha(path)
    write('source_sequences.json',fingerprints)
    summary.update(complete=True,actual_frames=count,actual_windows=windows,
                   sequences=len(items),source_sequences_sha256=sha(args.out/'source_sequences.json'),
                   elapsed_seconds=time.monotonic()-started)
    write('summary.json',summary)
    print(json.dumps(summary,indent=2,sort_keys=True),flush=True)


if __name__ == '__main__':
    main()
