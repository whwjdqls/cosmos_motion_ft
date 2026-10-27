"""Build a separate floor-calibrated baseline cache on a compute node."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import time

import numpy as np

from nymeria_baselines.prepare import prepare_window


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ('old', 'out', 'phase3', 'manifest', 'floor', 'mean', 'std'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    host = socket.gethostname().split('.')[0]
    if 'login' in host or host in ('cs-gpu-01', 'node37') or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use an approved allocated compute node')
    if a.out.exists():
        raise FileExistsError(a.out)
    old = json.loads((a.old / 'index.json').read_text())
    reference = json.loads((a.phase3 / 'index.json').read_text())
    for key in ('manifest', 'floor', 'mean', 'std'):
        assert digest(getattr(a, key)) == reference['source'][key], key
    floor = json.loads(a.floor.read_text())
    mean, std = np.load(a.mean), np.load(a.std)
    assert np.isfinite(std).all() and (std > 0).all()
    records = {}
    for row in old['rows']:
        info = (row['array'], row['split'])
        assert row['uuid'] not in records or records[row['uuid']] == info
        records[row['uuid']] = info
    refs = defaultdict(list)
    for row in reference['rows']:
        refs[row['uuid']].append(row)
    temporary = a.out.with_name(a.out.name + '.building.' + str(os.getpid()))
    temporary.mkdir(parents=True)
    (temporary / 'records').symlink_to(a.old / 'records', target_is_directory=True)
    accepted, rejected = [], []
    dropped = Counter()
    sums = [np.zeros(283, np.float64), np.zeros(18, np.float64)]
    squares = [v.copy() for v in sums]
    frames, checked, max_diff = 0, 0, 0.
    started = time.monotonic()
    with a.manifest.open() as stream:
        for line in stream:
            rec = json.loads(line)
            uuid = rec.get('uuid')
            if uuid not in records:
                continue
            array, split = records[uuid]
            motion = np.load(a.old / 'records' / (array + '.motion.npy'), mmap_mode='r')
            pose = np.load(a.old / 'records' / (array + '.pose_yup.npy'), mmap_mode='r')
            drop = {(int(x[0]), int(x[1])): str(x[2]) if len(x) > 2 else 'unknown'
                    for x in floor['dropped_windows'].get(uuid, [])}
            refrows = defaultdict(list)
            refarray = None
            if refs[uuid]:
                refarray = np.load(a.phase3 / (refs[uuid][0]['array'] + '.motion.npy'), mmap_mode='r')
                for rr in refs[uuid]:
                    refrows[rr['start']].append(rr['row'])
            for w in rec.get('t2w_windows', []):
                if not w.get('usable', False) or not w.get('caption'):
                    continue
                ws, we = int(w['start_frame']), int(w['end_frame'])
                reason = drop.get((ws, we))
                if reason is not None:
                    dropped[split + ':' + reason] += 1
                    continue
                off_gt = w.get('ground_offset_y')
                delta = float(floor['deltas'].get(uuid, floor['global_delta'])) if off_gt is not None else 0.
                off = float(off_gt) + delta if off_gt is not None else None
                for start in range(ws, min(we, int(rec['nb_frames'])) - 96, 97):
                    row = dict(uuid=uuid, array=array, split=split, start=start,
                               ground_offset_y=off, ground_offset_y_uncalibrated=off_gt,
                               floor_delta=delta, caption=w['caption'], parent_window=[ws, we])
                    m, c = prepare_window(motion[start:start+97].copy(), pose[start:start+97].copy(), off)
                    normalized = (m - mean) / std
                    if m.shape != (97, 283) or c.shape != (97, 18):
                        raise RuntimeError(('Invalid window', row, m.shape, c.shape))
                    if not np.isfinite(normalized).all() or np.abs(normalized).max() > 20:
                        rejected.append(dict(row, reason='non-finite or extreme normalized motion'))
                        continue
                    if not np.isfinite(c).all():
                        raise RuntimeError(('Non-finite camera', row))
                    if split != 'test':
                        candidates = refrows[start]
                        if not candidates:
                            raise RuntimeError(('Not in Phase3 accepted windows', row))
                        errors = [float(np.max(np.abs(normalized - refarray[i]))) for i in candidates]
                        best = int(np.argmin(errors))
                        error = errors[best]
                        if error > 1e-4:
                            raise RuntimeError(('Phase3 feature mismatch', error, row))
                        candidates.pop(best)
                        max_diff = max(max_diff, error)
                        checked += 1
                    accepted.append(row)
                    if split == 'train':
                        for j, value in enumerate((m, c)):
                            value = value.astype(np.float64)
                            sums[j] += value.sum(axis=0)
                            squares[j] += np.square(value).sum(axis=0)
                        frames += 97
            if split != 'test' and any(refrows.values()):
                raise RuntimeError(('Missing Phase3 windows', uuid))
            print(f'record={uuid} accepted={len(accepted)} rejected={len(rejected)} elapsed={time.monotonic()-started:.1f}s', flush=True)
    actual = Counter((r['uuid'], r['start']) for r in accepted if r['split'] != 'test')
    expected = Counter((r['uuid'], r['start']) for r in reference['rows'])
    assert actual == expected, 'Full Phase3 cohort including duplicate windows must match'
    old_train = {r['uuid'] for r in old['rows'] if r['split'] == 'train'}
    new_train = {r['uuid'] for r in accepted if r['split'] == 'train'}
    assert old_train == new_train, 'Shape PCA reuse requires identical training recording set'
    stats = {'train_frames': np.array(frames)}
    for j, name in enumerate(('motion', 'camera')):
        mu = sums[j] / frames
        sigma = np.sqrt(np.maximum(squares[j] / frames - mu * mu, 0))
        stats[name + '_mean'] = mu.astype(np.float32)
        stats[name + '_std'] = np.maximum(sigma, 1e-6).astype(np.float32)
    np.savez(temporary / 'train_stats.npz', **stats)
    shutil.copy2(a.old / 'body_shape16.npz', temporary / 'body_shape16.npz')
    result = dict(old)
    result.update(rows=accepted, windows=dict(Counter(r['split'] for r in accepted)), rejected=rejected,
                  floor_calibration_sha256=digest(a.floor), phase3_fingerprint=reference['fingerprint'],
                  source_cache=str(a.old), corrected_floor=True)
    (temporary / 'index.json').write_text(json.dumps(result, indent=2) + '\n')
    metadata = temporary / 'egoallo_metadata_v1'
    metadata.mkdir()
    for source in (a.old / 'egoallo_metadata_v1').glob('*.npy'):
        (metadata / source.name).symlink_to(source)
    mi = json.loads((a.old / 'egoallo_metadata_v1' / 'index.json').read_text())
    mi['source_manifest_sha256'] = digest(temporary / 'index.json')
    (metadata / 'index.json').write_text(json.dumps(mi, indent=2) + '\n')
    audit = dict(host=host, job=os.environ['SLURM_JOB_ID'], counts=result['windows'],
                 dropped_parent_windows=dict(dropped), rejected_windows=len(rejected),
                 phase3_matched_windows=checked, phase3_max_normalized_difference=max_diff,
                 train_frames=frames, shape_pca='unchanged train-only recording set',
                 statistics='recomputed using accepted training windows only',
                 split_note='578 train / 64 validation / 71 test; train+val matches Phase3 cohort',
                 elapsed_seconds=time.monotonic()-started)
    (temporary / 'acceptance.json').write_text(json.dumps(audit, indent=2) + '\n')
    os.rename(temporary, a.out)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
