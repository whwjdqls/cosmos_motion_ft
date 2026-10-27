"""Shared, resumable native DINOv2 CLS features for Nymeria UniEgoMotion.

Run only on allocated compute nodes. Multiple workers cooperate through
per-recording locks; checkpoints, source videos, and motion caches are read-only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

import numpy as np
import torch


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(4 << 20), b''):
            value.update(block)
    return value.hexdigest()


def require_compute():
    host = socket.gethostname().split('.')[0]
    if not os.environ.get('SLURM_JOB_ID') or not host.startswith('node') or host == 'node37':
        raise RuntimeError('Use an owned allowed compute node, never login/cs-gpu-01/node37')


def frame_ids(start):
    frames = np.arange(start, start + 97, dtype=np.int64)
    # Causal hold at5fps. Window frame0 is its actual observed RGB, not a
    # pre-window frame. No selected visual feature sees a future RGB frame.
    return np.maximum(start, frames // 4 * 4)


def load_dino(out):
    assets = out / 'assets'
    assets.mkdir(exist_ok=True)
    with (assets / 'download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        commit_file = assets / 'dinov2_commit.json'
        if not commit_file.exists():
            request = urllib.request.Request('https://api.github.com/repos/facebookresearch/dinov2/commits/main',
                                             headers={'User-Agent': 'nymeria-uniegomotion'})
            with urllib.request.urlopen(request, timeout=60) as stream:
                commit = json.load(stream)
            atomic_json(commit_file, {'sha': commit['sha'], 'source': commit['html_url']})
        commit = json.loads(commit_file.read_text())['sha']
        torch.hub.set_dir(str(assets / 'torch_hub'))
        model = torch.hub.load(f'facebookresearch/dinov2:{commit}', 'dinov2_vitl14_reg',
                               pretrained=True, trust_repo=True, skip_validation=True)
    return model.cuda().eval(), commit


def exact_read(stream, size):
    parts, remaining = [], size
    while remaining:
        piece = stream.read(remaining)
        if not piece:
            raise RuntimeError(f'Video decoder ended early with {remaining} bytes missing')
        parts.append(piece)
        remaining -= len(piece)
    return b''.join(parts)


def extract_record(args, model, transform, record, fingerprint):
    from PIL import Image
    key, uid, needed = record['key'], record['uuid'], record['needed']
    video = args.video_root / (uid + '.mp4')
    info = record['video']
    if info['framerate'] != 20 or info['width'] != info['height']:
        raise ValueError(f'Expected upright square20fps Nymeria video: {uid}')
    if needed[-1] >= info['nb_frames']:
        raise ValueError(f'RGB shorter than accepted motion windows: {uid}')
    extra = needed[needed % 4 != 0]
    selected = np.unique(np.concatenate((np.arange(0, int(needed[-1])+1, 4), extra)))
    # Balance the sum so FFmpeg's expression parser does not exhaust its
    # recursion limit on recordings with many off-grid window starts.
    terms = ['eq(mod(n\\,4)\\,0)'] + [f'eq(n\\,{int(n)})' for n in extra]
    while len(terms) > 1:
        terms = [f'({terms[i]}+{terms[i+1]})' if i+1 < len(terms) else terms[i]
                 for i in range(0, len(terms), 2)]
    expr = terms[0]
    prefix = args.output / 'records' / key
    temporary = Path(str(prefix) + f'.{os.getpid()}.building.npy')
    features = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float16,
                                        shape=(len(needed), 1024))
    batch, destinations, count = [], [], 0
    def flush():
        nonlocal count
        if not batch:
            return
        with torch.inference_mode():
            values = model.forward_features(torch.stack(batch).cuda())['x_norm_clstoken']
        if values.shape != (len(batch), 1024) or not torch.isfinite(values).all():
            raise RuntimeError('Invalid native DINOv2 features')
        features[destinations] = values.to(torch.float16).cpu().numpy()
        count += len(batch)
        batch.clear()
        destinations.clear()
    started = time.monotonic()
    with Path(str(prefix) + '.ffmpeg.log').open('w') as errors:
        command = [args.ffmpeg, '-nostdin', '-v', 'error', '-threads', '2', '-i', str(video),
                   '-vf', f'trim=end_frame={int(needed[-1])+1},select={expr}',
                   '-fps_mode', 'passthrough', '-threads', '2', '-an', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors) as process:
            try:
                wanted = {int(n): i for i, n in enumerate(needed)}
                for n in selected:
                    pixels = exact_read(process.stdout, int(info['width'])*int(info['height'])*3)
                    if int(n) in wanted:
                        image = Image.frombytes('RGB', (int(info['width']), int(info['height'])), pixels)
                        batch.append(transform(image))
                        destinations.append(wanted[int(n)])
                        if len(batch) == args.batch_size:
                            flush()
                flush()
                if process.stdout.read(1):
                    raise RuntimeError('Decoder emitted unexpected extra selected frames')
                if process.wait() != 0:
                    raise RuntimeError(f'FFmpeg failed: {uid}')
            except BaseException:
                process.terminate()
                process.wait()
                raise
    assert count == len(needed)
    features.flush()
    del features
    final = Path(str(prefix) + '.features.npy')
    os.replace(temporary, final)
    np.save(str(prefix) + '.frame_ids.npy', needed)
    stat = video.stat()
    metadata = dict(uuid=uid, key=key, fingerprint=fingerprint, frames=count,
                    source_video=str(video), video_bytes=stat.st_size, video_mtime_ns=stat.st_mtime_ns,
                    features_sha256=sha(final), seconds=time.monotonic()-started)
    atomic_json(str(prefix) + '.json', metadata)
    print(json.dumps(dict(stage='record_complete', **metadata)), flush=True)


def prepare(args):
    require_compute()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'records').mkdir(exist_ok=True)
    index = json.loads((args.cache / 'index.json').read_text())
    grouped = defaultdict(list)
    for row in index['rows']:
        grouped[row['uuid']].append(row)
    manifest = {}
    with (args.video_root / 'manifest_video.jsonl').open() as source:
        for line in source:
            row = json.loads(line)
            if row['uuid'] in grouped:
                manifest[row['uuid']] = row
    records = []
    for uid, rows in sorted(grouped.items()):
        needed = np.unique(np.concatenate([frame_ids(int(r['start'])) for r in rows]))
        records.append(dict(uuid=uid, key=rows[0]['array'], needed=needed, video=manifest[uid]))
    print(json.dumps(dict(stage='feature_plan', recordings=len(records),
                          unique_rgb_frames=sum(len(r['needed']) for r in records),
                          feature_bytes=sum(len(r['needed'])*1024*2 for r in records))), flush=True)
    model, commit = load_dino(args.output)
    from torchvision import transforms as T
    transform = T.Compose([T.Resize(336, interpolation=T.InterpolationMode.BICUBIC),
                           T.ToTensor(), T.Normalize((.485, .456, .406), (.229, .224, .225))])
    policy = dict(schema=1, cache=str(args.cache.resolve()), cache_index_sha256=sha(args.cache/'index.json'),
                  model='dinov2_vitl14_reg', source_commit=commit, tokens='x_norm_clstoken only, 1024D',
                  preprocessing='Native PIL bicubic shorter-side336, RGB, ImageNet normalization; already-upright Nymeria video; no extra rotation/crop',
                  sampling='Global5fps causal hold at20fps; exact observed window-start image; no future RGB',
                  precision='FP32 encoder inference, FP16 feature storage as upstream',
                  recordings=len(records), unique_rgb_frames=sum(len(r['needed']) for r in records))
    fingerprint = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
    policy['fingerprint'] = fingerprint
    with (args.output/'policy.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = args.output/'index.json'
        if path.exists() and json.loads(path.read_text()) != policy:
            raise RuntimeError('Feature-cache identity changed; use a fresh output')
        atomic_json(path, policy)
    # Different workers start at different offsets, reducing lock contention.
    offset = int(os.environ.get('SLURM_JOB_ID', '0')) % len(records)
    order = records[offset:] + records[:offset]
    while True:
        pending = 0
        for record in order:
            prefix = args.output/'records'/record['key']
            done = Path(str(prefix)+'.json')
            if done.exists():
                assert json.loads(done.read_text())['fingerprint'] == fingerprint
                continue
            pending += 1
            with Path(str(prefix)+'.lock').open('a') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if not done.exists():
                    extract_record(args, model, transform, record, fingerprint)
            atomic_json(args.output/f'progress_{os.environ["SLURM_JOB_ID"]}.json',
                        dict(stage='extracting', last_record=record['uuid'], time=time.time()))
        if pending == 0:
            atomic_json(args.output/'READY.json', policy)
            print(json.dumps(dict(stage='features_ready', **policy)), flush=True)
            return
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--video-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--batch-size', type=int, default=16)
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
