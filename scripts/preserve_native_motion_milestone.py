"""Preserve a complete native DCP milestone as an independent local copy."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import signal
import time


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate(path, step):
    import torch.distributed.checkpoint as dcp
    path = Path(path)
    assert not (path / '.deleting').exists()
    metadata = {}
    for component in ['model', 'trainer', 'optim', 'scheduler']:
        with (path / component / '.metadata').open('rb') as stream:
            meta = pickle.load(stream)
        assert meta.state_dict_metadata and meta.storage_data
        for item in meta.storage_data.values():
            relative = Path(item.relative_path)
            assert not relative.is_absolute() and '..' not in relative.parts
            assert (path / component / relative).stat().st_size >= item.offset + item.length
        metadata[component] = meta
    model_keys = metadata['model'].state_dict_metadata
    assert any(k.startswith('net.') and 'lora_A' in k for k in model_keys)
    assert any(k.startswith('net_ema.') and 'lora_A' in k for k in model_keys)
    assert any(k.startswith('net_ema.') and 'llm2action' in k for k in model_keys)
    trainer_keys = metadata['trainer'].state_dict_metadata
    assert any(k.startswith('rng_state_0.') for k in trainer_keys)
    assert any(k.startswith('rng_state_1.') for k in trainer_keys)
    state = {'iteration': 0}
    dcp.load(state, checkpoint_id=path / 'trainer')
    assert state['iteration'] == step
    scheduler = {'last_epoch': 0}
    dcp.load(scheduler, checkpoint_id=path / 'scheduler')
    assert scheduler['last_epoch'] == step
    return dict(step=step, raw_and_ema=True, optimizer=True, scheduler=True,
                rng_ranks=2, components={k: len(v.state_dict_metadata) for k, v in metadata.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source-operation', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--step', type=int, default=100000)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    for path in [args.run, args.source_operation, args.root]:
        assert path.resolve().is_relative_to('/mnt/local')
    source = args.run / 'checkpoints' / f'iter_{args.step:09d}'
    if args.validate_only:
        print(json.dumps(validate(source, args.step)))
        return
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    args.root.mkdir(parents=True, exist_ok=True)
    lock = (args.root / 'watcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save(args.root / 'WATCHER.json', dict(pid=os.getpid(), node=os.uname().nodename,
         start=Path('/proc', str(os.getpid()), 'stat').read_text().rsplit(')', 1)[1].split()[19],
         source=str(source), time=time.time()))
    try:
        while True:
            try:
                checks = validate(source, args.step)
                break
            except (FileNotFoundError, EOFError, AssertionError, RuntimeError) as error:
                save(args.root / 'status.json', dict(stage='waiting_for_complete_exact_milestone',
                     step=args.step, source=str(source), reason=type(error).__name__, time=time.time()))
                time.sleep(30)
        files = sorted(p for p in source.rglob('*') if p.is_file())
        signatures = {str(p.relative_to(source)): (p.stat().st_size, p.stat().st_mtime_ns) for p in files}
        assert shutil.disk_usage(args.root).free > sum(p.stat().st_size for p in files) + 2**30
        target = args.root / source.name
        temporary = args.root / (source.name + '.partial')
        if not target.exists():
            temporary.mkdir(exist_ok=True)
            for src in files:
                dst = temporary / src.relative_to(source)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                with dst.open('rb') as stream:
                    os.fsync(stream.fileno())
            validate(temporary, args.step)
            temporary.replace(target)
        hashes = {}
        for src in files:
            rel = str(src.relative_to(source)); dst = target / rel
            assert (src.stat().st_size, src.stat().st_mtime_ns) == signatures[rel]
            assert src.stat().st_ino != dst.stat().st_ino and dst.stat().st_nlink == 1
            hashes[rel] = sha(src)
            assert sha(dst) == hashes[rel]
        validate(target, args.step)
        for name in ['motion_contract.json', 'config.yaml', 'launch_info.yaml', 'job_env.yaml']:
            shutil.copyfile(args.run / name, args.root / name)
        manifest = json.loads((args.source_operation / 'source_manifest.json').read_text())
        for rel, digest in manifest.items():
            src = args.source_operation / 'source_snapshot' / rel
            assert sha(src) == digest
            dst = args.root / 'source_snapshot' / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        save(args.root / 'SHA256SUMS.json', hashes)
        save(args.root / 'source_manifest.json', manifest)
        result = dict(passed=True, independent_copy=True, checkpoint=str(target),
                      source=str(source), bytes=sum(p.stat().st_size for p in files),
                      source_files_verified=len(manifest), **checks, time=time.time())
        save(args.root / 'COMPLETE.json', result)
        save(args.root / 'status.json', dict(stage='preserved_verified', **result))
    except BaseException as error:
        save(args.root / 'FAILED.json', dict(type=type(error).__name__, error=str(error), time=time.time()))
        raise


if __name__ == '__main__':
    main()
