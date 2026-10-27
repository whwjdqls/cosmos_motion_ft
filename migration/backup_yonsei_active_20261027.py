#!/usr/bin/env python3
"""Additive, restartable Drive snapshot of the six active Yonsei runs.

Run only inside an owned compute allocation. No training/model code is imported.
Archives retain nanosecond mtimes required by existing data-cache identities.
"""
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

REPO = Path('/home/whwjdqls99/cosmos_motion_ft')
WEKA = Path('/lustre/whwjdqls99/cosmos/weka')
SCRATCH = Path('/scratch2/whwjdqls99/cosmos')
DATA = WEKA / 'nymeriaplus_kimodo_proportional'
RUNS = SCRATCH / 'cosmos_motion_ft_runs'
OPS = SCRATCH / 'operations/gdrive_training_backup_20261027'
REMOTE = 'gggdrive:cosmos_data/yonsei_training_20261027'
P1 = WEKA / 'cosmos_motion_ft_runs/cosmos3_camera/camera_world_edge/edge_phase1_T97_20fps_4gpu_bs8_camera_wearer_lora_exclude5_yonsei_20260905'
P2 = RUNS / 'motion_expert_t2m_edge/edge_7layer_self_cross_final_hidden_knorm_captiononly_nymeria_bones50_t2m_ti2m_nanoparity_gbs256_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260921'
STATS = WEKA / 'cosmos_motion_ft_runs/nymeria_camera_head_recanonicalization_v1/stats'
LATENTS = DATA / 'joint_latents_T97_256tier_256'
EVAL = RUNS / 'motion_expert_phase3_edge/edge_p3_p1ema200k_crosscaption200k_bridge7_T97_gbs32_wv10_yonsei_20260924/eval_heldout71_step_000200000_unipc35_20261002'
ACTIVE = {
    'p3_trajectory_film': RUNS / 'motion_expert_phase3_edge/edge_p3_p1ema200k_crosscaption200k_bridge7_fourtask_balanced_notext_floor18d_motionnoise_camtrajfilm_T97_gbs32_wv10_yonsei_20261023',
    'p3_hidden_film': RUNS / 'motion_expert_phase3_edge/edge_p3_p1ema200k_crosscaption200k_bridge7_fourtask_equal_notext_contextcamfilm_T97_gbs32_wv10_yonsei_20261021',
    'p3_control': RUNS / 'motion_expert_phase3_edge/edge_p3_p1ema200k_crosscaption200k_bridge7_fourtask_equal_notext_T97_gbs32_wv10_yonsei_20261020',
    'p1_prefix1': RUNS / 'cosmos3_camera/camera_world_edge/edge_phase1_T97_20fps_2gpu_bs16_lora_exclude5_fd_invdyn_no_text_p1_200k_yonsei_20261017',
    'egocontrol': RUNS / 'cosmos3_camera/egocontrol_edge/egocontrol_edge_7cond_m2v_p1ema200k_phase1recipe_T97_gbs32_200k_yonsei_20261026',
    'p1_action_only': RUNS / 'cosmos3_camera/camera_world_edge/edge_phase1_T97_20fps_2gpu_bs16_action_only_exclude5_fd_invdyn_no_text_p1_200k_yonsei_20261027',
}
META_NAMES = ('contract.json', 'sources.json', 'config.yaml', 'native_phase1_contract.json', 'egocontrol_contract.json')


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def status(phase, **extra):
    row = {"time": now(), "host": socket.gethostname(), "phase": phase, **extra}
    write_json(OPS / 'progress.json', row)
    print(json.dumps(row), flush=True)


def upload(path, relative):
    subprocess.run([
        '/usr/bin/rclone', 'copyto', str(path), REMOTE + '/' + relative,
        '--immutable', '--checksum', '--transfers', '1', '--checkers', '2',
        '--tpslimit', '4', '--tpslimit-burst', '4', '--drive-chunk-size', '64M',
        '--buffer-size', '16M', '--bwlimit', '16M', '--stats', '30s',
        '--stats-one-line', '--log-level', 'INFO', '--retries', '3',
        '--low-level-retries', '10', '--contimeout', '20s', '--timeout', '5m',
        '--drive-stop-on-upload-limit',
    ], check=True)


def metadata(run):
    return [run / name for name in META_NAMES if (run / name).is_file()]


def make_plan():
    assets = []
    def add(name, sources, purpose):
        assets.append(dict(name=name, sources=[str(p) for p in sources], purpose=purpose))
    add('001_normalization_and_contracts', [STATS, DATA / 'train_test_split.json',
        LATENTS / 'latent_cache_contract.json', LATENTS / 'latent_cache_complete.json']
        + sum((metadata(r) for r in list(ACTIVE.values()) + [P1, P2]), []),
        'Exact camhead statistics, split, latent contracts and active/frozen run configurations')
    add('002_dataset_metadata', [DATA / 'metadata', DATA / 'video/manifest_video.jsonl'],
        'Current metadata, calibration, quality filters and video index; preserve source identities')
    add('003_camera_rgb_source_identity', [DATA / 'camera_rgb'],
        'Current source bytes and nanosecond mtimes used by camera-cache validation')
    for i, key in enumerate(ACTIVE, 4):
        assets.append(dict(name=f'{i:03d}_recovery_{key}', snapshot_run=str(ACTIVE[key]),
            purpose='One complete checkpoint selected and pinned when this item starts; not a continuous mirror'))
    add('010_frozen_phase2_crosscaption200k', [P2 / 'checkpoints/step_000200000.pt'],
        'Frozen Edge self-cross final-hidden/K-norm/caption-only B256 phase2 specialist')
    add('011_phase3_motion_cache', [SCRATCH / 'caches/phase3_edge_camhead_T97_v1'],
        '112503-window motion/neutral cache and exact index')
    add('012_phase3_camera_cache', [SCRATCH / 'caches/phase3_edge_camera_conditioning_T97_revalidated_20261021'],
        'Native transition camera conditioning')
    add('013_phase3_trajectory18_cache', [SCRATCH / 'caches/phase3_edge_camera_trajectory18_T97_camfloor_v1_20261023'],
        'Floor-aware 18D conditioning and exact cache identities')
    add('014_recanonicalized_motion', [DATA / 'uniego_rep_camhead_v1'],
        'Original recanonicalized source files required by cache identity checks')
    for subject in [f'S{i:02d}' for i in range(1, 21) if i != 18]:
        add('020_wan_latents_' + subject, [LATENTS / subject], 'Native T97 256-tier Wan latents and camera actions')
    add('030_frozen_phase1_ema200k', [P1 / 'checkpoints/iter_000200000'],
        'Complete frozen Edge phase1 DCP; contains EMA used by phase3 and EgoControl')
    add('031_fixed_validation_inputs', [WEKA / 'cosmos_motion_ft_runs/native_phase1_eval_inputs_full71_256_T97_v2',
        EVAL / 'configuration.json', EVAL / 'inputs'], 'Full71 inputs used by active validation callbacks')
    add('040_cosmos3_edge_assets', [WEKA / 'Cosmos3-Edge'], 'Base Edge model/tokenizer assets; cache download bookkeeping excluded')
    add('041_cosmos3_edge_native_dcp', [WEKA / 'cosmos3_edge_dcp'], 'Native base Edge DCP')
    add('042_wan22_vae', [WEKA / 'wan22_vae/Wan2.2_VAE.pth'], 'Wan VAE used by model loading and validation')
    return dict(created=now(), remote=REMOTE, restore_root='/', archive_format='GNU tar PAX, nanosecond mtime retained',
        audit_scope='Documented cosmos_data, cosmos_ckpts and cosmos_models roots; historical raw trees not rehashed',
        drive_free_at_audit_bytes=2148923952344,
        excluded='BONES (not used by these active runs), raw RGB videos/VRS, unrelated experiments, environments, credentials and generated visualizations',
        assets=assets)


def pin_checkpoint(item):
    """Hardlink one published checkpoint on Scratch2 before retention can unlink it."""
    run = Path(item['snapshot_run'])
    pin = OPS / 'pins' / item['name']
    manifest = pin / 'snapshot.json'
    if manifest.is_file():
        return json.loads(manifest.read_text())
    pin.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        root = pin / f'attempt_{attempt}'
        root.mkdir(exist_ok=True)
        try:
            cdir = run / 'checkpoints'
            pointer = cdir / 'latest.json'
            if pointer.is_file():
                raw = pointer.read_bytes()
                ref = json.loads(raw)
                checkpoint = cdir / ref['file']
                step = ref['step']
            else:
                pointer = cdir / 'latest_checkpoint.txt'
                raw = pointer.read_bytes()
                checkpoint = cdir / raw.decode().strip()
                step = int(checkpoint.name.split('_')[-1])
            if checkpoint.parent != cdir or not checkpoint.exists():
                raise ValueError('Unexpected or missing checkpoint pointer: ' + str(checkpoint))
            dst = root / checkpoint.relative_to('/')
            dst.parent.mkdir(parents=True, exist_ok=True)
            if checkpoint.is_dir():
                shutil.copytree(checkpoint, dst, copy_function=os.link)
                if not list(dst.rglob('.metadata')):
                    raise ValueError('Native checkpoint has no DCP metadata: ' + str(checkpoint))
            else:
                os.link(checkpoint, dst)
            if pointer.read_bytes() != raw:
                raise RuntimeError('Latest checkpoint changed during snapshot; retrying')
            members = [str(checkpoint.relative_to('/'))]
            for source in metadata(run):
                target = root / source.relative_to('/')
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                members.append(str(source.relative_to('/')))
            ptarget = root / pointer.relative_to('/')
            ptarget.parent.mkdir(parents=True, exist_ok=True)
            ptarget.write_bytes(raw)
            members.append(str(pointer.relative_to('/')))
            result = dict(root=str(root), members=members, checkpoint=str(checkpoint), step=step, pinned_at=now())
            write_json(manifest, result)
            return result
        except (OSError, RuntimeError, ValueError) as exc:
            shutil.rmtree(root)
            if attempt == 4:
                raise
            print('Retry checkpoint pin:', exc, flush=True)
            time.sleep(2)


def archive_and_upload(item):
    name = item['name']
    receipt_path = OPS / 'receipts' / (name + '.json')
    if receipt_path.is_file():
        saved = json.loads(receipt_path.read_text())
        if saved.get('upload_complete'):
            return saved
    root = '/'
    snap = None
    if 'snapshot_run' in item:
        snap = pin_checkpoint(item)
        root, members = snap['root'], snap['members']
    else:
        sources = [Path(p) for p in item['sources']]
        missing = [str(p) for p in sources if not p.exists()]
        if missing:
            raise FileNotFoundError('Required source missing: ' + ', '.join(missing))
        members = [str(p.relative_to('/')) for p in sources]
    target = OPS / 'staging' / (name + '.tar')
    if not target.is_file():
        status('archiving', asset=name)
        partial = target.with_suffix('.partial.tar')
        subprocess.run(['/usr/bin/tar', '--format=pax', '--pax-option=delete=atime,delete=ctime',
            '--sort=name', '--dereference', '--hard-dereference', '--exclude=.cache',
            '--exclude=__pycache__', '--exclude=*.lock', '--exclude=_logs',
            '-C', root, '-cf', str(partial), '--', *members], check=True)
        partial.replace(target)
    status('hashing', asset=name, bytes=target.stat().st_size)
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with target.open('rb') as src:
        while block := src.read(8 * 1024 * 1024):
            sha.update(block)
            md5.update(block)
    receipt = dict(asset=item, archive=target.name, bytes=target.stat().st_size,
        sha256=sha.hexdigest(), md5=md5.hexdigest(), restore_root='/', members=members,
        snapshot=snap, remote=REMOTE + '/archives/' + target.name, upload_complete=False)
    status('uploading', asset=name, bytes=receipt['bytes'])
    upload(target, 'archives/' + target.name)
    receipt['upload_complete'] = True
    receipt['completed_at'] = now()
    candidate = OPS / 'receipts' / (name + '.pending.json')
    write_json(candidate, receipt)
    upload(candidate, 'receipts/' + name + '.json')
    candidate.replace(receipt_path)
    status('asset_complete', asset=name, bytes=receipt['bytes'])
    target.unlink()
    if snap:
        shutil.rmtree(OPS / 'pins' / name)
    return receipt


def main():
    host = socket.gethostname().split('.')[0]
    if not os.environ.get('SLURM_JOB_ID') or 'login' in host or host in ('cs-gpu-01', 'node37'):
        raise RuntimeError('Use an owned permitted compute allocation')
    for d in ('staging', 'receipts', 'pins', 'metadata'):
        (OPS / d).mkdir(parents=True, exist_ok=True)
    lock = (OPS / 'worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.nice(10)
    plan_path = OPS / 'plan.json'
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text())
    else:
        plan = make_plan()
        write_json(plan_path, plan)
    status('uploading_plan', total_assets=len(plan['assets']), remote=REMOTE)
    upload(plan_path, 'plan.json')
    for name in ('UPLOAD_MANIFEST.md', 'UPLOAD_PATH_MAP.md', 'migration/backup_yonsei_active_20261027.py'):
        target = OPS / 'metadata' / Path(name).name
        if not target.exists():
            shutil.copy2(REPO / name, target)
        upload(target, 'metadata/' + target.name)
    results, failures = [], []
    for item in plan['assets']:
        try:
            results.append(archive_and_upload(item))
        except Exception as exc:
            failure = dict(asset=item['name'], time=now(), error=str(exc))
            failures.append(failure)
            write_json(OPS / 'receipts' / (item['name'] + '.failed.json'), failure)
            status('asset_failed_continuing', **failure)
    report = dict(finished_at=now(), complete=not failures, completed=results, failures=failures)
    report_path = OPS / ('report_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json')
    write_json(report_path, report)
    upload(report_path, report_path.name)
    status('complete' if not failures else 'incomplete', completed_assets=len(results), failed_assets=len(failures),
        uploaded_archive_bytes=sum(r['bytes'] for r in results), report=str(report_path))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
