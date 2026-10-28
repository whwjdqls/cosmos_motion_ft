#!/usr/bin/env python3
"""Launch Phase-1 projection ablations on explicitly authorized free Foundary GPUs.

Source .site.local.sh and activate cosmos first. Uses two empty GPUs at 32 clips
each, or one empty GPU at 64 clips. An isolated four-update gradient/memory gate
must pass before a fresh 200k-step online W&B run starts.
"""
from __future__ import annotations

import datetime as dt
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def gpu_inventory():
    xml = subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True)
    result = []
    for index, gpu in enumerate(ET.fromstring(xml).findall('gpu')):
        result.append(dict(index=index, uuid=gpu.findtext('uuid'),
            memory_used_mib=int(gpu.findtext('fb_memory_usage/used').split()[0]),
            utilization=int(gpu.findtext('utilization/gpu_util').split()[0]),
            processes=[int(p.findtext('pid')) for p in gpu.findall('processes/process_info')]))
    return result


def choose_gpus(inventory, requested=None):
    free = [g for g in inventory if not g['processes'] and g['memory_used_mib'] <= 128
            and g['utilization'] == 0]
    if requested is not None:
        if len(requested) not in (1, 2) or len(set(requested)) != len(requested):
            raise ValueError('Select one or two distinct GPU indices')
        available = {g['index']: g for g in free}
        if any(index not in available for index in requested):
            raise RuntimeError('A requested GPU is unavailable; refusing to substitute devices')
        return [available[index] for index in requested]
    if not free:
        raise RuntimeError('No process-free GPU is available')
    return free[:2]


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def validate_resume_checkpoint(source, storage):
    source = source.resolve()
    if not source.is_relative_to(storage.resolve()) or source.stat().st_uid != os.getuid():
        raise RuntimeError('Resume source must be an owned checkpoint on this node')
    if not source.name.startswith('iter_'):
        raise RuntimeError('Invalid checkpoint directory name')
    int(source.name.split('_')[-1])
    for component in ('model', 'optim', 'scheduler', 'trainer'):
        directory = source / component
        if not (directory / '.metadata').is_file() or not any(directory.glob('*.distcp')):
            raise RuntimeError('Incomplete resume checkpoint component: ' + component)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clips-per-gpu', type=int, default=None,
                        help='Override the default 32 on two GPUs / 64 on one GPU')
    parser.add_argument('--resume-from', type=Path, default=None,
                        help='Continue a complete owned local checkpoint in a new named run')
    parser.add_argument('--gpus', type=int, nargs='+', default=None,
                        help='Use exactly these one or two process-free GPU indices')
    parser.add_argument('--adaptation', choices=('action_only', 'action_video_projections'),
                        default='action_only', help='Select the trainable projection modules')
    args = parser.parse_args()
    if args.clips_per_gpu is not None and args.clips_per_gpu <= 0:
        parser.error('--clips-per-gpu must be positive')
    host = socket.gethostname()
    if host not in ('lg-gwangmo-4xh200-sep-1', 'lg-gwangmo-4xh200-sep-2'):
        raise RuntimeError('This launcher is specific to the Foundary compute nodes')
    if os.environ.get('COSMOS_SITE') != 'custom':
        raise RuntimeError('Source the node-specific .site.local.sh first')
    repo = Path(__file__).resolve().parents[1]
    storage = Path(os.environ['COSMOS_STORAGE_ROOT'])
    weka = Path(os.environ['WEKA_ROOT'])
    runs = Path(os.environ['COSMOS_RUNS_ROOT'])
    for path in (storage, weka, runs, Path(os.environ['COSMOS_ENV_ROOT'])):
        if not path.resolve().is_relative_to(Path('/mnt/local')):
            raise RuntimeError('Training data, weights, environment and outputs must be on local NVMe')
    if shutil.disk_usage(storage).free < 200_000_000_000:
        raise RuntimeError('Need at least 200GB free on local NVMe')
    resume_source = validate_resume_checkpoint(args.resume_from, storage) if args.resume_from else None
    if resume_source is not None:
        source_contract = resume_source.parent.parent / 'native_phase1_contract.json'
        if (source_contract.is_file()
                and json.loads(source_contract.read_text())['adaptation_mode'] != args.adaptation):
            raise RuntimeError('Cannot resume optimizer state across different adaptation modes')
    import wandb
    api = wandb.Api(timeout=30)
    entity = os.environ.get('WANDB_ENTITY') or api.default_entity
    if not entity:
        raise RuntimeError('No authenticated W&B entity is available')
    inventory = gpu_inventory()
    selected = choose_gpus(inventory, args.gpus)
    lock_root = storage / 'operations/gpu_locks'
    lock_root.mkdir(parents=True, exist_ok=True)
    locks = []
    for gpu in selected:
        lock = (lock_root / f"gpu{gpu['index']}.lock").open('a+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    num_gpus = len(selected)
    clips = args.clips_per_gpu if args.clips_per_gpu is not None else 64 // num_gpus
    global_batch = clips * num_gpus
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    execution = f'foundary_phase1_{stamp}'
    ops = storage / 'operations' / execution
    ops.mkdir()
    sources = ops / 'source'
    sources.mkdir()
    hashes = {}
    for filename in ('train_phase1_foundary.py', 'compute_admission.py',
                     'train_phase1_edge_two_tasks_action_only.sh', 'train_phase1_edge_two_tasks_no_text.sh',
                     'train_phase1_edge_two_tasks_action_video_projections.sh', 'run_contract.py',
                     'train_phase1_edge_two_tasks.sh', 'probe_action_memory.py', 'validation.py',
                     'experiment.py', 'latent_omni_model_edge.py', 'run_latent_train.py'):
        source = repo / 'native_phase_training' / filename
        shutil.copy2(source, sources / filename)
        hashes[filename] = hashlib.sha256(source.read_bytes()).hexdigest()
    save(ops / 'source_manifest.json', dict(
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        sha256=hashes))
    node = host.rsplit('sep-', 1)[-1]
    name = f'edge_phase1_T97_20fps_{num_gpus}gpu_bs{clips}_{args.adaptation}_exclude5_fd_invdyn_no_text_p1_200k_sep{node}_{stamp}'
    run_dir = runs / 'cosmos3_camera/camera_world_edge' / name
    admission = dict(host=host, uid=os.getuid(), launcher_pid=os.getpid(),
        cuda_visible_devices=','.join(str(g['index']) for g in selected),
        gpus=selected, initial_inventory=inventory, global_batch=global_batch, clips_per_gpu=clips,
        execution_id=execution, run_dir=str(run_dir), adaptation_mode=args.adaptation, wandb_entity=entity,
        wandb_project='cosmos3_camera', created=stamp)
    save(ops / 'admission.json', admission)
    (ops / 'admission.json').chmod(0o600)
    env = os.environ.copy()
    env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=admission['cuda_visible_devices'],
        NATIVEP1_FOUNDRY_ADMISSION=str(ops / 'admission.json'), NATIVEP1_EXECUTION_ID=execution,
        NATIVEP1_NUM_GPUS=str(num_gpus), NATIVEP1_CLIPS_PER_GPU=str(clips),
        NATIVEP1_RUN_NAME=name, NATIVEP1_MODEL_FAMILY='edge', NATIVEP1_ADAPTATION_MODE=args.adaptation,
        NATIVEP1_PREFIX_LENGTHS='1', NATIVEP1_DISABLE_CAPTIONS='1',
        NATIVEP1_EXPECTED_LATENT_HW='16', NATIVEP1_EXPECTED_IMAGE_HW='256',
        NATIVEP1_REQUIRE_LATENT_CACHE_CONTRACT='1', NYMERIA_NUM_FRAMES='97',
        NYMERIA_RESOLUTION='256', NYMERIA_MODE='mixture', NYMERIA_DROP_MODES='policy,image2video',
        NYMERIA_QUALITY_FILTER='', NYMERIA_REPLACE_STANDALONE_C='1',
        NYMERIA_STANDALONE_C_SUBJECT='camera_wearer',
        NYMERIA_EXCLUDE_ORIGINAL_VIDEO_NAMES='brittney_goodwin,austin_lopez,kim_wilkins,kevin_davis,devon_norris',
        NATIVEP1_LORA_LR='5e-5', NATIVEP1_ACTION_LR_MULT='4', NATIVEP1_ACTION_LOSS_WEIGHT='10',
        NATIVEP1_NORMALIZE_LOSS_BY_ACTIVE='0', NATIVEP1_AUTO_EVAL='0', NATIVEP1_AUTO_EVAL_FULL71='0',
        NATIVEP1_LD_LIBRARY_PATH=env.get('LD_LIBRARY_PATH', ''),
        EDGE_MODEL_ROOT=str(weka / 'Cosmos3-Edge'), BASE_CHECKPOINT_PATH=str(weka / 'cosmos3_edge_dcp'),
        WAN_VAE_PATH=str(weka / 'wan22_vae/Wan2.2_VAE.pth'),
        NYMERIA_LATENT_ROOT=str(weka / 'nymeriaplus_kimodo_proportional/joint_latents_T97_256tier_256'),
        WANDB_ENTITY=entity, WANDB_PROJECT='cosmos3_camera', WANDB_MODE='online',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', TORCH_NCCL_ASYNC_ERROR_HANDLING='1',
        PYTHONPATH=f"{repo}:{repo / 'nymeria_world'}:{env['COSMOS_FRAMEWORK_EDGE_ROOT']}",
        TRITON_CACHE_DIR=str(storage / '.cache/triton' / execution))
    for key in ('SLURM_JOB_ID', 'SLURM_JOB_USER', 'NYMERIA_FULL_FT', 'NYMERIA_MAX_SAMPLES',
                'NATIVEP1_SHIFT_OVERRIDE', 'NATIVEP1_PREFIX_SAMPLING_WEIGHTS', 'RANK', 'LOCAL_RANK',
                'WORLD_SIZE', 'GROUP_RANK', 'ROLE_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
                'WANDB_RUN_ID', 'WANDB_RESUME', 'WANDB_NAME', 'WANDB_DISABLED', 'RESUME_CHECKPOINT'):
        env.pop(key, None)
    # Record only the reproducible non-secret recipe, not inherited credentials.
    save(ops / 'recipe.json', {k: v for k, v in env.items() if k.startswith(('NATIVEP1_', 'NYMERIA_'))})
    def status(phase, **extra):
        record = dict(time=dt.datetime.now(dt.timezone.utc).isoformat(), phase=phase,
                      operations=str(ops), **extra)
        save(ops / 'status.json', record)
        print(json.dumps(record), flush=True)
    def require_free_selected():
        current = {g['index']: g for g in gpu_inventory()}
        for gpu in selected:
            value = current[gpu['index']]
            if value['uuid'] != gpu['uuid'] or value['processes'] or value['memory_used_mib'] > 128:
                raise RuntimeError('An admitted GPU is now occupied: ' + str(value))
    require_free_selected()
    status('starting_memory_gate', admission=admission)
    probe = ops / 'memory_probe'
    probe.mkdir()
    probe_env = dict(env, PHASE1_MEMORY_PROBE_DIR=str(probe), PHASE1_MEMORY_PROBE_STEPS='4',
        PHASE1_MEMORY_PROBE_GLOBAL_BATCH=str(global_batch), NATIVEP1_WANDB_PREVIEWS='0', WANDB_MODE='disabled',
        COSMOS3_CHECKPOINT_WALL_CLOCK_MINUTES='0', IMAGINAIRE_OUTPUT_ROOT=str(probe / 'native_output'))
    cmd = [str(Path(env['COSMOS_ENV_ROOT']) / 'bin/torchrun'), '--standalone',
        f'--nproc_per_node={num_gpus}', '-m', 'native_phase_training.probe_action_memory',
        '--sft-toml', str(repo / 'native_phase_training/world_camera_nymeria_latent_edge.toml'),
        f'job.name=foundary_{args.adaptation}_memory', 'job.wandb_mode=disabled', 'trainer.grad_accum_iter=1',
        'trainer.logging_iter=1', 'trainer.callbacks.iter_speed.hit_thres=2',
        'scheduler.cycle_lengths=[200000]', 'scheduler.warm_up_steps=[500]',
        'scheduler.f_start=[0.4]', 'scheduler.f_max=[0.4]', 'scheduler.f_min=[0.0]',
        f"checkpoint.load_path={env['BASE_CHECKPOINT_PATH']}", 'checkpoint.load_training_state=false',
        'checkpoint.keys_to_skip_loading=[net_ema.]', 'checkpoint.keys_not_to_resume=[]',
        'model.config.compile.enabled=false', 'model.config.ema.enabled=true',
        'model.config.parallelism.data_parallel_shard_degree=1',
        f'model.config.parallelism.data_parallel_replicate_degree={num_gpus}',
        'model.config.parallelism.context_parallel_shard_degree=1',
        'model.config.parallelism.cfg_parallel_shard_degree=1']
    save(ops / 'probe_command.json', cmd)
    with (ops / 'probe.log').open('w') as log:
        result = subprocess.run(cmd, env=probe_env, cwd=env['COSMOS_FRAMEWORK_EDGE_ROOT'], stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        status('memory_gate_failed', returncode=result.returncode)
        raise SystemExit(result.returncode)
    for rank in range(num_gpus):
        report = json.loads((probe / f'rank{rank}.json').read_text())
        if report['stage'] != 'complete' or report['completed_steps'] != 4 or report['microbatch'] != clips:
            raise RuntimeError('Incomplete memory/gradient gate')
    status('memory_gate_passed')
    time.sleep(3)
    require_free_selected()
    env.update(NATIVEP1_WANDB_PREVIEWS='1', COSMOS3_CHECKPOINT_WALL_CLOCK_MINUTES='15',
               IMAGINAIRE_OUTPUT_ROOT=str(runs))
    if resume_source is not None:
        source = validate_resume_checkpoint(resume_source, storage)
        destination = run_dir / 'checkpoints' / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Published checkpoints are immutable. Pin them without duplicating tensors.
        shutil.copytree(source, destination, copy_function=os.link)
        env['RESUME_CHECKPOINT'] = str(destination)
        save(ops / 'continuation.json', dict(source_checkpoint=str(source),
            pinned_checkpoint=str(destination), source_run=str(source.parent.parent),
            step=int(source.name.split('_')[-1]), global_batch=global_batch,
            clips_per_gpu=clips, optimizer_and_scheduler_resumed=True))
    status('starting_production', run_dir=str(run_dir), gpus=selected,
           global_batch=global_batch, clips_per_gpu=clips,
           resume_checkpoint=env.get('RESUME_CHECKPOINT'))
    with (ops / 'production.log').open('w') as log:
        result = subprocess.run(['bash', str(repo / f'native_phase_training/train_phase1_edge_two_tasks_{args.adaptation}.sh')],
            env=env, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
    status('production_finished' if result.returncode == 0 else 'production_failed', returncode=result.returncode)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
