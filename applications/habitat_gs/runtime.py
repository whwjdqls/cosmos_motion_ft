"""Isolate official inference subprocesses and preserve evaluation launch settings."""
import os
from pathlib import Path
import socket
import subprocess


def require_compute():
    host = socket.gethostname().split('.')[0]
    if 'login' in host or host in {'cs-gpu-01','node37'}:
        raise RuntimeError('Applications must run on a permitted owned compute node')
    site = os.environ.get('COSMOS_SITE', 'yonsei')
    if site == 'yonsei' and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Yonsei requires an owned Slurm allocation')


def checked_command(command, log, cwd=None, env=None):
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open('a') as stream:
        stream.write('\nCOMMAND ' + repr(command) + '\n')
        stream.flush()
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, cwd=cwd, env=env)


def official_command(profile, inputs, out):
    command = [os.environ.get('COSMOS_PYTHON', os.sys.executable), '-m', 'torch.distributed.run',
        '--standalone', '--nproc_per_node=1', '-m', 'cosmos_framework.scripts.inference',
        '--checkpoint-path', profile['checkpoint'], '--config-file', profile['config_file'],
        '--experiment', profile['experiment'], '--sampler', profile['sampler'], '--use-ema-weights',
        '--parallelism-preset','latency','--dp-shard-size','1','--dp-replicate-size','1',
        '--cp-size','1','--cfgp-size','1','--no-use-torch-compile','--no-use-cuda-graphs',
        '--no-guardrails', '-o', str(out), '-i', str(inputs)]
    if not profile['diffusion_cache']:
        command.append('--no-diffusion-cache')
    return command


def evaluation_environment(profile):
    """Same resolved native exports as run_contract.write_eval_resolution."""
    contract = profile['contract']
    env = os.environ.copy()
    env.update(NATIVEP1_MODEL_FAMILY=contract['model_family'],
        NATIVEP1_ADAPTATION_MODE=contract['adaptation_mode'],
        NYMERIA_DROP_MODES=','.join(contract['dropped_modes']),
        NYMERIA_RESOLUTION=str(contract['model_resolution']),
        NYMERIA_NUM_FRAMES=str(contract['num_frames']),
        NATIVEP1_SHIFT_OVERRIDE=str(contract['training_shift']),
        NATIVEP1_TRAINING_SHIFT=str(contract['training_shift']),
        NATIVEP1_INFERENCE_SHIFT=str(contract['inference_shift']),
        NATIVEP1_EFFECTIVE_SHIFT=str(contract['inference_shift']),
        NATIVEP1_CONDITIONING_FPS=str(contract['conditioning_fps']),
        NATIVEP1_BASE_FPS=str(contract['base_fps']))
    return env
