"""Compute-only, process-aware queue for explicitly authorized idle GPU work.

One supervisor is launched per existing owned allocation. It never cancels a job,
stops another process, or creates a CUDA context while waiting. Each task also
holds a cross-allocation file lease. Existing full-test pool claims and training
loaders take precedence over this follow-up queue.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/scratch2/whwjdqls99/cosmos/operations/idle_gpu_followups_20261104')
OLD=ROOT.parent/'fulltest_v2m_ablations_20261104'
TRAIN=ROOT.parent/'uem_shape_threeway_ddp_20261104/training'
MODELS=('uem_final_trajectory','uem_final_video','uem_final_both')


def read(path):
    return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    stream=path.open('a')
    try:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close();return None
    return stream


def inventory():
    result=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used',
                                    '--format=csv,noheader,nounits'],text=True)
    rows=[r for r in csv.reader(result.splitlines()) if r]
    # NVML sees only the step's cgroup devices. UUIDs remain unambiguous even
    # when Slurm remaps CUDA index0 to a different physical index.
    return [dict(index=int(r[0]),uuid=r[1].strip(),memory=int(r[2])) for r in rows]


def busy_reason(gpu):
    own_job=os.environ['SLURM_JOB_ID']
    result=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'],text=True)
    for row in csv.reader(result.splitlines()):
        if row and row[0].strip()==gpu:
            return 'CUDA process '+row[1].strip()
    for item in inventory():
        if item['uuid']==gpu and item['memory']>512:
            return 'GPU memory still occupied'
    for owner in (OLD/'claims').glob('*.owner'):
        if owner.read_text().split()[:1]!=[own_job]:
            continue
        probe=lock(owner.with_suffix('.lock'))
        if probe is None:
            return 'Existing pool claim '+owner.stem
        probe.close()
    # Covers checkpoint/model loading before a CUDA process becomes visible.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():
            continue
        try:
            if proc.stat().st_uid!=os.getuid():
                continue
            command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if not any(x in command for x in ('-m nymeria_baselines.fulltest_v2m ',
                    '-m nymeria_baselines.fulltest_v2m_batched ',
                    '-m nymeria_baselines.train_uem_shape_modalities ')):
                continue
            environment=(proc/'environ').read_bytes().split(b'\0')
            if ('SLURM_JOB_ID='+own_job).encode() in environment:
                return 'Existing training/inference process '+proc.name
        except (FileNotFoundError,PermissionError,ProcessLookupError):
            continue
    return None


def tasks():
    p1=ROOT/'phase1_videoio';uem=ROOT/'uem_final350'
    if (p1/'configuration.json').exists() and not (p1/'metric_acceptance.json').exists():
        yield 'phase1_metric',['phase1_metric'],p1/'metric_acceptance.json'
    if (TRAIN/'COMPLETE.json').exists() and not (uem/'configuration.json').exists():
        yield 'uem_prepare',['uem_prepare'],uem/'configuration.json'
    if (uem/'configuration.json').exists():
        gt=uem/'scores/GT/COMPLETE.json'
        if not gt.exists():
            yield 'uem_GT',['uem_eval','GT'],gt
        # Generation is independent of GT scoring. Keep the same global model
        # lease for both stages, including older already-running dispatchers.
        for model in MODELS:
            inference=uem/'models'/model/'INFERENCE_COMPLETE.json'
            scoring=uem/'scores'/model/'COMPLETE.json'
            if not inference.exists():
                yield model,['uem_generate',model],inference
            elif gt.exists() and not scoring.exists():
                yield model,['uem_score',model],scoring
        if all((uem/'scores'/key/'COMPLETE.json').exists() for key in ('GT',*MODELS)) and not (uem/'COMPLETE.json').exists():
            yield 'uem_publish',['uem_publish'],uem/'COMPLETE.json'
    location=ROOT/'phase1_checkpoint.json'
    if not location.exists():
        return
    path=Path(read(location)['path'])
    if not (path/'model/.metadata').is_file():
        return
    preflight=p1/'generation_preflight.json'
    scored=p1/'metrics'/ (read(p1/'cases.json')[0]['id']+'.json')
    if not preflight.exists() or not scored.exists():
        yield 'phase1_preflight',['phase1_preflight','0'],scored
    elif (p1/'metric_acceptance.json').exists():
        count=read(p1/'configuration.json')['shards']
        for shard in range(count):
            marker=p1/'shards'/f'{shard:03d}_COMPLETE.json'
            if not marker.exists():
                yield f'phase1_{shard:03d}',['phase1_shard',str(shard)],marker
        if all((p1/'shards'/f'{shard:03d}_COMPLETE.json').exists() for shard in range(count)) and not (p1/'COMPLETE.json').exists():
            yield 'phase1_publish',['phase1_publish'],p1/'COMPLETE.json'


def worker(gpu,once=False):
    job=os.environ['SLURM_JOB_ID']
    identity=f'{job}_{gpu}'
    status=ROOT/'workers'/f'{identity}.json'
    lease=lock(ROOT/'leases'/f'{gpu}.lock')
    if lease is None:
        return
    stable=0
    while True:
        reason=busy_reason(gpu)
        stable=stable+1 if reason is None else 0
        save(status,dict(job=job,gpu=gpu,node=os.uname().nodename,time=time.time(),
                         state='waiting',reason=reason or 'Confirming idle twice',idle_checks=stable))
        if stable>=2 or (once and reason is None):
            for key,command,marker in tasks():
                failure=ROOT/'tasks'/f'{key}.failed.json'
                # A failed task remains reviewable; no repeated GPU retry storm.
                if failure.exists() or marker.exists():
                    continue
                task_lease=lock(ROOT/'tasks'/f'{key}.lock')
                if task_lease is None:
                    continue
                try:
                    if marker.exists() or busy_reason(gpu):
                        continue
                    save(status,dict(job=job,gpu=gpu,node=os.uname().nodename,time=time.time(),state='running',task=key))
                    environment=os.environ.copy();environment['CUDA_VISIBLE_DEVICES']=gpu
                    log=ROOT/'tasks'/f'{key}.log'
                    with log.open('a') as stream:
                        result=subprocess.run(['bash',str(ROOT/'task.sh'),*command],env=environment,
                                              stdout=stream,stderr=subprocess.STDOUT)
                    evidence=dict(job=job,gpu=gpu,task=key,exit_code=result.returncode,
                                  marker=str(marker),marker_exists=marker.exists(),time=time.time(),log=str(log))
                    save(ROOT/'tasks'/f'{key}.receipt.json',evidence)
                    if result.returncode or not marker.exists():
                        save(failure,evidence)
                    elif key=='uem_prepare' or key=='uem_GT' or key in MODELS:
                        with log.open('a') as stream:
                            subprocess.run(['bash',str(ROOT/'summarize.sh'),'uem'],env=environment,
                                           stdout=stream,stderr=subprocess.STDOUT)
                    elif key.startswith('phase1_') and key not in ('phase1_metric','phase1_preflight','phase1_publish'):
                        with log.open('a') as stream:
                            subprocess.run(['bash',str(ROOT/'summarize.sh'),'phase1'],env=environment,
                                           stdout=stream,stderr=subprocess.STDOUT)
                finally:
                    task_lease.close()
                stable=0
                break
        if (ROOT/'phase1_videoio/COMPLETE.json').exists() and (ROOT/'uem_final350/COMPLETE.json').exists():
            save(status,dict(job=job,gpu=gpu,node=os.uname().nodename,time=time.time(),state='complete'))
            return
        if once:
            return
        time.sleep(30)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu');p.add_argument('--once',action='store_true');p.add_argument('--inspect',action='store_true')
    a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and os.uname().nodename not in ('cs-gpu-01','node37')
    assert 'login' not in os.uname().nodename
    if a.inspect:
        print(json.dumps([dict(**g,busy=busy_reason(g['uuid'])) for g in inventory()]),flush=True)
    elif a.gpu:
        assert a.gpu in {g['uuid'] for g in inventory()}
        worker(a.gpu,a.once)
    else:
        processes=[subprocess.Popen([sys.executable,__file__,'--gpu',g['uuid'],*(['--once'] if a.once else [])]) for g in inventory()]
        codes=[proc.wait() for proc in processes]
        if any(codes):
            raise SystemExit(1)


if __name__=='__main__':
    main()
