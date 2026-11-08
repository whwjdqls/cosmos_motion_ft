"""Two previously coordinated GPU slots; leave every trainer untouched."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
PLAN=json.loads((ROOT/'plan.json').read_text())


def atomic(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def devices():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    return {int(i.strip()):(u.strip(),int(m.strip())) for i,u,m in (line.split(',') for line in rows.splitlines())}


def check_resident(gpu,expected):
    uuid,free=devices()[gpu]
    if free<65536:raise RuntimeError('Need64GiB free on approved co-evaluation slot')
    rows=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    residents={int(p.strip()) for p in rows.splitlines() if p.strip().isdigit()}
    if residents!={expected['pid']}:raise RuntimeError(f'Unexpected GPU{gpu} residents: {residents}')
    p=Path('/proc',str(expected['pid']))
    assert p.stat().st_uid==os.getuid()
    assert p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]==expected['start']
    return uuid


def main():
    assert not (ROOT/'RUNNING.json').exists()
    initial={7:361032,3:354544}
    training={gpu:dict(pid=pid,start=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19]) for gpu,pid in initial.items()}
    for gpu,identity in training.items():check_resident(gpu,identity)
    queue=[dict(run=PLAN['runs'][0],partition='all'),dict(run=PLAN['runs'][1],partition='all'),
           dict(run=PLAN['runs'][2],partition='m2v'),dict(run=PLAN['runs'][2],partition='v2m')]
    active={};finished=[];launches=[]
    while queue or active:
        for gpu in list(active):
            item=active[gpu];code=item['process'].poll()
            if code is None:continue
            entry={k:v for k,v in item.items() if k!='process'};entry['exit_code']=code
            finished.append(entry);del active[gpu]
            if code!=0:
                atomic(ROOT/'ERROR.json',dict(failed=entry,remaining=[(r['run']['id'],r['partition']) for r in queue],other_workers_left_running=True));return
            assert (Path(item['out'])/'COMPLETE.json').exists()
        for gpu in PLAN['eligible_gpus']:
            if gpu in active or not queue:continue
            uuid=check_resident(gpu,training[gpu]);task=queue.pop(0);run=task['run'];part=task['partition']
            out=Path(run['out'])/('part_'+part);out.mkdir(exist_ok=False)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=uuid,FOUNDRY_GPU_UUIDS=uuid,EVAL_PHYSICAL_GPU=str(gpu),
                WANDB_MODE='disabled',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TMPDIR=str(ROOT/'tmp'))
            env['PYTHONPATH']=':'.join([str(ROOT),env['COSMOS_FRAMEWORK_EDGE_ROOT'],'/home/jinhyung.kim/cosmos_motion_ft/.deps/kimodo_open',env.get('PYTHONPATH','')])
            for name in ['RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT']:env.pop(name,None)
            with (out/'worker.log').open('x') as log:
                process=subprocess.Popen(['nice','-n','10',sys.executable,str(ROOT/'run_eval.py'),run['id'],part],stdout=log,stderr=subprocess.STDOUT,env=env,cwd=env['COSMOS_FRAMEWORK_EDGE_ROOT'],start_new_session=True)
            entry=dict(id=run['id'],partition=part,pid=process.pid,gpu=gpu,out=str(out),training_resident=training[gpu]['pid']);launches.append(entry);active[gpu]={**entry,'process':process};print(json.dumps(dict(event='launched',**entry)),flush=True)
        atomic(ROOT/'RUNNING.json',dict(launches=launches,active=[{k:v for k,v in a.items() if k!='process'} for a in active.values()],queued=[(r['run']['id'],r['partition']) for r in queue],finished=finished))
        if queue or active:time.sleep(3)
    atomic(ROOT/'ALL_METRICS_COMPLETE.json',dict(models=3,predictions=852,variants_per_model=4,workers=4,runs=finished))
if __name__=='__main__':main()
