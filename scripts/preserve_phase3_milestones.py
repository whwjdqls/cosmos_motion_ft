"""Preserve exact Phase3 100k states as independent, verified node-local copies."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import time
import torch

def read(path):return json.loads(Path(path).read_text())
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def fingerprint(contract):return hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def validate(path,contract,step=100000):
    state=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    assert state['step']==step and state['fingerprint']==fingerprint(contract),'Wrong step or model identity'
    assert state['scheduler']['step']==step and state['scheduler']['recipe']==contract['recipe']
    assert state['next_global_draw']==step*32 and len(state['rng_by_rank'])==2
    assert state['bridges'] and state['optimizer']['state'] and state['optimizer']['param_groups']
    assert state['ema']['num_updates']==step and state['ema']['decay']==.999
    assert state['ema']['state']['bridges'].keys()==state['bridges'].keys()
    for group in [state['bridges'],state['ema']['state']['bridges']]:
        assert all(bool(torch.isfinite(value).all()) for value in group.values())
    result=dict(step=step,raw_tensors=len(state['bridges']),ema_tensors=len(state['ema']['state']['bridges']),
        ema_updates=state['ema']['num_updates'],optimizer_states=len(state['optimizer']['state']),
        rng_ranks=2,next_global_draw=state['next_global_draw'],training_fingerprint=state['fingerprint'])
    del state
    return result

def preserve_one(model,folder):
    """Return None until an atomically published exact100k checkpoint exists."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    run=Path(model['out']);contract_path=run/'contract.json';source=run/'checkpoints/step_000100000.pt'
    if not contract_path.exists():return None
    contract=read(contract_path)
    for key,value in model['expected_recipe'].items():
        assert contract['recipe'].get(key)==value,'Run recipe differs: '+key
    if not source.is_file():return None
    free=shutil.disk_usage(folder).free
    if free<source.stat().st_size+2**30:raise RuntimeError('Insufficient local space for independent milestone copy')
    checks=validate(source,contract)
    target=folder/source.name;temporary=target.with_name(target.name+'.partial.'+str(os.getpid()))
    original_stat=source.stat();source_identity=(original_stat.st_dev,original_stat.st_ino,original_stat.st_size,original_stat.st_mtime_ns)
    try:
        if not target.exists():
            with source.open('rb') as src,temporary.open('xb') as dst:
                shutil.copyfileobj(src,dst,16*1024**2);dst.flush();os.fsync(dst.fileno())
            source_hash=sha(source);target_hash=sha(temporary)
            assert source_hash==target_hash,'Copy checksum mismatch'
            current=source.stat()
            assert source_identity==(current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns),'Source changed while copying'
            validate(temporary,contract)
            temporary.replace(target)
        else:
            source_hash=sha(source);target_hash=sha(target)
            assert source_hash==target_hash,'Refusing to replace a different preserved checkpoint'
            validate(target,contract)
    finally:
        if temporary.exists():temporary.unlink()
    assert target.stat().st_ino!=source.stat().st_ino and target.stat().st_nlink==1,'Need an independent file, not a hardlink'
    manifest_path=run/'launch/source_manifest.json';manifest=read(manifest_path);snapshot=run/'launch/source_snapshot'
    for relative,expected in manifest.items():
        rel=Path(relative)
        assert not rel.is_absolute() and '..' not in rel.parts
        src=snapshot/rel;assert sha(src)==expected,'Frozen source changed: '+relative
        dst=folder/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():shutil.copyfile(src,dst)
        assert sha(dst)==expected
    atomic_json(folder/'contract.json',contract);atomic_json(folder/'source_manifest.json',manifest)
    record=dict(status='preserved_verified',run=str(run),path=str(target),bytes=target.stat().st_size,
        sha256=target_hash,independent_copy=True,source_inode=source.stat().st_ino,destination_inode=target.stat().st_ino,
        source_files_verified=len(manifest),full_resume_state_verified=True,**checks,time=time.time())
    atomic_json(folder/'VERIFIED.json',record)
    return record

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--registry',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--once',action='store_true')
    args=parser.parse_args();registry=read(args.registry)
    if os.uname().nodename!=registry['node']:raise RuntimeError('Wrong node')
    for path in [args.registry,args.root]:
        if not path.resolve().is_relative_to('/mnt/local'):raise ValueError('Only node-local paths are allowed')
    args.root.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    os.environ['CUDA_VISIBLE_DEVICES']='';torch.set_num_threads(1)
    lock=(args.root/'watcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19]
    atomic_json(args.root/'WATCHER.json',dict(pid=os.getpid(),start=start,node=os.uname().nodename,registry=str(args.registry),time=time.time()))
    verified={}
    try:
        while True:
            states={}
            for key,model in registry['models'].items():
                folder=args.root/key;receipt=folder/'VERIFIED.json'
                if key not in verified and receipt.exists():
                    old=read(receipt);assert Path(old['path']).stat().st_size==old['bytes'] and sha(old['path'])==old['sha256']
                    verified[key]=old
                if key not in verified:
                    record=preserve_one(model,folder)
                    if record:
                        verified[key]=record;print('PRESERVED_EXACT100K',key,record['sha256'],flush=True)
                if key in verified:states[key]=verified[key]
                else:
                    progress=Path(model['out'])/'status.json'
                    states[key]=dict(status='waiting_for_exact100k',current=read(progress) if progress.exists() else None)
            complete=len(verified)==len(registry['models'])
            atomic_json(args.root/'status.json',dict(pid=os.getpid(),node=os.uname().nodename,complete=complete,models=states,time=time.time()))
            if complete:
                atomic_json(args.root/'COMPLETE.json',dict(passed=True,all_exact100k_independently_preserved=True,models=states,time=time.time()));break
            if args.once:break
            time.sleep(30)
    except Exception as error:
        atomic_json(args.root/'FAILED.json',dict(error=type(error).__name__,message=str(error),time=time.time()));raise

if __name__=='__main__':main()
