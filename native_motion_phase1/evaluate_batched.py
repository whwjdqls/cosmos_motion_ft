"""Batched execution of the frozen full-test native motion evaluation protocol."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from native_motion_phase1.evaluate_fulltest import TASKS, sha, save_json, load_model
from native_motion_phase1.data import inference_batch


def inputs(args,row,model):
    assert sha(row['bundle'])==row['bundle_sha256']
    with np.load(row['bundle'],allow_pickle=False) as z:
        motion,neutral=z['motion'].copy(),z['neutral'].copy()
    p=Path(row['latent'])
    if not p.exists():
        from nymeria_camera_dataset import decode_window_pyav
        from native_phase_training.evaluate_inverse_forward import _resize_gt_like_native
        rgb=_resize_gt_like_native(decode_window_pyav(row['rgb'],row['start'],97,20.),256,256)
        assert rgb.shape==(97,256,256,3)
        latent=model.encode(torch.from_numpy(rgb).permute(3,0,1,2)[None].to('cuda',torch.float32)/127.5-1)[0].float().cpu().numpy()
        p.parent.mkdir(exist_ok=True)
        temp=p.with_suffix('.tmp')
        with temp.open('wb') as f:np.savez(f,latents=latent)
        temp.replace(p)
    with np.load(p,allow_pickle=False) as z:latent=z['latents'].copy()
    assert latent.shape==(48,25,16,16) and np.isfinite(latent).all()
    return motion,neutral,latent


def merge(batches):
    out={}
    for key in batches[0]:
        values=[b[key] for b in batches]
        if isinstance(values[0],list):out[key]=sum(values,[])
        elif isinstance(values[0],torch.Tensor):out[key]=torch.cat(values,0)
        else:raise TypeError((key,type(values[0])))
    return out


@torch.no_grad()
def main(args):
    cfg=json.loads((args.out/'configuration.json').read_text())
    rows=json.loads((args.out/'cases.json').read_text())
    assert sha(args.out/'cases.json')==cfg['cases_sha256']
    assert sha(args.checkpoint/'model/.metadata')==cfg['checkpoint_model_metadata_sha256']
    lock=(args.out/'sampling.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    runtime=dict(source_sha256=sha(__file__),configuration_sha256=sha(args.out/'configuration.json'),
                 batch_size=args.batch_size,weights='EMA200k',sampling=cfg['sampling'])
    runtime_path=args.out/'batch_execution.json'
    if runtime_path.exists():assert json.loads(runtime_path.read_text())==runtime
    else:save_json(runtime_path,runtime)
    model=load_model(args)
    order=cfg['qualitative_indices']+[i for i in range(len(rows)) if i not in cfg['qualitative_indices']]
    if args.limit:order=order[:args.limit]
    begin=time.monotonic();newly=0
    for start in range(0,len(order),args.batch_size):
        indices=order[start:start+args.batch_size]
        active=[i for i in indices if any(not (args.out/'predictions'/t/(rows[i]['id']+'.npz')).exists() for t in TASKS)]
        cached={i:inputs(args,rows[i],model) for i in active}
        for task in TASKS:
            selected=[i for i in active if not (args.out/'predictions'/task/(rows[i]['id']+'.npz')).exists()]
            if not selected:continue
            samples=[]
            for index in selected:
                motion,neutral,latent=cached[index]
                motion=motion if task=='motion2video' else np.zeros_like(motion)
                visual=latent.copy()
                if task=='motion2video':visual[:,1:]=0
                samples.append(inference_batch(motion,neutral,visual,task,rows[index]['caption']))
            batch=merge(samples);before=time.monotonic()
            result=model.generate_samples_from_batch(batch,net=model.net,
                seed=[rows[i]['seed'] for i in selected],num_steps=30,shift=10.,guidance=1.)
            for pos,index in enumerate(selected):
                row=rows[index];motion,neutral,latent=cached[index]
                if task=='motion2video':
                    predicted=result['vision'][pos]
                    observed=float((predicted[0,:,:1].float()-torch.from_numpy(latent[:,:1]).to('cuda').float()).abs().max())
                    assert observed<1e-4
                    decoded=model.decode(predicted.float())
                    assert decoded.shape==(1,3,97,256,256) and torch.isfinite(decoded).all()
                    generated=((decoded[0].float().clamp(-1,1)+1)*127.5).round().byte().permute(1,2,3,0).cpu().numpy()
                    del decoded
                else:
                    action=result['action'][pos].reshape(98,-1)[:,:283].float().cpu().numpy()
                    observed=float(np.max(np.abs(action[0]-samples[pos]['action'][0][0,:283].numpy())))
                    assert observed<1e-4
                    generated=action[1:]
                    assert generated.shape==(97,283) and np.isfinite(generated).all()
                dest=args.out/'predictions'/task/(row['id']+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
                temp=dest.with_suffix('.tmp')
                with temp.open('wb') as f:
                    np.savez(f,generated=generated,configuration_sha256=runtime['configuration_sha256'],
                             execution_sha256=sha(__file__),observed_error=observed,seed=row['seed'])
                temp.replace(dest);newly+=1
            record=dict(stage='sampling',task=task,completed_batch_cases=start+len(indices),total_cases=len(rows),
                        newly_generated=newly,batch_cases=len(selected),batch_seconds=time.monotonic()-before,
                        elapsed=time.monotonic()-begin,peak_gib=torch.cuda.max_memory_allocated()/2**30,time=time.time())
            save_json(args.out/'status.json',record);print(json.dumps(record),flush=True)
            del result,batch,samples
        del cached
    complete=dict(stage='smoke_complete' if args.limit else 'inference_complete',cases=len(order),time=time.time())
    save_json(args.out/('BATCH_SMOKE_COMPLETE.json' if args.limit else 'INFERENCE_COMPLETE.json'),complete)
    print(json.dumps(complete),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    assert a.out.resolve().is_relative_to('/mnt/local') and a.checkpoint.resolve().is_relative_to('/mnt/local')
    assert 1<=a.batch_size<=32
    main(a)
