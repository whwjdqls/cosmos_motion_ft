"""Fixed71 EMA camera-to-motion monitoring and qualitative W&B previews."""
from pathlib import Path
import json
import numpy as np
import torch
import torch.distributed as dist
from .runtime import atomic_json,film_state,sample

def validate(model,ema,plan,step,rank,world,device,mean,std,wandb_run):
    from checkpoint import capture_rng_state,restore_rng_state
    from motion_expert_phase3_edge_joint.validation_metrics import physical
    folder=Path(plan['out'])/'validation'/f'step_{step:09d}'
    if (folder/'COMPLETE.json').exists():return
    rng=capture_rng_state();raw=film_state(model);was_training=model.training
    model.film.load_state_dict(ema,strict=True)
    rows=json.loads(Path(plan['validation_cases']).read_text());assert len(rows)==71
    mine=rows[rank::world];predictions=[]
    try:
        for start in range(0,len(mine),16):
            batch=mine[start:start+16]
            arrays=[]
            for row in batch:
                with np.load(row['bundle']) as z:arrays.append((z['neutral'].copy(),z['camera_features'].copy()))
            neutral=torch.from_numpy(np.stack([a[0] for a in arrays])).to(device).float();camera=torch.from_numpy(np.stack([a[1] for a in arrays])).to(device).float()
            predicted=sample(model,neutral,camera,[r['seed'] for r in batch]).cpu().numpy()
            predictions.extend((r['id'],p) for r,p in zip(batch,predicted,strict=True))
        if world>1:
            received=[None]*world;dist.all_gather_object(received,predictions);predictions=sum(received,[])
        if rank==0:
            from decode_uniego_torch import decode_joints
            from render_motion import render_sidebyside,write_mp4
            folder.mkdir(parents=True,exist_ok=True);pred=dict(predictions);scores=[];logs={'validation/step':step}
            for i,row in enumerate(rows):
                with np.load(row['bundle']) as z:target=z['motion'].copy()
                value=pred[row['id']];assert value.shape==(97,283) and np.isfinite(value).all()
                np.savez(folder/(row['id']+'.npz'),generated=value,film_step=step,base_step=200000,weights='ema',seed=row['seed'])
                a,b=[x*std.cpu().numpy()+mean.cpu().numpy() for x in (value,target)]
                scores.append(dict(id=row['id'],normalized_mse=float(np.square(value-target).mean()),**physical(a,b)))
                if i<2:
                    with torch.no_grad():g,p=[decode_joints(torch.from_numpy(x)[None])[0].numpy() for x in (b,a)]
                    frames=render_sidebyside(g,p,caption=f'Camera to motion | FiLM step {step}',width=768,height=384)
                    path=folder/(row['id']+'_comparison.mp4');write_mp4(frames,str(path),fps=20)
                    if wandb_run is not None:
                        import wandb
                        logs[f'validation_preview/camera2motion_{i}']=wandb.Video(str(path),fps=20,format='mp4')
            summary={k:float(np.mean([s[k] for s in scores])) for k in scores[0] if k!='id'}
            atomic_json(folder/'per_case.json',scores);atomic_json(folder/'summary.json',summary)
            logs.update({'validation/camera2motion/'+k:v for k,v in summary.items()})
            if wandb_run is not None:wandb_run.log(logs)
            atomic_json(folder/'COMPLETE.json',dict(passed=True,step=step,cases=71,weights='ema',task='camera2motion',sampler='UniPC35_shift3_CFG1',no_video_or_caption=True))
    finally:
        model.film.load_state_dict(raw,strict=True);model.train(was_training);restore_rng_state(rng)
    if world>1:dist.barrier()
