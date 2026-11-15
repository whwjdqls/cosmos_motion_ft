"""Lossless qualitative pixels plus requested decoded-joint metrics."""
from pathlib import Path
import json
import numpy as np
import torch
from motion_error_metrics import extra_metrics,protocol

class CaptureVAE:
    def __init__(self,vae,records,folder,indices):
        self.vae,self.records,self.folder=vae,records,Path(folder)
        self.selected={records[i]['id'] for i in indices};self.offset=0
        (self.folder/'qualitative_rgb').mkdir(exist_ok=True)
    def get_output_range(self):return self.vae.get_output_range()
    def decode(self,latents):
        output=self.vae.decode(latents)
        assert torch.isfinite(output).all()
        rows=self.records[self.offset:self.offset+len(latents)]
        assert len(rows)==len(latents)
        for i,row in enumerate(rows):
            if row['id'] in self.selected:
                pixels=((output[i].float().clamp(-1,1)+1)*127.5).round().byte().permute(1,2,3,0).cpu().numpy()
                assert pixels.shape==(97,256,256,3)
                np.save(self.folder/'qualitative_rgb'/(row['id']+'.npy'),pixels)
        self.offset+=len(latents)
        return output

def extra_motion_metrics(records,folder,contract,indices):
    from decode_uniego_torch import decode_transforms
    from nymeria_baselines.body_shape import BODY_JOINTS
    from motion_expert_phase3_edge_joint.validation import load_case,local_path
    from runtime_paths import resolve_legacy_path
    from motion_expert_phase3_edge_joint.settings import atomic_json
    mean,std=[np.load(local_path(resolve_legacy_path(contract[k]))).astype(np.float32) for k in ('motion_stats_mean','motion_stats_std')]
    folder=Path(folder);original=json.loads((folder/'v2m_per_case.json').read_text())
    measured=[];maximum=0.;selected={records[i]['id'] for i in indices}
    (folder/'qualitative_joints').mkdir(exist_ok=True)
    for index,row in enumerate(records):
        target,_,_=load_case(row,contract)
        with np.load(folder/'video2motion'/(row['id']+'.npz')) as z:
            assert int(z['seed'])==row['seed'];prediction=z['generated'].copy()
        with torch.inference_mode():
            full=[decode_transforms(torch.from_numpy(x*std+mean)[None])[0].numpy()[...,:3,3] for x in (prediction,target)]
        p,g=[np.take(x,BODY_JOINTS,axis=1) for x in full]
        values=extra_metrics(p,g,fps=20.)
        error=abs(float(np.linalg.norm(p-g,axis=-1).mean()*100)-original[index]['mpjpe_cm'])
        maximum=max(maximum,error);assert error<1e-4
        assert extra_metrics(g,g)['sjpe']==0
        measured.append(dict(id=row['id'],seed=row['seed'],**values))
        if row['id'] in selected:np.savez(folder/'qualitative_joints'/(row['id']+'.npz'),prediction=full[0],gt=full[1])
    atomic_json(folder/'extra_motion_per_case.json',measured)
    summary={k:float(np.mean([r[k] for r in measured])) for k in values}
    summary.update(upe_cm=summary['upe_m']*100,lpe_cm=summary['lpe_m']*100)
    atomic_json(folder/'extra_motion_protocol.json',dict(protocol=protocol(),maximum_mpjpe_difference_cm=maximum,cases=71))
    return summary
