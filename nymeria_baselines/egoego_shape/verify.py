"""Real geometry and two-GPU model acceptance before fresh production."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from scipy.spatial.transform import Rotation

from nymeria_baselines.body_shape import BODY_JOINTS
from motion_expert_joint_attention.camera_head_recanonicalization import decode_uniego
from .data import CACHE, MotionDataset, open_arrays, require_compute
from .model import CameraShapeDiffusion, ShapeDenoiser, native
from .train import contract, save_json


def geometry(stats):
    reports=[]
    for split in ['train','val','test']:
        ds=MotionDataset(stats,split)
        for index in [0,len(ds)//2,len(ds)-1]:
            row=ds.rows[index];target,camera,shape,(anchor,origin)=ds.raw(index)
            source=open_arrays(CACHE,row['array'])[0]
            # Decode the complete prefix independently; never reset a cropped
            # motion's delta to infer a world-frame target.
            end=row['start']+97;decoded=decode_uniego(source[:end])
            expected=decoded.world_positions[row['start']:end,BODY_JOINTS].copy()
            floor=0. if row['ground_offset_y'] is None else row['ground_offset_y']
            expected[...,1]-=floor
            recovered=np.einsum('ij,tkj->tki',anchor,target[:,:69].reshape(97,23,3))+origin
            error=float(np.max(np.abs(expected-recovered)))
            assert error<2e-4,(split,index,error)
            expected_rot=decoded.world_rotations[row['start']:end,BODY_JOINTS]
            expected_rot=anchor.T@expected_rot
            r6=target[:,69:].reshape(97,23,6)
            rotation_error=float(np.max(np.abs(expected_rot[...,:2,:].reshape(97,23,6)-r6)))
            assert rotation_error<2e-4,(split,index,rotation_error)
            assert np.max(np.abs(shape.mean(0)))<1e-6
            assert np.max(np.abs(camera[0,[0,2]]))<1e-5
            for value in ds[index].values():assert np.isfinite(value).all()
            reports.append({'split':split,'index':index,'world_position_max_m':error,
                            'global_row6D_max_difference':rotation_error})
    return reports


def main():
    p=argparse.ArgumentParser();p.add_argument('--stats',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();require_compute();rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);assert world==2
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']));device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    torch.set_num_threads(2);dist.init_process_group('nccl')
    torch.manual_seed(62);torch.cuda.manual_seed_all(62)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    reports=geometry(a.stats) if rank==0 else None
    dist.barrier()
    model=CameraShapeDiffusion().to(device)
    # The original denoiser path is bitwise unchanged when shape is omitted.
    reference=native.TransformerDiffusionModel(d_feats=207,d_model=512,n_dec_layers=4,n_head=4,
                                             d_k=256,d_v=256,max_timesteps=99).to(device).eval()
    reference.load_state_dict({k:v for k,v in model.denoise_fn.state_dict().items() if not k.startswith('shape_encoder.')},strict=True)
    model.eval();x=torch.randn(2,97,414,device=device);t=torch.tensor([0,999],device=device)
    with torch.no_grad():assert torch.equal(reference(x,t),model.denoise_fn(x,t))
    del reference,x,t
    ds=MotionDataset(a.stats,'train')
    batch={k:torch.from_numpy(np.stack([ds[rank*32+i][k] for i in range(32)])).to(device) for k in ['motion','camera','shape']}
    assert batch['motion'].shape==(32,97,207) and batch['shape'].shape==(32,30,3)
    # Observed head slots contain only the supplied camera and private noise.
    filler=torch.randn_like(batch['motion']);condition=model.condition(batch['camera'],filler)
    assert torch.equal(condition[:,:,18:21],batch['camera'][:,:,:3])
    assert torch.equal(condition[:,:,105:111],batch['camera'][:,:,3:])
    model.eval();t=torch.full((2,),500,device=device,dtype=torch.long);source=torch.randn(2,97,207,device=device)
    camera=batch['camera'][:2];shape=batch['shape'][:2];filler=filler[:2]
    with torch.no_grad():
        def output(c,s):return model.denoise_fn(torch.cat((source,model.condition(c,filler)),-1),t,shape=s)
        first=output(camera,shape);altered=camera.clone();altered[:,:,0]+=.1
        camera_effect=float((first-output(altered,shape)).abs().max())
        altered_shape=shape.clone();altered_shape[:,0,0]+=.01;altered_shape-=altered_shape.mean(1,keepdim=True)
        shape_effect=float((first-output(camera,altered_shape)).abs().max())
    assert camera_effect>0 and shape_effect>0
    model.train();ddp=DDP(model,device_ids=[device.index]);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.01)
    before=model.denoise_fn.linear_out.weight.detach().clone();losses=[];shape_gradient=None
    for step in range(4):
        optimizer.zero_grad(set_to_none=True);loss=ddp(batch['motion'],batch['camera'],batch['shape'])
        assert torch.isfinite(loss);loss.backward()
        shape_gradient=sum(p.grad.float().square().sum() for p in model.denoise_fn.shape_encoder.parameters()).sqrt().item()
        assert np.isfinite(shape_gradient) and shape_gradient>0
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        losses.append(float(loss));print('PREFLIGHT',rank,step,float(loss),float(norm),flush=True)
    assert not torch.equal(before,model.denoise_fn.linear_out.weight)
    digest=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    all_digests=[None]*world;dist.all_gather_object(all_digests,digest);assert len(set(all_digests))==1
    model.eval()
    generated=model.sample(batch['camera'][:1],batch['shape'][:1],seed=62)
    assert generated.shape==(1,97,207) and torch.isfinite(generated).all()
    receipt={'rank':rank,'native_network_without_shape_bitwise':True,'microbatch':32,
             'losses':losses,'shape_gradient_norm':shape_gradient,'camera_effect':camera_effect,
             'shape_effect':shape_effect,'replica_sha256':digest,
             'native_1000_step_sampling_finite':True,'peak_gib':torch.cuda.max_memory_allocated()/2**30}
    receipts=[None]*world;dist.all_gather_object(receipts,receipt)
    if rank==0:
        a.out.mkdir(parents=True,exist_ok=True)
        save_json(a.out/'ACCEPTED.json',{'passed':True,'contract':contract(a.stats),'geometry':reports,'ranks':receipts})
        print('EGOEGO_ACCEPTANCE_PASSED',flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
