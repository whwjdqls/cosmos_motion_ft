"""Reuse current native objectives and camera acceptance; no legacy weight adapter."""
import argparse
import inspect
import json
import os
from pathlib import Path
import sys
import torch
from motion_expert_phase3_edge_image_joint_camera import train as camera_train
from motion_expert_phase3_edge_joint.settings import RECIPE, sha256, atomic_json
from .policy import configure
from .data import Phase3Dataset
from .attention import attention_scope

def adapt_preflight_source(source):
    if RECIPE['bridge_attention_scope'] != 'global':return source
    old='''        assert torch.equal(mask[:1600, :1600], meta.gen_frame[:, None] == meta.gen_frame[None, :])
        motion_groups = torch.div(meta.motion_frame[1:] + 3, 4, rounding_mode="floor")
        assert torch.equal(mask[1601:, 1601:], motion_groups[:, None] == motion_groups[None, :])'''
    new='''        assert mask[:1600, :1600].all()
        assert mask[1601:, 1601:].all()
        assert mask[0, 1599] and mask[1601, -1]
        from motion_expert_phase3_edge_camera_directional_nojoint.attention import global_attention_mask
        assert torch.equal(mask, global_attention_mask(meta, bridge.conditioning_attention))'''
    if source.count(old)!=1:raise RuntimeError('Review native temporal topology acceptance')
    return source.replace(old,new).replace('causal_Wan_intervals_and_', 'global_temporal_edges_and_')

def prepare_trainer(probe_updates=0):
    from .validation import install
    install()
    # The current native formula/logging already use RECIPE[video_weight].
    # Retain the camera adapter's exact-world preflight and checkpoint gates.
    source=inspect.getsource(camera_train.prepare_trainer)
    marker="exec(compile(source.replace(old, 'reasoner_inputs=model.motion.prepare_conditions(inputs[\"captions\"])'),"
    replacement="exec(compile(adapt_preflight_source(source.replace(old, 'reasoner_inputs=model.motion.prepare_conditions(inputs[\"captions\"])')),"
    if source.count(marker)!=1:raise RuntimeError('Review camera preflight caption adapter')
    source=source.replace(marker,replacement)
    namespace=camera_train.__dict__.copy()
    namespace.update(configure=configure,Phase3Dataset=Phase3Dataset,adapt_preflight_source=adapt_preflight_source)
    exec(compile(source,__file__+':current_camera_adapter','exec'),namespace)
    train=namespace['prepare_trainer'](probe_updates)
    original_acceptance=train.camera_require_acceptance
    def require_acceptance(receipt,fingerprint,world,microbatch):
        original_acceptance(receipt,fingerprint,world,microbatch)
        needed={'camera_batched_sampling_from_noise_initial_image_preserved','no_joint_5050_empty_caption_recipe','scope_'+RECIPE['bridge_attention_scope']}
        if not needed.issubset(receipt.get('checks',[])) or receipt.get('bridge_attention_scope')!=RECIPE['bridge_attention_scope']:
            raise RuntimeError('Missing exact camera-aware sampling and attention-scope acceptance')
    train.camera_require_acceptance=require_acceptance
    original_identity=train.source_identity
    def source_identity(args,dataset,model):
        contract=original_identity(args,dataset,model)
        contract['camera_directional_nojoint_implementation']={p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
        contract['runtime_overlays']['attention_scope']=RECIPE['bridge_attention_scope']
        contract['runtime_overlays']['validation']='batched_native_camera_noise_no_GT_camera'
        return contract
    train.source_identity=source_identity
    original_preflight=train.preflight
    def preflight(args,dataset,model,ddp,mean,std,rank,world,device,fingerprint):
        original_preflight(args,dataset,model,ddp,mean,std,rank,world,device,fingerprint)
        train.barrier()
        receipt=json.loads((args.out/'acceptance.json').read_text())
        train.barrier()
        if rank==0:(args.out/'acceptance.json').unlink()
        train.barrier()
        from .sample import sample
        raw=train.collate([dataset[(rank*2+i,(rank*2+i)%len(dataset))] for i in range(2)])
        values={k:raw[k].to(device).float() for k in ['motion','neutral','video']}
        audits=[]
        with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            for mode in ['video2motion','motimg2video']:
                result=sample(model,mode=mode,caption=['POISON']*2,seed=[700+rank*2,701+rank*2],steps=2,return_all=True,**values)
                assert result['camera'].shape==(2,96,64)
                assert all(bool(torch.isfinite(v).all()) for v in result.values())
                if mode=='motimg2video':torch.testing.assert_close(result['video'][:,:,0],values['video'][:,:,0],rtol=0,atol=0)
                audits.append(dict(task=mode,batch=2,steps=2,camera_from_private_noise=True,finite=all(bool(torch.isfinite(v).all()) for v in result.values())))
        reports=train.all_objects(audits,world)
        if rank==0:
            receipt['camera_validation_sampling_by_rank']=reports
            receipt['bridge_attention_scope']=RECIPE['bridge_attention_scope']
            receipt['checks']+=['camera_batched_sampling_from_noise_initial_image_preserved','no_joint_5050_empty_caption_recipe','scope_'+RECIPE['bridge_attention_scope']]
            atomic_json(args.out/'acceptance.json',receipt)
            print('[phase3/camera-directional] CAMERA_NOJOINT_ACCEPTANCE_PASSED',flush=True)
        model.train();train.barrier()
    train.preflight=preflight
    return train

def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--probe-updates',type=int,default=0)
    args,rest=parser.parse_known_args()
    if args.probe_updates<0:raise ValueError('Invalid probe count')
    configure()
    if '--conditioning-attention' not in rest:rest+=['--conditioning-attention',RECIPE['conditioning_attention']]
    if '--clean-motion-timestep' not in rest:rest+=['--clean-motion-timestep','zero']
    train=prepare_trainer(args.probe_updates)
    sys.argv=[sys.argv[0],*rest]
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from scripts.phase3_performance import execution_policy
    with attention_scope(RECIPE['bridge_attention_scope']),execution_policy(cache_bridge_masks=True,checkpoint_motion=True):
        train.main()

if __name__=='__main__':main()
