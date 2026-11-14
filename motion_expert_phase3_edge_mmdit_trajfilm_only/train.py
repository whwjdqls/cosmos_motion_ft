"""Two-GPU, global32, FiLM-only camera-to-motion post-training with recovery."""
from pathlib import Path
from datetime import timedelta
import argparse,fcntl,hashlib,json,os,random,signal,time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from .data import CameraOnlyDataset,collate
from .runtime import (read,sha,atomic_json,load_base,optimizer_for,lr_for,noisy,motion_loss,frozen_versions,assert_frozen,new_ema,update_ema,save_checkpoint,restore,film_state)
from .validation import validate
STOP=False

def stop(*args):
    global STOP;STOP=True

def objects(value,world):
    if world==1:return [value]
    out=[None]*world;dist.all_gather_object(out,value);return out

def checkpoint(path,model,optimizer,ema,step,recipe,contract,rank,world):
    from checkpoint import capture_rng_state
    rng=objects(capture_rng_state(),world)
    if rank==0:save_checkpoint(path,model,optimizer,ema,step,recipe,contract,rng)
    if world>1:dist.barrier()

def preflight(model,ddp,dataset,recipe,contract,plan,mean,std,device,rank,world):
    folder=Path(plan['preflight']);folder.mkdir(parents=True,exist_ok=True)
    batch=collate([dataset[(rank*16+i,rank*16+i)] for i in range(16)]);inputs,target=noisy(batch,device)
    versions=frozen_versions(model);native_calls=[0]
    def forbidden(*a,**k):native_calls[0]+=1;raise RuntimeError('Camera-only must not execute the native generator')
    hook=model.base.native.register_forward_pre_hook(forbidden)
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        initial=model(**inputs);disabled=model(**inputs,film_enabled=False)
        torch.testing.assert_close(initial,disabled,rtol=0,atol=0)
        other={**inputs,'camera_features':inputs['camera_features']+100}
        torch.testing.assert_close(model(**other),initial,rtol=0,atol=0)
        reference=model.base.motion(reasoner_inputs=model.base.motion.prepare_conditions(['']*len(target)),x_sigma=inputs['motion'],sigma=inputs['motion_sigma'],neutral_joints=inputs['neutral'],motion_pad_mask=torch.zeros(len(target),97,device=device,dtype=torch.bool))
        torch.testing.assert_close(initial,reference,rtol=0,atol=0)
    optimizer=optimizer_for(model,recipe);ema=new_ema(model);model.train();losses=[]
    for step in range(1,5):
        for g in optimizer.param_groups:g['lr']=lr_for(step,recipe)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=ddp(**inputs);loss,_=motion_loss(pred,target,mean,std,recipe['motion_weights'])
        assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.film.parameters(),1.,error_if_nonfinite=True)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.film.parameters())
        if step==1:assert all(block.mlp[-1].weight.grad.abs().sum()>0 for block in model.film.values())
        optimizer.step();update_ema(model,ema);assert_frozen(model,versions);losses.append(float(loss.detach()))
    expected=film_state(model);expected_ema={k:v.detach().cpu().clone() for k,v in ema.items()};path=folder/'roundtrip.pt';checkpoint(path,model,optimizer,ema,4,recipe,contract,rank,world)
    assert restore(path,model,optimizer,ema,contract,rank,world)==4
    for k,v in model.film.state_dict().items():torch.testing.assert_close(v.cpu(),expected[k],rtol=0,atol=0)
    for k,v in ema.items():torch.testing.assert_close(v.cpu(),expected_ema[k],rtol=0,atol=0)
    digest=hashlib.sha256()
    for value in expected.values():digest.update(value.numpy().tobytes())
    assert len(set(objects(digest.hexdigest(),world)))==1
    for g in optimizer.param_groups:g['lr']=lr_for(5,recipe)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):pred=ddp(**inputs);loss,_=motion_loss(pred,target,mean,std,recipe['motion_weights'])
    loss.backward();torch.nn.utils.clip_grad_norm_(model.film.parameters(),1.,error_if_nonfinite=True);optimizer.step();update_ema(model,ema)
    assert_frozen(model,versions);assert native_calls[0]==0;hook.remove()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        model.eval();a=model(**inputs);b=model(**{**inputs,'camera_features':inputs['camera_features']+1})
        end_to_end_camera_difference=float((a-b).abs().max())
        # The first warmup updates can be below BF16 rounding. Check that the
        # trained FP32 camera path is active without requiring premature BF16 changes.
        h=torch.randn(len(target),97,2048,device=device,dtype=torch.float32)
        with torch.autocast('cuda',enabled=False):context=model.base.motion.heads.time_embedder(inputs['motion_sigma'].float()).float()
        block=model.film[str(model.layers[0])]
        assert (block(h,inputs['camera_features'],context)-block(h,inputs['camera_features']+1,context)).abs().max()>0
    if rank==0:atomic_json(Path(plan['acceptance']),dict(passed=True,world_size=world,contract=contract,checks=['zero_film_exact_identity','phase2_reference_bitwise_parity','camera_only_no_native_generator_or_video','all_film_gradients_finite','all_base_weights_frozen','ddp_identical_film','optimizer_ema_rng_save_reload_continue','camera_path_active_FP32_after_update'],end_to_end_camera_difference=end_to_end_camera_difference,losses=losses))
    if world>1:dist.barrier()

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--preflight',action='store_true');p.add_argument('--resume',default='auto');args=p.parse_args();plan=read(args.plan);recipe=plan['recipe']
    assert os.uname().nodename==plan['node'] and recipe['tasks']=={'camera2motion':1.} and recipe['steps']==50000
    rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1));assert world==2
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    torch.manual_seed(0);torch.cuda.manual_seed_all(0);random.seed(0);np.random.seed(0)
    dataset=CameraOnlyDataset(plan['motion_cache'],plan['trajectory_cache'],plan.get('cache_pairing_receipt'));model=load_base(plan,device)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==15663872
    import config
    source=model.base.provenance['phase2_contract'];assert sha(config.MOTION_STATS_MEAN)==source['motion_mean'] and sha(config.MOTION_STATS_STD)==source['motion_std']
    mean=torch.from_numpy(np.load(config.MOTION_STATS_MEAN)).to(device).float();std=torch.from_numpy(np.load(config.MOTION_STATS_STD)).to(device).float()
    contract=dict(recipe=recipe,base_sha256=plan['base_sha256'],base_step=200000,base_weights='ema',phase1=model.base.provenance['phase1_metadata_sha256'],phase2_contract=source,motion_index_sha256=sha(Path(plan['motion_cache'])/'index.json'),trajectory_index_sha256=sha(Path(plan['trajectory_cache'])/'index.json'),cache_pairing_sha256=sha(plan['cache_pairing_receipt']),validation_cases_sha256=sha(plan['validation_cases']),package_source_sha256=plan['package_source_sha256'],trainable_parameters=15663872)
    ddp=DDP(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False)
    if args.preflight:
        preflight(model,ddp,dataset,recipe,contract,plan,mean,std,device,rank,world);dist.destroy_process_group();return
    acceptance=read(plan['acceptance']);assert acceptance['passed'] and acceptance['world_size']==2 and acceptance['contract']==contract
    out=Path(plan['out']);out.mkdir(parents=True,exist_ok=True);writer=None
    if rank==0:writer=(out/'.writer.lock').open('a');fcntl.flock(writer,fcntl.LOCK_EX|fcntl.LOCK_NB);atomic_json(out/'contract.json',contract)
    dist.barrier();optimizer=optimizer_for(model,recipe);ema=new_ema(model);start=0
    if args.resume=='auto':
        pointer=out/'checkpoints/latest.json';resume=out/'checkpoints'/read(pointer)['file'] if pointer.exists() else None
    else:resume=None if args.resume=='none' else Path(args.resume)
    if resume is not None:start=restore(resume,model,optimizer,ema,contract,rank,world)
    else:torch.manual_seed(rank);torch.cuda.manual_seed_all(rank);random.seed(rank);np.random.seed(rank)
    from motion_expert_phase3_edge_joint.data import GlobalDrawSampler
    sampler=GlobalDrawSampler(len(dataset),start,recipe['steps'],16,rank,world)
    loader=DataLoader(dataset,batch_sampler=sampler,collate_fn=collate,num_workers=2,persistent_workers=True,prefetch_factor=2,multiprocessing_context='spawn',pin_memory=True,generator=torch.Generator().manual_seed(rank));iterator=iter(loader)
    versions=frozen_versions(model);run=None
    if rank==0:
        import wandb
        run=wandb.init(entity='jungbinc-upenn',project='cosmos-motion-ft',id=plan['wandb_id'],name=plan['wandb_name'],dir=str(out),resume='allow',config=contract)
        run.define_metric('train/step');run.define_metric('train/*',step_metric='train/step');run.define_metric('validation/step');run.define_metric('validation/*',step_metric='validation/step');run.define_metric('validation_preview/*',step_metric='validation/step')
        atomic_json(out/'wandb_identity.json',dict(id=run.id,url=run.url,name=run.name))
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(sig,stop)
    if start==0:checkpoint(out/'checkpoints/step_000000000.pt',model,optimizer,ema,0,recipe,contract,rank,world)
    validate(model,ema,plan,start,rank,world,device,mean,std,run)
    model.train();step=start
    for step in range(start+1,recipe['steps']+1):
        before=time.monotonic();batch=next(iterator);inputs,target=noisy(batch,device);optimizer.zero_grad(set_to_none=True)
        lr=lr_for(step,recipe)
        for g in optimizer.param_groups:g['lr']=lr
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=ddp(**inputs);loss,terms=motion_loss(pred,target,mean,std,recipe['motion_weights'])
        finite=torch.tensor(int(torch.isfinite(loss)),device=device);dist.all_reduce(finite,op=dist.ReduceOp.MIN)
        if not finite.item():raise FloatingPointError('Nonfinite FiLM training loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.film.parameters(),1.,error_if_nonfinite=True);optimizer.step();update_ema(model,ema,recipe['ema_decay']);assert_frozen(model,versions)
        if step%10==0 or step==start+1:
            averaged=loss.detach().clone();dist.all_reduce(averaged);averaged/=world;torch.cuda.synchronize()
            if rank==0:
                values=dict(step=step,stage='training',task='camera2motion',loss=float(averaged),lr=lr,grad_norm=float(norm),seconds=time.monotonic()-before,ema_decay=recipe['ema_decay'],ema_updates=step,global_batch=32,accumulation=1)
                with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(values)+'\n')
                atomic_json(out/'status.json',values);run.log({'train/'+k:v for k,v in values.items() if isinstance(v,(int,float))})
        stop_flag=torch.tensor(int(STOP),device=device);dist.all_reduce(stop_flag,op=dist.ReduceOp.MAX)
        save=step%5000==0 or step==recipe['steps'] or stop_flag.item()
        if save:
            filename=f'step_{step:09d}.pt' if step%5000==0 else 'recovery_latest.pt';checkpoint(out/'checkpoints'/filename,model,optimizer,ema,step,recipe,contract,rank,world)
            if rank==0:atomic_json(out/'checkpoints/latest.json',dict(file=filename,step=step))
        if step%5000==0:validate(model,ema,plan,step,rank,world,device,mean,std,run);assert_frozen(model,versions)
        if stop_flag.item():break
    if rank==0:
        atomic_json(out/'status.json',dict(stage='stopped' if STOP else 'complete',step=step,target=50000));run.finish();writer.close()
    dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
