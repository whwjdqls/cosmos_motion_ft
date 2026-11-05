"""Paired grouped FP32/BF16 benchmarks using the current shape-excluded bridge."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import time

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from modality_bridge import LocalModalityBridge
from scripts.phase3_performance import execution_policy
from scripts.benchmark_grouped_bridge import gpu_ownership, atomic_json
from motion_expert_phase3_edge_joint.test_temporally_local_attention import metadata


MODES=('fp32','bf16_attention','bf16_bridge')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--share-owned-gpu',action='store_true')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=6)
    parser.add_argument('--repeats',type=int,default=4)
    parser.add_argument('--layers',type=int,default=7)
    args=parser.parse_args()
    host=socket.gethostname()
    if host not in ('lg-gwangmo-4xh200-sep-1','lg-gwangmo-4xh200-sep-2'):raise RuntimeError('Wrong node')
    if not args.out.resolve().is_relative_to('/mnt/local'):raise ValueError('Use node-local output')
    if args.out.exists():raise FileExistsError(args.out)
    if min(args.iterations,args.repeats,args.layers)<1 or args.layers>7:raise ValueError('Invalid benchmark size')
    peers=gpu_ownership(args.gpu,args.share_owned_gpu)
    uuid=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    os.environ['CUDA_VISIBLE_DEVICES']=uuid
    torch.set_num_threads(1);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.manual_seed(371)
    if torch.cuda.mem_get_info()[0]<16*2**30:raise RuntimeError('Need 16 GiB free')
    device=torch.device('cuda',0)
    meta=metadata()
    for key in vars(meta):
        if isinstance(getattr(meta,key),torch.Tensor):setattr(meta,key,getattr(meta,key).to(device))
    model=nn.ModuleList([LocalModalityBridge(2048,16,128,'bidirectional','mmdit').to(device) for _ in range(args.layers)])
    with torch.no_grad():
        for b in model:b.gate_g.fill_(.1);b.gate_m.fill_(.1)
    g=torch.randn(1600,2048,device=device,dtype=torch.bfloat16,requires_grad=True)
    m=torch.randn(98,2048,device=device,dtype=torch.bfloat16,requires_grad=True)
    parameters=list(model.parameters());leaves=parameters+[g,m]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    report=dict(host=host,gpu=args.gpu,gpu_uuid=uuid,torch=torch.__version__,device=torch.cuda.get_device_name(),
        concurrent_owned_pids=peers,contention=bool(peers),shape='excluded_from_bridge',grouped=True,backend='manual',
        stream_dtype='bfloat16',master_parameter_dtype='float32',softmax_dtype='float32',tf32=False,
        layers=args.layers,hidden=2048,heads=16,head_dim=128,video_tokens=1600,motion_frames=97,
        checkpoint_bridges=True,fresh_metadata=True,iterations=args.iterations,repeats=args.repeats,
        scope='synthetic bridge stack only; specialists/data/optimizer/DDP excluded from timing',accuracy={},timing={})
    start_wall=time.monotonic()
    def clean():
        for p in leaves:p.grad=None
    def run(backward,capture=False):
        clean();batch_meta=replace(meta);states=g,m
        with torch.set_grad_enabled(backward):
            for block in model:
                states=checkpoint(block,*states,batch_meta,use_reentrant=False) if backward else block(*states,batch_meta)
            if backward or capture:loss=sum(x.float().square().mean() for x in states)
            if backward:loss.backward()
        if capture:
            return [x.detach().clone() for x in states], [p.grad.detach().clone() for p in leaves], float(loss.detach())

    # Match weights and inputs across all three paths; no optimizer updates here.
    with execution_policy(grouped_attention=True,precision='fp32'):reference,gradients,ref_loss=run(True,True)
    for mode in MODES:
        seen={};handles=[]
        for name,layer in model[0].named_modules():
            if isinstance(layer,(nn.Linear,nn.LayerNorm)):
                handles.append(layer.register_forward_hook(lambda module,args,out,key=name:seen.__setitem__(key,str(out.dtype))))
        with execution_policy(grouped_attention=True,precision=mode):out,grads,loss=run(True,True)
        for h in handles:h.remove()
        diff=sum((a.float()-b.float()).square().sum() for a,b in zip(gradients,grads))
        denom=sum(a.float().square().sum() for a in gradients).clamp_min(1e-20)
        grad_relative=float((diff/denom).sqrt())
        output_diff=sum((a.float()-b.float()).square().sum() for a,b in zip(reference,out))
        output_denom=sum(a.float().square().sum() for a in reference).clamp_min(1e-20)
        finite=all(bool(torch.isfinite(x).all()) for x in out+grads)
        shape_exact=torch.equal(out[1][0].view(torch.uint8),m.detach()[0].view(torch.uint8))
        report['accuracy'][mode]=dict(loss=loss,loss_relative=abs(loss-ref_loss)/abs(ref_loss),
            output_max_abs=max(float((a.float()-b.float()).abs().max()) for a,b in zip(reference,out)),
            output_relative_l2=float((output_diff/output_denom).sqrt()),gradient_relative_l2=grad_relative,
            finite=finite,shape_bit_exact=shape_exact,operator_output_dtypes=seen,
            master_dtypes=sorted({str(p.dtype) for p in parameters}),gradient_dtypes=sorted({str(p.grad.dtype) for p in parameters}))
        if not finite or not shape_exact or grad_relative>.05:raise AssertionError((mode,report['accuracy'][mode]))
        del out,grads
    del reference,gradients
    atomic_json(args.out,report)
    for backward in (False,True):
        values={mode:[] for mode in MODES};wall={mode:[] for mode in MODES};peak={mode:[] for mode in MODES}
        for repeat in range(args.repeats):
            order=MODES[repeat%3:]+MODES[:repeat%3]
            if repeat%2:order=order[::-1]
            for mode in order:
                with execution_policy(grouped_attention=True,precision=mode):
                    for _ in range(2):run(backward)
                    torch.cuda.synchronize();clean()
                    baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                    for _ in range(args.iterations):
                        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        before=time.monotonic();a.record();run(backward);b.record();b.synchronize()
                        values[mode].append(a.elapsed_time(b));wall[mode].append((time.monotonic()-before)*1000)
                    peak[mode].append((torch.cuda.max_memory_allocated()-baseline)/2**20)
        name='forward_backward' if backward else 'forward'
        report['timing'][name]={mode:dict(median_ms=statistics.median(values[mode]),
            median_wall_ms=statistics.median(wall[mode]),min_ms=min(values[mode]),peak_increment_mib=max(peak[mode]),
            samples_ms=values[mode],speedup=statistics.median(values['fp32'])/statistics.median(values[mode])) for mode in MODES}
        atomic_json(args.out,report);print(json.dumps({name:report['timing'][name]}),flush=True)

    # Brief disposable optimizer sanity checks, separate from timing/accuracy.
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    report['optimizer_smoke']={}
    for mode in MODES:
        model.load_state_dict(initial)
        optimizer=torch.optim.AdamW(parameters,lr=2e-4,betas=(.9,.99),eps=1e-8,weight_decay=0)
        losses=[]
        with execution_policy(grouped_attention=True,precision=mode):
            for _ in range(3):
                _,grads,loss=run(True,True)
                del grads
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
                optimizer.step();losses.append(loss)
        report['optimizer_smoke'][mode]=dict(losses=losses,loss_decreased=losses[-1]<losses[0],
            optimizer_tensor_dtypes=sorted({str(v.dtype) for state in optimizer.state.values() for v in state.values() if isinstance(v,torch.Tensor)}))
        if not losses[-1]<losses[0]:raise AssertionError((mode,'synthetic loss did not decrease'))
        del optimizer
    clean();report['concurrent_owned_pids_after']=gpu_ownership(args.gpu,True)
    report['wall_seconds']=time.monotonic()-start_wall;report['complete']=True
    atomic_json(args.out,report);print('COMPLETE '+str(args.out),flush=True)


if __name__=='__main__':main()
