"""Short, paired FP32 local-attention benchmark; never changes a training run."""
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

from modality_bridge import LocalModalityBridge, grouped_joint_attention
from scripts.phase3_performance import attention_bias, execution_policy
from motion_expert_phase3_edge_joint.test_temporally_local_attention import metadata


def gpu_ownership(index, share):
    text=subprocess.check_output(['nvidia-smi','-i',str(index),'--query-compute-apps=pid',
                                  '--format=csv,noheader,nounits'],text=True).strip()
    pids=[int(line) for line in text.splitlines() if line.strip() and int(line)!=os.getpid()]
    if pids and not share:raise RuntimeError('GPU is occupied; explicit --share-owned-gpu is required')
    for pid in pids:
        if Path(f'/proc/{pid}').stat().st_uid!=os.getuid():
            raise RuntimeError(f'GPU has another user process {pid}; refusing benchmark')
    return pids


def atomic_json(path,payload):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload,indent=2)+'\n');temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--share-owned-gpu',action='store_true')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=6)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--stream-dtype',choices=['float32','bfloat16'],default='float32')
    parser.add_argument('--parameterizations',nargs='+',choices=['shared','mmdit'],default=['shared','mmdit'])
    parser.add_argument('--layers',type=int,default=1)
    parser.add_argument('--checkpoint-bridges',action='store_true')
    parser.add_argument('--fresh-metadata',action='store_true',help='Include first-use layout preparation once per bridge stack')
    args=parser.parse_args()
    host=socket.gethostname()
    if host not in ('lg-gwangmo-4xh200-sep-1','lg-gwangmo-4xh200-sep-2'):
        raise RuntimeError('Expected a Foundary compute node')
    if not args.out.resolve().is_relative_to('/mnt/local'):raise ValueError('Output must be node local')
    if args.out.exists():raise FileExistsError(args.out)
    if args.iterations<1 or args.repeats<2 or not 1<=args.layers<=7:
        raise ValueError('Use positive iterations, at least two paired repeats, and 1..7 layers')
    pids=gpu_ownership(args.gpu,args.share_owned_gpu)
    uuid=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=uuid',
                                  '--format=csv,noheader'],text=True).strip()
    os.environ['CUDA_VISIBLE_DEVICES']=uuid
    torch.set_num_threads(1);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    if torch.cuda.mem_get_info()[0]<16*2**30:raise RuntimeError('Need 16 GiB free for a bounded benchmark')
    torch.manual_seed(341);device=torch.device('cuda',0)
    dtype=getattr(torch,args.stream_dtype)
    meta=metadata()
    for name in vars(meta):
        if isinstance(getattr(meta,name),torch.Tensor):setattr(meta,name,getattr(meta,name).to(device))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    report=dict(host=host,gpu=args.gpu,gpu_uuid=uuid,torch=torch.__version__,
        device=torch.cuda.get_device_name(),concurrent_owned_pids=pids,
        contention=('shared GPU; paired alternating measurements' if pids else 'no concurrent GPU process before benchmark'),
        attention_dtype='float32',parameter_dtype='float32',stream_dtype=args.stream_dtype,
        tf32=False,tokens=dict(video=1600,motion=98),hidden=2048,heads=16,head_dim=128,
        iterations=args.iterations,repeats=args.repeats,layers=args.layers,
        checkpoint_bridges=args.checkpoint_bridges,fresh_metadata=args.fresh_metadata,rows=[],parity={})
    started=time.monotonic()

    def measure(label,inputs,model,runner):
        parameters=list(model.parameters()) if model is not None else []
        def clean():
            for p in parameters+inputs:p.grad=None
        def run(variant,backward,capture=False):
            clean()
            with torch.set_grad_enabled(backward):
                output=runner(variant)
                if backward:sum(x.float().square().mean() for x in output).backward()
            if capture:
                return [x.detach().clone() for x in output], [p.grad.detach().clone() for p in parameters+inputs]
        # Real nonzero-gate forward/backward parity before any timing.
        with execution_policy():expected,grads=run('dense',True,True)
        with execution_policy(grouped_attention=True):actual,ggrads=run('grouped',True,True)
        max_output=max(float((a-b).abs().max()) for a,b in zip(expected,actual))
        max_grad_relative=max(float((a.float()-b.float()).norm()/a.float().norm().clamp_min(1e-12))
                              for a,b in zip(grads,ggrads))
        tolerance=5e-3 if dtype==torch.bfloat16 and model is not None else 1e-4
        if max_grad_relative>tolerance:raise AssertionError((label,'gradient parity',max_grad_relative))
        for a,b in zip(expected,actual):
            torch.testing.assert_close(a,b,rtol=.016 if a.dtype==torch.bfloat16 else 1e-4,
                                       atol=.016 if a.dtype==torch.bfloat16 else 1e-5)
        report['parity'][label]=dict(max_output_abs=max_output,max_gradient_relative_l2=max_grad_relative)
        del expected,actual,grads,ggrads
        for backward in (False,True):
            samples={v:[] for v in ('dense','grouped')};wall={v:[] for v in samples};memory={v:[] for v in samples}
            for repeat in range(args.repeats):
                order=('dense','grouped') if repeat%2==0 else ('grouped','dense')
                for variant in order:
                    with execution_policy(grouped_attention=variant=='grouped'):
                        for _ in range(3):run(variant,backward)
                        torch.cuda.synchronize();clean()
                        baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                        for _ in range(args.iterations):
                            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            before=time.monotonic();start.record();run(variant,backward);end.record();end.synchronize()
                            samples[variant].append(start.elapsed_time(end));wall[variant].append((time.monotonic()-before)*1000)
                        memory[variant].append((torch.cuda.max_memory_allocated()-baseline)/2**20)
            row=dict(workload=label,phase='forward_backward' if backward else 'forward',
                     variants={v:dict(median_ms=statistics.median(samples[v]),min_ms=min(samples[v]),
                         median_wall_ms=statistics.median(wall[v]),peak_increment_mib=max(memory[v]),
                         samples_ms=samples[v]) for v in samples})
            row['speedup']=row['variants']['dense']['median_ms']/row['variants']['grouped']['median_ms']
            report['rows'].append(row);atomic_json(args.out,report)
            print(json.dumps({k:row[k] for k in ('workload','phase','speedup')}),flush=True)
        clean()

    q,k,v=[torch.randn(16,1698,128,device=device,requires_grad=True) for _ in range(3)]
    bias=attention_bias(meta,device,'bidirectional')
    def attention(variant):
        if variant=='grouped':return (grouped_joint_attention(q,k,v,meta,'bidirectional'),)
        scores=q@k.transpose(-2,-1)*(128**-.5)
        return (torch.softmax(scores+bias,-1)@v,)
    measure('attention_only',[q,k,v],None,attention)
    del q,k,v,bias
    for kind in args.parameterizations:
        model=nn.ModuleList([LocalModalityBridge(2048,16,128,'bidirectional',kind).to(device)
                             for _ in range(args.layers)])
        with torch.no_grad():
            for block in model:block.gate_g.fill_(.1);block.gate_m.fill_(.1)
        g=torch.randn(1600,2048,device=device,dtype=dtype,requires_grad=True)
        m=torch.randn(98,2048,device=device,dtype=dtype,requires_grad=True)
        def run_bridges(_variant):
            states=g,m
            batch_meta=replace(meta) if args.fresh_metadata else meta
            for block in model:
                if args.checkpoint_bridges and torch.is_grad_enabled():
                    states=checkpoint(block,*states,batch_meta,use_reentrant=False)
                else:
                    states=block(*states,batch_meta)
            return states
        label='bridge_'+kind if args.layers==1 else f'bridge_{kind}_stack{args.layers}'
        measure(label,[g,m],model,run_bridges)
        del model,g,m
    report['wall_seconds']=time.monotonic()-started
    report['concurrent_owned_pids_after']=gpu_ownership(args.gpu,True)
    report['complete']=True;atomic_json(args.out,report)
    print('COMPLETE '+str(args.out),flush=True)


if __name__=='__main__':main()
