"""Inference-only joint or image-only CFG on a matched diagnostic cohort.

Current TI2M CFG keeps the image in its null branch. Joint CFG instead uses the
trained native T2M empty-text branch: no text and no image. This also changes
the reasoner prompt layout; it is not a layout-matched image-only intervention.
Image-only CFG retains the caption in the native T2M reference branch.
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch
from diagnose_image_conditioning import atomic_json,sha,physics_setup

SCALES=(2.,1.,2.,3.,5.)


def variants(args):
    return ('text_cfg2',)+tuple(f'{args.target}_cfg{n}' for n in (1,2,3,5))


@torch.inference_mode()
def run(args):
    import config
    from checkpoint import configure_attention_from_checkpoint,load_checkpoint
    from model import EdgePhase2MotionExpert
    from flow import sample_x0_unipc
    from decode_uniego_torch import decode_joints
    from eval_motion_recon import joint_metrics
    source=json.loads((args.source/'configuration.json').read_text())
    names=variants(args)
    cases=json.loads((args.source/'cases.json').read_text())
    assert len(cases)==32 and len({c['uuid'] for c in cases})==32
    assert sha(source['checkpoint'])==source['checkpoint_sha256']
    contract=configure_attention_from_checkpoint(source['checkpoint'])
    assert contract==source['checkpoint_contract']
    for key,path in [('motion_mean',config.MOTION_STATS_MEAN),('motion_std',config.MOTION_STATS_STD)]:
        assert sha(path)==contract[key]
    cfg={'source':str(args.source),'source_configuration_sha256':sha(args.source/'configuration.json'),
         'cases_sha256':sha(args.source/'cases.json'),'checkpoint':source['checkpoint'],
         'checkpoint_sha256':source['checkpoint_sha256'],'checkpoint_contract':contract,
         'cases':32,'seeds':source['sampling_seeds'],'variants':list(names),'guidance':list(SCALES),
         'sampling':source['sampling'],'source_sha256':sha(__file__),
         'target':args.target,
         'formula':'x0_reference + scale * (x0_text_image - x0_reference)',
         'text_cfg2_null':'TI2M empty caption + SAME observed image',
         'sweep_reference':('native T2M SAME caption, no image' if args.target=='image' else 'native T2M empty caption, no image')+' (different native prompt layout)',
         'controls':'All variants share target cases, caption, correct image, shape, length and initial noise.',
         'scope':'Matched 32-case exploratory preview, separate from the full 11892-case benchmark. No reliable FID claim.'}
    if (args.out/'configuration.json').exists():assert json.loads((args.out/'configuration.json').read_text())==cfg
    else:atomic_json(args.out/'configuration.json',cfg)
    atomic_json(args.out/'cases.json',cases)
    model=EdgePhase2MotionExpert(device='cuda',dtype=torch.bfloat16)
    payload=load_checkpoint(source['checkpoint'],model=model);assert payload['step']==200000
    del payload
    model.eval().requires_grad_(False)
    mean=torch.from_numpy(np.load(config.MOTION_STATS_MEAN)).cuda().float()
    std=torch.from_numpy(np.load(config.MOTION_STATS_STD)).cuda().float()
    physics=physics_setup()
    scales=torch.tensor(SCALES,device='cuda')[:,None,None]
    started=time.monotonic()
    for case in cases:
        slot=case['slot'];dest=args.out/f'case_{slot:03d}';dest.mkdir(exist_ok=True)
        if (dest/'results.json').exists():continue
        p=args.source/'inputs'/f'case_{slot:03d}.npz'
        assert sha(p)==source['input_hashes'][p.name]
        with np.load(p,allow_pickle=False) as saved:data={k:np.array(saved[k]) for k in saved.files}
        image=torch.from_numpy(data['image'])
        neutral=torch.from_numpy(data['neutral']).cuda().float()[None].expand(5,-1,-1)
        pad=torch.zeros((5,97),device='cuda',dtype=torch.bool)
        cond=model.prepare_conditions([case['caption']]*5,modes=['textimg2motion']*5,reasoner_images=[image]*5)
        reference_caption=case['caption'] if args.target=='image' else ''
        null=model.prepare_conditions(['']+[reference_caption]*4,modes=['textimg2motion']+['text2motion']*4,reasoner_images=[image]+[None]*4)
        audit={'conditional_tokens':[len(c['input_ids']) for c in cond],
               'null_tokens':[len(c['input_ids']) for c in null],
               'conditional_image_tokens':[int(c.get('visual_pos_mask',torch.zeros(1)).sum()) for c in cond],
               'null_image_tokens':[int(c.get('visual_pos_mask',torch.zeros(1)).sum()) for c in null],
               'reference_caption':reference_caption,'caption_retained_in_reference':args.target=='image'}
        assert len(set(audit['conditional_tokens']))==1
        assert all(v>0 for v in audit['conditional_image_tokens'])
        assert audit['null_image_tokens'][0]>0 and audit['null_image_tokens'][1:]==[0,0,0,0]
        expected=model._edge.tokenize_text(reference_caption)
        assert all(torch.equal(c['input_ids'].cpu(),expected.cpu()) for c in null[1:])
        gt=torch.from_numpy(data['motion']).cuda().float()[None]
        gt_raw=gt*std+mean;gt_joints=decode_joints(gt_raw);gt_np=gt_joints[0].cpu().numpy()
        gt_metrics=joint_metrics(gt_np,gt_np)
        gt_metrics.update(feat_mse=0.,first_frame_mpjpe_m=0.,**physics(gt_joints,gt_raw))
        def predict(context,x,sigma):
            return model(reasoner_inputs=context,x_sigma=x,sigma=sigma,neutral_joints=neutral,motion_pad_mask=pad).float()
        def guided(x,sigma):
            c=predict(cond,x,sigma);u=predict(null,x,sigma)
            return torch.where(scales==1,c,u+scales*(c-u))
        measurements=[]
        for seed in cfg['seeds']:
            noise=torch.randn((1,97,283),generator=torch.Generator().manual_seed(seed)).cuda().expand(5,-1,-1).contiguous()
            # Guidance is applied exactly once in guided(); the solver is unchanged.
            sample=sample_x0_unipc(guided,T=97,motion_dim=283,steps=35,guidance=1.,batch=5,device='cuda',initial_noise=noise,shift=3.)
            assert torch.isfinite(sample).all()
            raw=sample*std+mean;joints=decode_joints(raw)
            arrays={'gt_features':gt_raw[0].cpu().numpy(),'gt_joints':gt_np}
            for i,name in enumerate(names):
                m=joint_metrics(joints[i].cpu().numpy(),gt_np)
                m.update(physics(joints[i:i+1],raw[i:i+1]))
                m.update(variant=name,seed=seed,feat_mse=float((sample[i:i+1]-gt).square().mean()),
                         first_frame_mpjpe_m=float((joints[i,0]-gt_joints[0,0]).norm(dim=-1).mean()),
                         baseline_joint_change_m=float((joints[i]-joints[0]).norm(dim=-1).mean()))
                measurements.append(m)
                arrays[name+'_features']=raw[i].cpu().numpy();arrays[name+'_joints']=joints[i].cpu().numpy()
            np.savez_compressed(dest/f'seed_{seed}.npz',**arrays)
        atomic_json(dest/'results.json',{'case':case,'GT':gt_metrics,'measurements':measurements,'audit':audit})
        done=len(list(args.out.glob('case_*/results.json')))
        atomic_json(args.out/'progress.json',{'completed':done,'total':len(cases),'seconds':time.monotonic()-started})
        print(f'[{args.target}-CFG] {done}/{len(cases)} cases, all five settings and three seeds',flush=True)
        if slot<6:render_case(args,case)
    summarize(args,cases,cfg)


def render_case(args,case):
    from PIL import Image,ImageDraw
    from render_motion import render_single,head_rotation_from_features,write_mp4
    slot=case['slot'];gallery=args.out/'gallery';gallery.mkdir(exist_ok=True)
    with np.load(args.out/f'case_{slot:03d}'/'seed_0.npz',allow_pickle=False) as z:data={k:np.array(z[k]) for k in z.files}
    title='Image-only' if args.target=='image' else 'Joint text+image'
    labels=[('gt','GT','tab:blue'),('text_cfg2','Current: text CFG 2','tab:green'),
            (f'{args.target}_cfg1','CFG 1 (no CFG)','tab:gray'),(f'{args.target}_cfg2',f'{title} CFG 2','tab:orange'),
            (f'{args.target}_cfg3',f'{title} CFG 3','tab:purple'),(f'{args.target}_cfg5',f'{title} CFG 5','tab:red')]
    panels=[render_single(data[name+'_joints'],caption=title,color=color,width=360,height=360,frame_stride=2,max_frames=97,
                         head_rotation0=head_rotation_from_features(data[name+'_features'])) for name,title,color in labels]
    grid=np.concatenate([np.concatenate(panels[:3],axis=2),np.concatenate(panels[3:],axis=2)],axis=1)
    banner=Image.new('RGB',(1080,200),(245,245,245));draw=ImageDraw.Draw(banner)
    with np.load(args.source/'inputs'/f'case_{slot:03d}.npz',allow_pickle=False) as z:pixels=np.array(z['image']).transpose(1,2,0)
    banner.paste(Image.fromarray(pixels).resize((160,160)),(900,25));draw.rectangle((897,22,1063,188),outline='green',width=3)
    draw.text((20,12),'Same observed image, caption, skeleton and initial noise across settings',fill='black')
    import textwrap
    draw.multiline_text((20,45),'\n'.join(textwrap.wrap(case['caption'],110)),fill='black',spacing=6)
    frames=np.concatenate([np.broadcast_to(np.asarray(banner),(len(grid),200,1080,3)),grid],axis=1)
    path=gallery/f'case_{slot:03d}.mp4';write_mp4(frames,str(path),fps=10)
    Image.fromarray(frames[0]).save(path.with_suffix('.png'))
    import av
    with av.open(str(path)) as container:assert sum(1 for _ in container.decode(video=0))==len(frames)


def summarize(args,cases,cfg):
    records=[json.loads((args.out/f"case_{c['slot']:03d}"/'results.json').read_text()) for c in cases]
    metrics=('mpjpe_m','pa_mpjpe_m','first_frame_mpjpe_m','feat_mse','foot_skate_from_height','foot_depth_mean_cm','all_feet_above5cm_frame_fraction')
    result={}
    per_case=[]
    for name in ('GT',)+variants(args):
        values=[]
        for record in records:
            selected=[record['GT']] if name=='GT' else [m for m in record['measurements'] if m['variant']==name]
            avg={key:float(np.mean([m[key] for m in selected])) for key in metrics}
            values.append(avg);per_case.append({'slot':record['case']['slot'],'uuid':record['case']['uuid'],'variant':name,**avg})
        result[name]={key:float(np.mean([v[key] for v in values])) for key in metrics}
    branch=('Image-only CFG keeps the same caption in both branches, removing only the image from the native T2M reference.' if args.target=='image' else 'Joint CFG removes both modalities using native T2M empty text.')
    report={'status':'complete','configuration':cfg,'results':result,'cases':len(cases),
            'interpretation':'Exploratory matched sampling. Reconstruction metrics do not measure all valid generative motions. '+branch+' Native prompt layout changes too. Larger differences alone do not establish better conditioning.'}
    atomic_json(args.out/'summary.json',report)
    with (args.out/'per_case.csv').open('w') as handle:
        w=csv.DictWriter(handle,fieldnames=list(per_case[0]));w.writeheader();w.writerows(per_case)
    lines=[f'# {args.target.title()} CFG diagnostic','',report['interpretation'],'',
           '| Condition | MPJPE cm | PA-MPJPE cm | Frame-0 cm | Foot skate | Foot penetration cm |',
           '|---|---:|---:|---:|---:|---:|']
    for name,m in result.items():lines.append(f"| {name} | {m['mpjpe_m']*100:.2f} | {m['pa_mpjpe_m']*100:.2f} | {m['first_frame_mpjpe_m']*100:.2f} | {m['foot_skate_from_height']:.6f} | {m['foot_depth_mean_cm']:.4f} |")
    (args.out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    gallery=args.out/'gallery'
    gallery.mkdir(exist_ok=True)
    (gallery/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><h1>Matched {args.target} CFG sweep</h1>'+''.join(f'<h2>Case {c["slot"]}</h2><video controls width="1080" src="case_{c["slot"]:03d}.mp4"></video>' for c in cases[:6]))
    print('\n'.join(lines),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--target',choices=['joint','image'],default='joint')
    p.add_argument('--memory-fraction',type=float,default=1.0)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or 'login' in socket.gethostname() or socket.gethostname().split('.')[0] in {'cs-gpu-01','node37'}:
        raise RuntimeError('Use an owned compute GPU')
    args.out.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    if not 0<args.memory_fraction<=1:raise ValueError('memory fraction must be in (0,1]')
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    run(args)


if __name__=='__main__':main()
