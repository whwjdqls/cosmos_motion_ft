"""Text CFG fixed at2; image-increment CFG sweep, matched pure-noise generation."""
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
import evaluate_generation_modalities as modalities
from diagnose_image_conditioning import atomic_json,sha

base=modalities.base
quant=base.quant
IMAGE_WEIGHTS=(0.,.5,1.,2.,3.,5.)
VARIANTS=('text2_img0','text2_img0p5','text2_img1','text2_img2','text2_img3','text2_img5')
read_inputs=base.read_inputs

def combine_cfg(u,t,both,weights):
    text=u+2.*(t-u)
    result=text+weights*(both-t)
    # Preserve the exact special cases without cancellation roundoff.
    result=torch.where(weights==0.,text,result)
    return torch.where(weights==2.,u+2.*(both-u),result)


def prepare(args):
    old=json.loads((args.source_operation/'configuration.json').read_text())
    assert sha(old['manifest'])==old['manifest_sha256']
    assert sha(args.source_operation/'cases.json')==old['cases_sha256']
    cfg=dict(old)
    cfg.update(schema='edge_text2_image_increment_cfg_v1',variants=list(VARIANTS),
               batch_cases=args.batch_cases,shards=args.shards,source_sha256=sha(__file__),
               source_operation=str(args.source_operation),source_configuration_sha256=sha(args.source_operation/'configuration.json'),
               conditioning='Same caption and correct first image; native unconditional / text-only / text+image branches',
               guidance_reference='Native T2M empty text and no image',
               text_guidance=2.,image_guidance=list(IMAGE_WEIGHTS),
               formula='u + 2*(t-u) + image_weight*(both-t)',
               caveat='Text-first conditional decomposition: the image increment includes text-image interaction. Native T2M/TI2M layouts differ. Not a claim of independent causal modality effects.')
    cfg['sampling']=dict(old['sampling'],CFG='text2; image weights0/0.5/1/2/3/5',TI2M_null='native T2M empty text and no image')
    if (args.out/'configuration.json').exists():
        assert json.loads((args.out/'configuration.json').read_text())==cfg
        return
    (args.out/'cases.json').write_bytes((args.source_operation/'cases.json').read_bytes())
    atomic_json(args.out/'configuration.json',cfg)
    (args.out/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    u=torch.randn(6,3,5);t=torch.randn_like(u);both=torch.randn_like(u)
    w=torch.tensor(IMAGE_WEIGHTS)[:,None,None]
    copies=[v.clone() for v in (u,t,both)]
    actual=combine_cfg(u,t,both,w)
    assert torch.equal(actual[0],(u+2*(t-u))[0])
    assert torch.equal(actual[3],(u+2*(both-u))[3])
    for v,c in zip((u,t,both),copies):assert torch.equal(v,c)
    atomic_json(args.out/'guidance_algebra_audit.json',{'status':'passed','weight0_exact_text_cfg2':True,
        'weight2_exact_joint_cfg2':True,'inputs_unchanged':True,'guidance_applied_once':True,
        'cases_manifest_identical':sha(args.out/'cases.json')==old['cases_sha256']})
    print('[prepared] '+json.dumps({'cases':cfg['cases'],'image_weights':IMAGE_WEIGHTS}),flush=True)


@torch.inference_mode()
def worker(args):
    print('[imports] loading the Nymeria evaluator and Edge adapters',flush=True)
    from nymeria_eval import NymeriaTMREmbedder
    from nymeria_data import NymeriaShapeTMRDataset
    from st_dataset import collate_st
    from checkpoint import configure_attention_from_checkpoint, load_checkpoint
    from model import EdgePhase2MotionExpert
    from flow import sample_x0_unipc
    from shape_tmr_eval_common import stable_seed
    from decode_uniego_torch import decode_joints
    from uniego_layout import ground_features
    import eval_phase2_shape_tmr as ref
    import config

    cfg, rows = read_inputs(args)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    assert args.shards == cfg['shards'] and args.batch_cases == cfg['batch_cases']
    source = cfg['reference_request']
    work = args.out / f'shard_{args.shard}'
    work.mkdir(exist_ok=True)
    lock = (work / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    runtime={'script_sha256':sha(__file__),'configuration_sha256':sha(args.out/'configuration.json')}
    if (work/'runtime.json').exists():
        assert json.loads((work/'runtime.json').read_text())==runtime
    else:
        atomic_json(work/'runtime.json',runtime)
    print('[loading] verified request; loading evaluator and generator',flush=True)
    assert sha(cfg['checkpoint']) == cfg['checkpoint_sha256']
    assert sha(source['evaluator_checkpoint']) == cfg['evaluator_sha256']
    configure_attention_from_checkpoint(cfg['checkpoint'])
    for key,path in [('motion_mean',config.MOTION_STATS_MEAN),('motion_std',config.MOTION_STATS_STD)]:
        assert sha(path) == cfg['checkpoint_contract'][key]
    embedder = NymeriaTMREmbedder(source['evaluator_checkpoint'], source['evaluator_stats'], source['text_cache'], 'cuda')
    assert embedder.contract['manifests']['test'] == source['evaluator_training_test_manifest_sha256']
    gt_dataset = NymeriaShapeTMRDataset(cfg['manifest'],embedder.rep,train=False,
                                      src_fps=20.0,rep_fps=embedder.fps)
    model = EdgePhase2MotionExpert(device='cuda', dtype=torch.bfloat16)
    payload = load_checkpoint(cfg['checkpoint'], model=model)
    assert payload['step'] == 200000
    del payload
    model.eval().requires_grad_(False)
    print('[sampling] models loaded; generating from pure noise',flush=True)
    mean = torch.from_numpy(np.load(config.MOTION_STATS_MEAN)).cuda().float()
    std = torch.from_numpy(np.load(config.MOTION_STATS_STD)).cuda().float()
    skeleton = ref.SOMASkeleton30().cuda()
    starts = list(range(0,len(rows),args.batch_cases))[args.shard::args.shards]
    beginning = time.monotonic()
    newly_done = 0
    for start in starts:
        path = work / f'batch_{start:07d}.npz'
        if path.exists():
            continue
        batch = rows[start:start+args.batch_cases]
        n = len(batch)
        def load_array(row, field, digest):
            p = Path(source['data_dir']) / row[field]
            assert sha(p) == row[digest]
            return np.load(p, allow_pickle=False).astype(np.float32)
        # Benchmark files contain ungrounded source features. Match the
        # evaluator's raw_window() exactly, including its per-recording floor.
        gt = torch.from_numpy(np.stack([ground_features(load_array(r,'features_file','features_sha256'),r['floor_offset']) for r in batch])).cuda()
        neutral = torch.from_numpy(np.stack([load_array(r,'neutral_file','neutral_sha256') for r in batch])).cuda()
        neutral = neutral - neutral.mean(1,keepdim=True)
        assert gt.shape == (n,97,283)
        images=[]; image_hashes=[]
        for variant in VARIANTS:
            for row in batch:
                donor = row
                if donor is None:
                    images.append(None)
                    image_hashes.append('none')
                else:
                    assert quant.identity(donor['image_path']) == donor['image_source_identity']
                    image_path=Path(donor['image_file'])
                    if not image_path.exists():
                        from nymeria_camera_dataset import decode_window_pyav
                        frames=decode_window_pyav(donor['image_path'],donor['image_start'],1,20.0)
                        assert len(frames)==1
                        pixels=np.ascontiguousarray(frames[0].transpose(2,0,1))
                        image_path.parent.mkdir(parents=True,exist_ok=True)
                        tmp=image_path.with_suffix(f'.{os.getpid()}.tmp')
                        with tmp.open('wb') as handle:np.save(handle,pixels,allow_pickle=False)
                        os.replace(tmp,image_path)
                    digest=sha(image_path)
                    if donor['image_sha256'] is not None:assert digest==donor['image_sha256']
                    image_hashes.append(digest)
                    pixels=np.load(image_path,allow_pickle=False)
                    assert pixels.ndim==3 and pixels.shape[0]==3 and pixels.dtype==np.uint8
                    images.append(torch.from_numpy(pixels))
        captions=[r['text'] for r in batch]*len(VARIANTS)
        cond=model.prepare_conditions(captions,modes=['textimg2motion']*len(captions),reasoner_images=images)
        text_cond=model.prepare_conditions(captions,modes=['text2motion']*len(captions),reasoner_images=[None]*len(captions))
        null=model.prepare_conditions(['']*len(captions),modes=['text2motion']*len(captions),reasoner_images=[None]*len(captions))
        empty=model._edge.tokenize_text('')
        assert all(torch.equal(c['input_ids'].cpu(),empty.cpu()) for c in null)
        for i,caption in enumerate(captions):
            assert torch.equal(text_cond[i]['input_ids'].cpu(),model._edge.tokenize_text(caption).cpu())
        audit=np.array([[len(c['input_ids']),int(c.get('visual_pos_mask',torch.zeros(1)).sum())]
                        for c in cond]).reshape(len(VARIANTS),n,2)
        assert (audit[:,:,1]>0).all()
        assert all(int(c.get('visual_pos_mask',torch.zeros(1)).sum())==0 for c in text_cond+null)
        weights=torch.tensor(IMAGE_WEIGHTS,device='cuda').repeat_interleave(n)[:,None,None]
        noise=torch.stack([torch.randn((97,283),generator=torch.Generator().manual_seed(stable_seed('nymeria32:'+r['case_id']))) for r in batch]).cuda().repeat(len(VARIANTS),1,1)
        shape=neutral.repeat(len(VARIANTS),1,1)
        pad=torch.zeros((n*len(VARIANTS),97),device='cuda',dtype=torch.bool)
        def closure(context):
            return lambda x,sigma:model(reasoner_inputs=context,x_sigma=x,sigma=sigma,neutral_joints=shape,motion_pad_mask=pad)
        def guided(x,sigma):
            u=closure(null)(x,sigma).float()
            t=closure(text_cond)(x,sigma).float()
            both=closure(cond)(x,sigma).float()
            return combine_cfg(u,t,both,weights)
        # Guidance is applied once here; the native solver receives guidance=1.
        sampled=sample_x0_unipc(guided,T=97,motion_dim=283,steps=35,guidance=1,
                              batch=n*len(VARIANTS),device='cuda',initial_noise=noise,shift=3)
        assert torch.isfinite(sampled).all()
        raw=sampled.float()*std+mean
        joints=decode_joints(raw)
        lengths=torch.full((n,),97,device='cuda',dtype=torch.long)
        values={'index':np.arange(start,start+n),'case_id':np.array([r['case_id'] for r in batch]),'token_audit':audit.transpose(1,0,2),
                'image_sha256':np.array(image_hashes).reshape(len(VARIANTS),n).T,
                'text_embedding':embedder.embed_text([r['text'] for r in batch]).cpu().numpy()}
        for j,name in enumerate(('GT',)+VARIANTS):
            features=gt if name=='GT' else raw[(j-1)*n:j*n]
            posed=decode_joints(gt) if name=='GT' else joints[(j-1)*n:j*n]
            metrics=quant.measure(ref,embedder,skeleton,posed,features[...,279:283]>.5,lengths,neutral)
            if name=='GT':
                # The established reference decodes GT on CPU in the dataset.
                # Sub-micrometre GPU decode differences can cross representation
                # thresholds. Reproduce the reference path; do not relax checks.
                packed=collate_st([gt_dataset[row['index']] for row in batch])
                metrics['embedding']=embedder.embed_motion_features(
                    packed['features'],packed['mask'],packed['neutral_joints']).cpu().numpy()
                # Independent gate against the already-established full-test
                # GT reference, including normalization and floor conventions.
                for i,row in enumerate(batch):
                    refstart=row['index']//64*64
                    with np.load(Path(source['reference_dir'])/f'batch_{refstart:07d}.npz',allow_pickle=False) as saved:
                        offset=row['index']-refstart
                        assert saved['case_id'][offset]==row['case_id']
                        for key in ('embedding','foot_skate_from_height','foot_depth_mean_cm'):
                            assert np.allclose(metrics[key][i],saved[key][offset],atol=2e-5,rtol=2e-4),(row['case_id'],key)
            values.update({name+'__'+key:value for key,value in metrics.items()})
        # Retain predetermined examples, without writing tens of GB of redundant motions.
        if start==0:
            quant.atomic_npz(args.out/'qualitative_samples.npz',raw_features=raw.cpu().numpy().reshape(len(VARIANTS),n,97,283),GT=gt.cpu().numpy(),neutral=neutral.cpu().numpy(),case_id=values['case_id'])
        quant.atomic_npz(path,**values)
        newly_done+=n
        done=sum(min(args.batch_cases,len(rows)-s) for s in starts if (work/f'batch_{s:07d}.npz').exists())
        total=sum(min(args.batch_cases,len(rows)-s) for s in starts)
        elapsed=time.monotonic()-beginning
        status={'stage':'sampling','shard':args.shard,'completed':done,'total':total,'cases_per_second':newly_done/elapsed,
                'eta_hours':(total-done)*elapsed/newly_done/3600,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'time_unix':time.time()}
        atomic_json(work/'progress.json',status)
        print('[progress] '+json.dumps(status),flush=True)
        if args.max_batches and newly_done>=args.max_batches*args.batch_cases:
            return
    atomic_json(work/'complete.json',{'status':'complete','cases':sum(min(args.batch_cases,len(rows)-s) for s in starts),'time_unix':time.time()})


def summarize(args):
    base.VARIANTS=VARIANTS
    modalities.VARIANTS=VARIANTS
    modalities.summarize(args)
    report=json.loads((args.out/'summary.json').read_text())
    report['interpretation']=[
        'Text guidance fixed at2; image increment weights0,0.5,1,2,3,5. Formula u+2*(t-u)+w*(both-t).',
        'Weight0 is text-only CFG2; weight2 is joint CFG2. Native prompt layouts differ.',
        'The text-first decomposition assigns text-image interaction to the image increment; it does not identify independent causal contributions.',
        'All settings use identical full-test cases and paired initial noise. No MPJPE ranking, GT-pose conditioning or output grounding.',
        'R32 uncertainty is across three grouping seeds, not generation seeds. FID/diversity are in Nymeria TMR space; compare diversity to measured GT.',
        'Illustrative examples were chosen by caption/action variety and GT motion before the sweep; full-test metrics use every eligible case.'
    ]
    atomic_json(args.out/'summary.json',report)
    lines=['# Text CFG2 with varying image CFG','',report['interpretation'][0],'',
        '| Condition | R@1 % | R@2 % | R@3 % | FID | Diversity | Foot skate | Foot penetration cm |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,result in report['results'].items():
        ret=result['retrieval'];phy=result['physics_20fps']
        lines.append(f"| {name} | {ret['R32_R01_mean']:.3f} | {ret['R32_R02_mean']:.3f} | {ret['R32_R03_mean']:.3f} | {result['FID']:.6f} | {result['diversity']:.6f} | {phy['foot_skate_from_height_mean']:.6f} | {phy['foot_depth_mean_cm_mean']:.4f} |")
    (args.out/'RESULTS.md').write_text('\n'.join(lines+['']+report['interpretation'])+'\n')
    atomic_json(args.out/'final_summary_ready.json',{'status':'complete','cases':report['cases'],
        'summary_sha256':sha(args.out/'summary.json'),'skipped':0})
    print('\n'.join(lines),flush=True)


@torch.inference_mode()
def preview(args):
    import config
    import subprocess
    import sys
    from checkpoint import configure_attention_from_checkpoint,load_checkpoint
    from model import EdgePhase2MotionExpert
    from flow import sample_x0_unipc
    from decode_uniego_torch import decode_joints
    from diagnose_image_conditioning import physics_setup
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    cfg,_=read_inputs(args)
    source=json.loads((args.preview_source/'configuration.json').read_text())
    selection=json.loads((args.out/'selection.json').read_text())
    cases=selection['cases'];names=VARIANTS+('image_only',)
    assert source['checkpoint_sha256']==cfg['checkpoint_sha256']==sha(cfg['checkpoint'])
    assert configure_attention_from_checkpoint(cfg['checkpoint'])==cfg['checkpoint_contract']
    for key,path in [('motion_mean',config.MOTION_STATS_MEAN),('motion_std',config.MOTION_STATS_STD)]:
        assert sha(path)==cfg['checkpoint_contract'][key]
    destination=args.out/'preview';destination.mkdir(exist_ok=True)
    contract={'source_configuration_sha256':sha(args.preview_source/'configuration.json'),
        'selection_sha256':sha(args.out/'selection.json'),'cases':cases,'seeds':[0,1,2],
        'variants':names,'text_weight':2,'image_weights':IMAGE_WEIGHTS,'steps':35,'shift':3,
        'frames':97,'source_sha256':sha(__file__),'checkpoint_sha256':cfg['checkpoint_sha256'],
        'scope':'Eight action-diverse illustrative cases, selected before this sweep; not a generative benchmark.'}
    # Normalize tuples once so immutable resume comparisons use the JSON form.
    contract=json.loads(json.dumps(contract))
    if (destination/'configuration.json').exists():assert json.loads((destination/'configuration.json').read_text())==contract
    else:atomic_json(destination/'configuration.json',contract)
    model=EdgePhase2MotionExpert(device='cuda',dtype=torch.bfloat16)
    payload=load_checkpoint(cfg['checkpoint'],model=model);assert payload['step']==200000
    del payload
    model.eval().requires_grad_(False)
    mean=torch.from_numpy(np.load(config.MOTION_STATS_MEAN)).cuda().float()
    std=torch.from_numpy(np.load(config.MOTION_STATS_STD)).cuda().float()
    physics=physics_setup()
    weights=torch.tensor(IMAGE_WEIGHTS+(2.,),device='cuda')[:,None,None]
    for case in cases:
        slot=case['slot'];dest=destination/f'case_{slot:03d}';dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists():continue
        path=args.preview_source/'inputs'/f'case_{slot:03d}.npz'
        assert sha(path)==source['input_hashes'][path.name]
        with np.load(path,allow_pickle=False) as saved:data={k:np.array(saved[k]) for k in saved.files}
        image=torch.from_numpy(data['image'])
        captions=[case['caption']]*6+['']
        both=model.prepare_conditions(captions,modes=['textimg2motion']*7,reasoner_images=[image]*7)
        text=model.prepare_conditions(captions,modes=['text2motion']*7,reasoner_images=[None]*7)
        null=model.prepare_conditions(['']*7,modes=['text2motion']*7,reasoner_images=[None]*7)
        empty=model._edge.tokenize_text('')
        assert all(torch.equal(c['input_ids'].cpu(),empty.cpu()) for c in null)
        assert all(torch.equal(c['input_ids'].cpu(),model._edge.tokenize_text(caption).cpu()) for c,caption in zip(text,captions))
        audit={'both_image_tokens':[int(c.get('visual_pos_mask',torch.zeros(1)).sum()) for c in both],
            'text_image_tokens':[int(c.get('visual_pos_mask',torch.zeros(1)).sum()) for c in text],
            'null_image_tokens':[int(c.get('visual_pos_mask',torch.zeros(1)).sum()) for c in null],
            'caption_for_image_only':'','text_weight':2.,'image_weights':list(IMAGE_WEIGHTS),
            'same_initial_noise':True,'GT_motion_conditioning':False,'endpoint_checks':[]}
        assert all(v>0 for v in audit['both_image_tokens']) and audit['text_image_tokens']==audit['null_image_tokens']==[0]*7
        neutral=torch.from_numpy(data['neutral']).cuda().float()[None].expand(7,-1,-1)
        pad=torch.zeros((7,97),device='cuda',dtype=torch.bool)
        def predict(context,x,sigma):
            return model(reasoner_inputs=context,x_sigma=x,sigma=sigma,neutral_joints=neutral,motion_pad_mask=pad).float()
        gt_raw=torch.from_numpy(data['motion']).cuda().float()*std+mean
        gt_joints=decode_joints(gt_raw[None])
        gt_metrics=physics(gt_joints,gt_raw[None]);measured=[]
        for seed in contract['seeds']:
            calls=0
            def guided(x,sigma):
                nonlocal calls
                u=predict(null,x,sigma);t=predict(text,x,sigma);b=predict(both,x,sigma)
                output=combine_cfg(u,t,b,weights)
                # Image-only comparator has no caption in its conditional path.
                output=torch.cat([output[:-1],(u[-1:]+2*(b[-1:]-u[-1:]))],dim=0)
                if calls==0:
                    assert torch.equal(output[0],u[0]+2*(t[0]-u[0]))
                    assert torch.equal(output[3],u[3]+2*(b[3]-u[3]))
                    audit['endpoint_checks'].append({'seed':seed,'weight0_exact_text_cfg2':True,'weight2_exact_joint_cfg2':True})
                calls+=1
                return output
            noise=torch.randn((1,97,283),generator=torch.Generator().manual_seed(seed)).cuda().repeat(7,1,1)
            assert all(torch.equal(noise[0],noise[index]) for index in range(1,7))
            sampled=sample_x0_unipc(guided,T=97,motion_dim=283,steps=35,guidance=1,
                                   batch=7,device='cuda',initial_noise=noise,shift=3)
            assert torch.isfinite(sampled).all()
            raw=sampled.float()*std+mean;joints=decode_joints(raw)
            arrays={'gt_features':gt_raw.cpu().numpy(),'gt_joints':gt_joints[0].cpu().numpy()}
            for index,name in enumerate(names):
                arrays[name+'_features']=raw[index].cpu().numpy()
                arrays[name+'_joints']=joints[index].cpu().numpy()
                measured.append({'variant':name,'seed':seed,**physics(joints[index:index+1],raw[index:index+1])})
            quant.atomic_npz(dest/f'seed_{seed}.npz',**arrays)
        render_case(args,case,data['image'])
        atomic_json(dest/'complete.json',{'case':case,'audit':audit,'GT':gt_metrics,'physics':measured})
        count=len(list(destination.glob('case_*/complete.json')))
        atomic_json(destination/'progress.json',{'completed':count,'total':len(cases),'time_unix':time.time()})
        print(f'[preview] {count}/{len(cases)}; slot{slot}; six image weights, image-only, three seeds',flush=True)
        if args.publisher and slot in (6,16,14,30,19):
            subprocess.run([sys.executable,str(args.publisher)],check=True)
    atomic_json(destination/'completed.json',{'status':'complete','cases':len(cases),'seeds':contract['seeds'],
        'scope':'Illustrative action-variety set; full-test generative scores remain separate.'})


def render_case(args,case,pixels):
    from PIL import Image,ImageDraw
    from render_motion import render_single,head_rotation_from_features,write_mp4
    import textwrap
    import av
    slot=case['slot'];gallery=args.out/'gallery';gallery.mkdir(exist_ok=True)
    with np.load(args.out/'preview'/f'case_{slot:03d}'/'seed_0.npz',allow_pickle=False) as saved:
        data={k:np.array(saved[k]) for k in saved.files}
    labels=[('gt','GT','tab:blue'),('image_only','Image only; CFG2','tab:green'),
            ('text2_img0','Text2 / Image0 (text only)','tab:orange'),
            ('text2_img0p5','Text2 / Image0.5','tab:gray'),('text2_img1','Text2 / Image1','tab:cyan'),
            ('text2_img2','Text2 / Image2','tab:purple'),('text2_img3','Text2 / Image3','tab:red'),
            ('text2_img5','Text2 / Image5','tab:brown')]
    panels={name:render_single(data[name+'_joints'],caption=label,color=color,width=440,height=440,
        frame_stride=2,max_frames=97,head_rotation0=head_rotation_from_features(data[name+'_features'])) for name,label,color in labels}
    for kind,order,columns in [('modalities',['gt','image_only','text2_img0','text2_img2'],2),
                               ('sweep',[item[0] for item in labels],4)]:
        grid=np.concatenate([np.concatenate([panels[k] for k in order[i:i+columns]],axis=2)
                             for i in range(0,len(order),columns)],axis=1)
        width=grid.shape[2];height=220
        banner=Image.new('RGB',(width,height),(245,245,245));draw=ImageDraw.Draw(banner)
        banner.paste(Image.fromarray(pixels.transpose(1,2,0)).resize((160,160)),(width-175,28))
        draw.rectangle((width-178,25,width-12,191),outline='green',width=3)
        draw.text((14,12),f'Case {slot} | Same seed0 / UniPC35 / shift3 | Text weight2; image weights0,0.5,1,2,3,5',fill='black')
        draw.multiline_text((14,43),'\n'.join(textwrap.wrap(case['caption'],max(60,(width-215)//6))),fill='black',spacing=5)
        draw.text((14,195),'GT shown for context; no observed motion. Image-only receives no caption. Selection: action diversity.',fill='black')
        frames=np.concatenate([np.broadcast_to(np.asarray(banner),(len(grid),height,width,3)),grid],axis=1)
        path=gallery/f'case_{slot:03d}_{kind}.mp4';write_mp4(frames,str(path),fps=10)
        Image.fromarray(frames[0]).save(path.with_suffix('.png'))
        with av.open(str(path)) as container:assert sum(1 for _ in container.decode(video=0))==49
    (gallery/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Text2, image weight ablation</h1>'+''.join(
        f'<h2>{path.stem}</h2><video controls width="1100" src="{path.name}"></video>' for path in sorted(gallery.glob('*.mp4'))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','preview','worker','summarize'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--source-operation',type=Path)
    parser.add_argument('--preview-source',type=Path)
    parser.add_argument('--publisher',type=Path)
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=2)
    parser.add_argument('--batch-cases',type=int,default=4)
    parser.add_argument('--max-batches',type=int,default=0)
    parser.add_argument('--memory-fraction',type=float,default=.8)
    args=parser.parse_args()
    host=socket.gethostname().split('.')[0]
    if not os.environ.get('SLURM_JOB_ID') or 'login' in host or host in {'cs-gpu-01','node37'}:
        raise RuntimeError('Use an owned compute allocation')
    if not 0<args.memory_fraction<=1:raise ValueError('Invalid memory fraction')
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base.setup();base.VARIANTS=VARIANTS;modalities.VARIANTS=VARIANTS
    if args.stage in {'prepare','preview','summarize'}:
        lock=(args.out/(args.stage+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    globals()[args.stage](args)


if __name__=='__main__':main()
