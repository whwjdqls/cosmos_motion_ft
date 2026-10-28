"""Matched image-only, text-only and image+text generative evaluation.

All modes sample from pure noise with native UniPC35, shift3, CFG2 against the
same unconditional T2M reference. No reconstruction-error ranking is used.
The frozen checkpoint and the immutable full Nymeria TI2M cohort are unchanged.
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
import evaluate_image_conditioning as base
from diagnose_image_conditioning import atomic_json, sha

quant=base.quant
VARIANTS=('image_only','text_only','image_text')
base.VARIANTS=VARIANTS
read_inputs=base.read_inputs


def prepare(args):
    old=json.loads((args.source_operation/'configuration.json').read_text())
    assert sha(old['manifest'])==old['manifest_sha256']
    assert sha(args.source_operation/'cases.json')==old['cases_sha256']
    rows=json.loads((args.source_operation/'cases.json').read_text())
    assert len(rows)==old['cases']
    cfg={k:v for k,v in old.items() if k not in {'wrong_images','no_image','source_request','source_request_sha256'}}
    cfg.update(schema='edge_generation_modalities_v1',variants=list(VARIANTS),
               batch_cases=args.batch_cases,shards=args.shards,source_sha256=sha(__file__),
               source_operation=str(args.source_operation),
               source_configuration_sha256=sha(args.source_operation/'configuration.json'),
               conditioning={'image_only':'first image, EMPTY caption',
                             'text_only':'caption, NO image',
                             'image_text':'caption and the same first image'},
               guidance_reference='native T2M empty text, no image for ALL three modes',
               formula='x0_unconditional + 2*(x0_conditioned-x0_unconditional)',
               caveat='Native T2M and TI2M prompt layouts differ. Image-only retrieval uses withheld captions, never provided to the generator.',
               diversity={'pairs':1000,'seed':20261028,'embedding':'Nymeria TMR','same_pairs_for_all_modes':True})
    cfg['sampling']=dict(old['sampling'],TI2M_null='native T2M empty text, no image')
    if (args.out/'configuration.json').exists():
        assert json.loads((args.out/'configuration.json').read_text())==cfg
        return
    # Preserve byte-for-byte original case manifest, including selection hashes.
    (args.out/'cases.json').write_bytes((args.source_operation/'cases.json').read_bytes())
    atomic_json(args.out/'configuration.json',cfg)
    (args.out/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    atomic_json(args.out/'protocol_audit.json',{'status':'passed','cases':len(rows),
        'same_case_manifest_sha256':sha(args.out/'cases.json')==old['cases_sha256'],
        'same_checkpoint':cfg['checkpoint_sha256']==old['checkpoint_sha256'],
        'same_evaluator':cfg['evaluator_sha256']==old['evaluator_sha256'],
        'same_noise_seed_policy':cfg['sampling']['seed']==old['sampling']['seed'],
        'all_modes_generate_from_pure_noise':True,'GT_motion_conditioning':False})
    print('[prepared] '+json.dumps({'cases':len(rows),'variants':VARIANTS}),flush=True)


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
                donor = None if variant == 'text_only' else row
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
        captions=['']*n+[r['text'] for r in batch]*2
        modes=['textimg2motion']*n+['text2motion']*n+['textimg2motion']*n
        cond=model.prepare_conditions(captions,modes=modes,reasoner_images=images)
        null=model.prepare_conditions(['']*len(captions),modes=['text2motion']*len(captions),
                                      reasoner_images=[None]*len(captions))
        # No evaluation caption may reach image-only conditioning. Native TI2M
        # still supplies its structural prompt; text-only uses native T2M.
        assert captions[:n]==['']*n and captions[n:2*n]==captions[2*n:]
        empty=model._edge.tokenize_text('')
        for i in range(n):
            assert torch.equal(cond[n+i]['input_ids'].cpu(),model._edge.tokenize_text(batch[i]['text']).cpu())
        assert all(torch.equal(c['input_ids'].cpu(),empty.cpu()) for c in null)
        audit=np.array([[len(c['input_ids']),int(c.get('visual_pos_mask',torch.zeros(1)).sum())]
                        for c in cond]).reshape(len(VARIANTS),n,2)
        assert (audit[0,:,1]>0).all() and (audit[1,:,1]==0).all() and (audit[2,:,1]>0).all()
        assert all(int(c.get('visual_pos_mask',torch.zeros(1)).sum())==0 for c in null)
        noise=torch.stack([torch.randn((97,283),generator=torch.Generator().manual_seed(stable_seed('nymeria32:'+r['case_id']))) for r in batch]).cuda().repeat(len(VARIANTS),1,1)
        shape=neutral.repeat(len(VARIANTS),1,1)
        pad=torch.zeros((n*len(VARIANTS),97),device='cuda',dtype=torch.bool)
        def closure(context):
            return lambda x,sigma:model(reasoner_inputs=context,x_sigma=x,sigma=sigma,neutral_joints=shape,motion_pad_mask=pad)
        sampled=sample_x0_unipc(closure(cond),predict_null=closure(null),T=97,motion_dim=283,steps=35,guidance=2,batch=n*len(VARIANTS),device='cuda',initial_noise=noise,shift=3)
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
    # Reuse the established GT reference, evaluator statistics, candidate groups,
    # and physical measures. Only the three generation conditions differ.
    base.summarize(args)
    report=json.loads((args.out/'summary.json').read_text())
    with np.load(args.out/'all_measurements.npz',allow_pickle=False) as data:
        count=len(data['case_id'])
        order=np.random.default_rng(20261028).permutation(count)
        pairs=order[:2000].reshape(1000,2)
        quant.atomic_npz(args.out/'diversity_pairs.npz',indices=pairs,case_id=data['case_id'])
        for name in ('GT',)+VARIANTS:
            embedding=data[name+'__embedding'].astype(np.float64)
            distances=np.linalg.norm(embedding[pairs[:,0]]-embedding[pairs[:,1]],axis=-1)
            report['results'][name]['diversity']=float(distances.mean())
    report['interpretation']=[
        'All three modes generate from pure noise; no MPJPE/GT-reconstruction ranking.',
        'The same unconditional empty-text/no-image reference and CFG2 are used for all modes.',
        'Image-only receives NO caption. Its retrieval score against withheld captions measures image-implied semantics, not compliance with a supplied prompt.',
        'R32 SD is across grouping seeds, not generation-seed uncertainty.',
        'FID and diversity are in Nymeria TMR embedding space; diversity should be compared to measured GT, not maximized.',
        'Native T2M/TI2M token layouts differ. Physical contact metrics describe a separate outcome.'
    ]
    atomic_json(args.out/'summary.json',report)
    lines=['# Image-only, text-only, and image + text generation','',
           f"Matched full test: {report['cases']} cases, {report['configuration']['sequences']} recordings. UniPC35 / shift3 / common unconditional CFG2.",'',
           '| Condition | R@1 % | R@2 % | R@3 % | FID | Diversity | Foot skate | Foot penetration cm |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,result in report['results'].items():
        ret=result['retrieval'];phy=result['physics_20fps']
        lines.append(f"| {name} | {ret['R32_R01_mean']:.3f} | {ret['R32_R02_mean']:.3f} | {ret['R32_R03_mean']:.3f} | {result['FID']:.6f} | {result['diversity']:.6f} | {phy['foot_skate_from_height_mean']:.6f} | {phy['foot_depth_mean_cm_mean']:.4f} |")
    lines+=['']+report['interpretation']
    (args.out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    atomic_json(args.out/'final_summary_ready.json',{'status':'complete','cases':report['cases'],
        'summary_sha256':sha(args.out/'summary.json'),'skipped':0})
    print('\n'.join(lines),flush=True)


@torch.inference_mode()
def preview(args):
    import config
    from checkpoint import configure_attention_from_checkpoint,load_checkpoint
    from model import EdgePhase2MotionExpert
    from flow import sample_x0_unipc
    from decode_uniego_torch import decode_joints
    import subprocess
    import sys
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    source=json.loads((args.preview_source/'configuration.json').read_text())
    cases=json.loads((args.preview_source/'cases.json').read_text())[:6]
    cfg,_=read_inputs(args)
    assert source['checkpoint_sha256']==cfg['checkpoint_sha256']==sha(cfg['checkpoint'])
    assert configure_attention_from_checkpoint(cfg['checkpoint'])==cfg['checkpoint_contract']
    for key,path in [('motion_mean',config.MOTION_STATS_MEAN),('motion_std',config.MOTION_STATS_STD)]:
        assert sha(path)==cfg['checkpoint_contract'][key]
    destination=args.out/'preview';destination.mkdir(exist_ok=True)
    contract={'source_configuration_sha256':sha(args.preview_source/'configuration.json'),
              'source_cases_sha256':sha(args.preview_source/'cases.json'),
              'cases':cases,'seeds':[0,1,2],'variants':list(VARIANTS),
              'guidance':2,'steps':35,'shift':3,'frames':97,
              'source_sha256':sha(__file__),'checkpoint_sha256':cfg['checkpoint_sha256'],
              'scope':'Six predetermined original diagnostic cases for qualitative comparison; full-test quantitative results are computed separately.'}
    if (destination/'configuration.json').exists():assert json.loads((destination/'configuration.json').read_text())==contract
    else:atomic_json(destination/'configuration.json',contract)
    model=EdgePhase2MotionExpert(device='cuda',dtype=torch.bfloat16)
    payload=load_checkpoint(cfg['checkpoint'],model=model);assert payload['step']==200000
    del payload
    model.eval().requires_grad_(False)
    mean=torch.from_numpy(np.load(config.MOTION_STATS_MEAN)).cuda().float()
    std=torch.from_numpy(np.load(config.MOTION_STATS_STD)).cuda().float()
    for case in cases:
        slot=case['slot'];dest=destination/f'case_{slot:03d}';dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists():continue
        path=args.preview_source/'inputs'/f'case_{slot:03d}.npz'
        assert sha(path)==source['input_hashes'][path.name]
        with np.load(path,allow_pickle=False) as saved:data={k:np.array(saved[k]) for k in saved.files}
        image=torch.from_numpy(data['image'])
        captions=['',case['caption'],case['caption']]
        modes=['textimg2motion','text2motion','textimg2motion']
        cond=model.prepare_conditions(captions,modes=modes,reasoner_images=[image,None,image])
        null=model.prepare_conditions(['']*3,modes=['text2motion']*3,reasoner_images=[None]*3)
        empty=model._edge.tokenize_text('')
        assert all(torch.equal(c['input_ids'].cpu(),empty.cpu()) for c in null)
        audit={'captions_to_generator':captions,'modes':modes,
               'image_tokens':[int(c.get('visual_pos_mask',torch.zeros(1)).sum()) for c in cond],
               'null_image_tokens':[int(c.get('visual_pos_mask',torch.zeros(1)).sum()) for c in null],
               'same_noise_between_modes':True,'GT_motion_conditioning':False}
        assert audit['image_tokens'][0]>0 and audit['image_tokens'][1]==0 and audit['image_tokens'][2]>0
        assert audit['null_image_tokens']==[0,0,0]
        assert torch.equal(cond[1]['input_ids'].cpu(),model._edge.tokenize_text(case['caption']).cpu())
        neutral=torch.from_numpy(data['neutral']).cuda().float()[None].expand(3,-1,-1)
        pad=torch.zeros((3,97),device='cuda',dtype=torch.bool)
        def closure(context):
            return lambda x,sigma:model(reasoner_inputs=context,x_sigma=x,sigma=sigma,neutral_joints=neutral,motion_pad_mask=pad)
        gt_raw=torch.from_numpy(data['motion']).cuda().float()*std+mean
        gt_joints=decode_joints(gt_raw[None])[0].cpu().numpy()
        for seed in contract['seeds']:
            noise=torch.randn((1,97,283),generator=torch.Generator().manual_seed(seed)).cuda().repeat(3,1,1)
            assert torch.equal(noise[0],noise[1]) and torch.equal(noise[0],noise[2])
            sample=sample_x0_unipc(closure(cond),predict_null=closure(null),T=97,motion_dim=283,steps=35,
                                  guidance=2,batch=3,device='cuda',initial_noise=noise,shift=3)
            assert torch.isfinite(sample).all()
            raw=sample.float()*std+mean;joints=decode_joints(raw)
            arrays={'gt_features':gt_raw.cpu().numpy(),'gt_joints':gt_joints}
            for index,name in enumerate(VARIANTS):
                arrays[name+'_features']=raw[index].cpu().numpy()
                arrays[name+'_joints']=joints[index].cpu().numpy()
            quant.atomic_npz(dest/f'seed_{seed}.npz',**arrays)
        render_preview(args,case,data['image'])
        atomic_json(dest/'complete.json',{'case':case,'audit':audit,'seeds':contract['seeds']})
        atomic_json(destination/'progress.json',{'completed':len(list(destination.glob('case_*/complete.json'))),'total':6})
        print(f'[preview] {slot+1}/6 complete, three modalities and three seeds',flush=True)
        if args.publisher and slot in (0,1,5):
            subprocess.run([sys.executable,str(args.publisher)],check=True)


def render_preview(args,case,pixels):
    from PIL import Image,ImageDraw
    from render_motion import render_single,head_rotation_from_features,write_mp4
    import textwrap
    import av
    slot=case['slot'];gallery=args.out/'gallery';gallery.mkdir(exist_ok=True)
    with np.load(args.out/'preview'/f'case_{slot:03d}'/'seed_0.npz',allow_pickle=False) as saved:
        data={k:np.array(saved[k]) for k in saved.files}
    panels=[]
    for name,label,color in [('gt','GT','tab:blue'),('image_only','Image only (no caption)','tab:green'),
                              ('text_only','Text only (no image)','tab:orange'),('image_text','Image + text','tab:purple')]:
        panels.append(render_single(data[name+'_joints'],caption=label,color=color,width=480,height=480,
                                    frame_stride=2,max_frames=97,head_rotation0=head_rotation_from_features(data[name+'_features'])))
    grid=np.concatenate([np.concatenate(panels[:2],axis=2),np.concatenate(panels[2:],axis=2)],axis=1)
    banner=Image.new('RGB',(960,200),(245,245,245));draw=ImageDraw.Draw(banner)
    banner.paste(Image.fromarray(pixels.transpose(1,2,0)).resize((160,160)),(785,25))
    draw.rectangle((782,22,948,188),outline='green',width=3)
    draw.text((15,12),'Observed image: image-only / both. Caption below: text-only / both ONLY.',fill='black')
    draw.multiline_text((15,45),'\n'.join(textwrap.wrap(case['caption'],90)),fill='black',spacing=6)
    draw.text((15,175),'Same initial noise; UniPC35; CFG2; no observed motion. Seed 0.',fill='black')
    frames=np.concatenate([np.broadcast_to(np.asarray(banner),(len(grid),200,960,3)),grid],axis=1)
    path=gallery/f'case_{slot:03d}.mp4';write_mp4(frames,str(path),fps=10)
    Image.fromarray(frames[0]).save(path.with_suffix('.png'))
    with av.open(str(path)) as container:assert sum(1 for _ in container.decode(video=0))==49
    (gallery/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>GT / Image only / Text only / Image + text</h1>'+''.join(f'<h2>Case {index}</h2><video controls width="960" src="case_{index:03d}.mp4"></video>' for index in range(6)))


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
    parser.add_argument('--memory-fraction',type=float,default=.4)
    args=parser.parse_args()
    host=socket.gethostname().split('.')[0]
    if not os.environ.get('SLURM_JOB_ID') or 'login' in host or host in {'cs-gpu-01','node37'}:
        raise RuntimeError('Use an owned compute allocation')
    if not 0<args.memory_fraction<=1:raise ValueError('Invalid memory cap')
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base.setup()
    if args.stage in {'prepare','preview','summarize'}:
        lock=(args.out/(args.stage+'.lock')).open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    globals()[args.stage](args)


if __name__=='__main__':main()
