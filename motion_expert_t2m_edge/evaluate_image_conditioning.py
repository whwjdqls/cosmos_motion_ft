"""Paired full-test generative image ablation; run only in an owned GPU job.

Reuses the immutable Nymeria TI2M benchmark and validation-selected evaluator.
Correct/wrong images share token layout; native T2M is a separate layout control.
No training, checkpoint, or existing evaluation artifacts are modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(1, str(REPO / 'shape_aware_TMR'))
sys.path.insert(2, str(REPO / 'motion_expert'))
from diagnose_image_conditioning import atomic_json, sha
import phase2_quantitative as quant

VARIANTS = ('correct', 'wrong1', 'wrong2', 'no_image')


def setup():
    from shape_tmr_eval_common import add_bundle_python_paths
    add_bundle_python_paths(os.environ['SHAPE_TMR_BUNDLE'])
    # The bundle pins Kimodo, while Nymeria's evaluator adapters and checkpoint
    # loaders must come from this repository, including its C45 compatibility fix.
    for path in reversed([REPO/'motion_expert_t2m_edge', REPO/'shape_aware_TMR',
                          REPO/'motion_expert_joint_attention', REPO/'motion_expert']):
        if str(path) in sys.path:
            sys.path.remove(str(path))
        sys.path.insert(0,str(path))


def read_inputs(args):
    cfg = json.loads((args.out / 'configuration.json').read_text())
    rows = json.loads((args.out / 'cases.json').read_text())
    assert sha(cfg['manifest']) == cfg['manifest_sha256']
    assert sha(args.out / 'cases.json') == cfg['cases_sha256']
    return cfg, rows


def prepare(args):
    from checkpoint import configure_attention_from_checkpoint
    import config
    source = json.loads(args.source_request.read_text())
    diagnostic = json.loads(args.diagnostic_config.read_text())
    contract = configure_attention_from_checkpoint(diagnostic['checkpoint'])
    assert contract == diagnostic['checkpoint_contract']
    assert sha(diagnostic['checkpoint']) == diagnostic['checkpoint_sha256']
    for key, path in [('motion_mean', config.MOTION_STATS_MEAN), ('motion_std', config.MOTION_STATS_STD)]:
        assert sha(path) == contract[key], key
    manifest = Path(source['data_dir']) / 'test.jsonl'
    assert sha(manifest) == source['expected_manifest_sha256']
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(rows) == source['expected_cases'] and len({r['case_id'] for r in rows}) == len(rows)
    order = sorted(range(len(rows)), key=lambda i: (rows[i]['uuid'], rows[i]['case_id']))
    image_sources = {}
    for pos, index in enumerate(order):
        row = rows[index]
        assert row['num_frames'] == 97
        row['index'] = index
        for label, offset in [('wrong1', len(rows)//3), ('wrong2', 2*len(rows)//3)]:
            donor = order[(pos + offset) % len(rows)]
            assert rows[donor]['uuid'] != row['uuid']
            row[label + '_index'] = donor
        source_path = row['image_path']
        if source_path not in image_sources:
            image_sources[source_path] = quant.identity(source_path)
        key = hashlib.sha256(json.dumps([image_sources[source_path], row['image_start'], 20.0], sort_keys=True).encode()).hexdigest()
        image_file = Path(source['image_cache']) / (key + '.npy')
        row['image_source_identity'] = image_sources[source_path]
        if not image_file.is_file():
            # Storage moves invalidate the old identity-keyed cache. Re-decode
            # the exact source frame into this operation, never guess an old key.
            image_file = args.out / 'lossless_first_frames' / (key + '.npy')
        row['image_file'] = str(image_file)
        row['image_sha256'] = sha(image_file) if image_file.exists() else None
        for field, digest in [('features_file', 'features_sha256'), ('neutral_file', 'neutral_sha256')]:
            assert sha(Path(source['data_dir']) / row[field]) == row[digest]
        if index % 1000 == 0:
            print(f'[prepare] verified {index}/{len(rows)}', flush=True)
    for variant in ('wrong1', 'wrong2'):
        assert sorted(r[variant + '_index'] for r in rows) == list(range(len(rows)))
    cfg = {
        'schema': 'edge_paired_generative_image_v1', 'source_request': str(args.source_request),
        'source_request_sha256': sha(args.source_request), 'reference_request': source,
        'checkpoint': diagnostic['checkpoint'], 'checkpoint_sha256': diagnostic['checkpoint_sha256'],
        'checkpoint_contract': contract, 'generator_stats': diagnostic['statistics'],
        'manifest': str(manifest), 'manifest_sha256': sha(manifest),
        'evaluator_sha256': sha(source['evaluator_checkpoint']),
        'variants': list(VARIANTS), 'cases': len(rows), 'sequences': len({r['uuid'] for r in rows}),
        'physical_windows': len({(r['uuid'],r['start_frame']) for r in rows}),
        'windows_per_sequence': dict(Counter(r['uuid'] for r in rows)),
        'batch_cases': args.batch_cases, 'shards': args.shards,
        'sampling': {'solver': 'UniPC', 'steps': 35, 'shift': 3, 'CFG': 2,
                     'seed': "stable_seed('nymeria32:' + case_id)", 'frames': 97, 'fps': 20,
                     'GT_motion_conditioning': False, 'initial_noise_shared_between_variants': True,
                     'TI2M_null': 'empty text, same variant image'},
        'retrieval': {'candidates': 32, 'seeds': [0,1,2], 'positive': 'paired only', 'drop_last': True},
        'wrong_images': 'Two fixed cross-recording permutations preserving the full image marginal distribution',
        'no_image': 'Native T2M prompt; layout differs from TI2M. Same caption, shape, length, noise.',
        'physics': 'Raw decoded coordinates, no prediction grounding or alignment',
        'source_sha256': sha(__file__),
    }
    atomic_json(args.out / 'cases.json', rows)
    cfg['cases_sha256'] = sha(args.out / 'cases.json')
    atomic_json(args.out / 'configuration.json', cfg)
    print('[prepared] ' + json.dumps({k:cfg[k] for k in ('cases','sequences','physical_windows')}), flush=True)


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
                donor = row if variant == 'correct' else (rows[row[variant+'_index']] if variant != 'no_image' else None)
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
        modes=['textimg2motion']*(n*3)+['text2motion']*n
        cond=model.prepare_conditions(captions,modes=modes,reasoner_images=images)
        null=model.prepare_conditions(['']*len(captions),modes=modes,reasoner_images=images)
        for group in (cond,null):
            for i in range(n):
                for j in (n+i,2*n+i):
                    for key in ('input_ids','position_ids','visual_pos_mask'):
                        assert torch.equal(group[i][key],group[j][key]),(start,key)
        audit = np.array([[len(c['input_ids']),int(c.get('visual_pos_mask',torch.zeros(1)).sum())] for c in cond]).reshape(4,n,2)
        noise=torch.stack([torch.randn((97,283),generator=torch.Generator().manual_seed(stable_seed('nymeria32:'+r['case_id']))) for r in batch]).cuda().repeat(4,1,1)
        shape=neutral.repeat(4,1,1)
        pad=torch.zeros((n*4,97),device='cuda',dtype=torch.bool)
        def closure(context):
            return lambda x,sigma:model(reasoner_inputs=context,x_sigma=x,sigma=sigma,neutral_joints=shape,motion_pad_mask=pad)
        sampled=sample_x0_unipc(closure(cond),predict_null=closure(null),T=97,motion_dim=283,steps=35,guidance=2,batch=n*4,device='cuda',initial_noise=noise,shift=3)
        assert torch.isfinite(sampled).all()
        raw=sampled.float()*std+mean
        joints=decode_joints(raw)
        lengths=torch.full((n,),97,device='cuda',dtype=torch.long)
        values={'index':np.arange(start,start+n),'case_id':np.array([r['case_id'] for r in batch]),'token_audit':audit.transpose(1,0,2),
                'image_sha256':np.array(image_hashes).reshape(4,n).T,
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
            quant.atomic_npz(args.out/'qualitative_samples.npz',raw_features=raw.cpu().numpy().reshape(4,n,97,283),GT=gt.cpu().numpy(),neutral=neutral.cpu().numpy(),case_id=values['case_id'])
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
    from retrieval32 import paired_retrieval32
    from kimodo.metrics.tmr import calculate_activation_statistics, calculate_frechet_distance
    cfg,rows=read_inputs(args)
    arrays={}
    for start in range(0,len(rows),cfg['batch_cases']):
        shard=(start//cfg['batch_cases'])%cfg['shards']
        with np.load(args.out/f'shard_{shard}'/f'batch_{start:07d}.npz',allow_pickle=False) as data:
            for key in data.files:
                arrays.setdefault(key,[]).append(np.array(data[key]))
    arrays={key:np.concatenate(value) for key,value in arrays.items()}
    assert arrays['index'].tolist()==list(range(len(rows)))
    assert arrays['case_id'].tolist()==[r['case_id'] for r in rows]
    assert all(np.isfinite(v).all() for v in arrays.values() if v.dtype.kind in 'fc')
    # Use one immutable measured GT reference for the entire comparison,
    # including batches saved before the CPU-decoding guard correction.
    source=cfg['reference_request']
    reference=Path(source['reference_dir'])
    reference_signature=json.loads((reference/'signature.json').read_text())
    assert reference_signature['manifest_sha256']==cfg['manifest_sha256']
    assert reference_signature['evaluator_sha256']==cfg['evaluator_sha256']
    gt_parts=[]
    for start in range(0,len(rows),64):
        with np.load(reference/f'batch_{start:07d}.npz',allow_pickle=False) as data:
            assert data['case_id'].tolist()==[r['case_id'] for r in rows[start:start+64]]
            gt_parts.append({key:np.array(data[key]) for key in data.files if key not in {'case_id','index'}})
    for key in gt_parts[0]:
        arrays['GT__'+key]=np.concatenate([part[key] for part in gt_parts])
    atomic_json(args.out/'GT_reference_provenance.json',{
        'reference':str(reference),'signature':reference_signature,
        'note':'Canonical measured GT reused for all conditions; completed generation batches unchanged. CPU dataset decoder matches the historical evaluator.'})
    text=arrays['text_embedding']
    gt_mu,gt_cov=calculate_activation_statistics(arrays['GT__embedding'])
    results={}
    per_case=[]
    for variant in ('GT',)+VARIANTS:
        measured={k.split('__',1)[1]:v for k,v in arrays.items() if k.startswith(variant+'__')}
        mu,cov=calculate_activation_statistics(measured['embedding'])
        fid=0.0 if variant=='GT' else float(calculate_frechet_distance(mu,cov,gt_mu,gt_cov))
        results[variant]={'FID':fid,'retrieval':paired_retrieval32(measured['embedding'],text),
                         'matching_distance':float(np.linalg.norm(text-measured['embedding'],axis=-1).mean()),
                         'physics_20fps':quant.physical_summary(measured)}
        for i,row in enumerate(rows):
            per_case.append({'case_id':row['case_id'],'uuid':row['uuid'],'variant':variant,
                'matching_distance':float(np.linalg.norm(text[i]-measured['embedding'][i])),
                **{k:float(v[i]) for k,v in measured.items() if k!='embedding'}})
    # Save candidate groups and ranks so every paired retrieval comparison is auditable.
    for seed in (0,1,2):
        order=np.random.default_rng(seed).permutation(len(rows))
        used=len(rows)//32*32
        groups=order[:used].reshape(-1,32)
        saved={'groups':groups,'dropped_indices':order[used:],'case_id':arrays['case_id']}
        for variant in ('GT',)+VARIANTS:
            t=text[groups].astype(np.float64);m=arrays[variant+'__embedding'][groups].astype(np.float64)
            d=np.maximum((t*t).sum(-1)[:,:,None]+(m*m).sum(-1)[:,None,:]-2*np.einsum('gqd,gkd->gqk',t,m),0)
            ranks=np.argmax(np.argsort(d,axis=-1,kind='stable')==np.arange(32)[None,:,None],axis=-1)+1
            saved[variant+'_paired_rank']=ranks
        quant.atomic_npz(args.out/f'retrieval_groups_seed{seed}.npz',**saved)
    quant.atomic_npz(args.out/'all_measurements.npz',**arrays)
    with (args.out/'per_case.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(per_case[0]));writer.writeheader();writer.writerows(per_case)
    result={'status':'complete','configuration':cfg,'results':results,'cases':len(rows),'skipped':0,
            'interpretation':['Full-sampler generative test, not single-denoiser GT reconstruction.',
                              'R32 SD is across grouping seeds, not generation-seed uncertainty.',
                              'FID belongs to the Nymeria evaluator space, not BONES C45.',
                              'No-image also changes the prompt layout. Foot contact is a separate outcome.']}
    atomic_json(args.out/'summary.json',result)
    lines=['# Full-test generative image-conditioning evaluation','',f"{len(rows)} windows, {cfg['sequences']} held-out recordings; UniPC35, shift3, CFG2.",'',
           '| Condition | R@1 % | R@2 % | R@3 % | FID | Match distance | Foot skate | Foot penetration cm |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in results.items():
        p=r['physics_20fps'];ret=r['retrieval']
        lines.append(f"| {name} | {ret['R32_R01_mean']:.3f} | {ret['R32_R02_mean']:.3f} | {ret['R32_R03_mean']:.3f} | {r['FID']:.6f} | {r['matching_distance']:.6f} | {p['foot_skate_from_height_mean']:.6f} | {p['foot_depth_mean_cm_mean']:.4f} |")
    lines+=['']+result['interpretation']
    (args.out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','worker','summarize'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source-request',type=Path)
    p.add_argument('--diagnostic-config',type=Path)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=2)
    p.add_argument('--batch-cases',type=int,default=8)
    p.add_argument('--max-batches',type=int,default=0)
    args=p.parse_args()
    host=socket.gethostname().split('.')[0]
    if 'login' in host or host in {'cs-gpu-01','node37'} or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use an owned compute allocation')
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    setup()
    globals()[args.stage](args)


if __name__=='__main__':
    main()
