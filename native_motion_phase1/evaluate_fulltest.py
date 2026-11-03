"""Resumable, full held-out T97 evaluation of the native motion ablation.

All paths and weights are node-local. The checkpoint's native EMA network,
283-D representation, observed shape token and actual-sigma x0 conversion are
retained. Camera actions are unsupported by this architecture.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

TASKS = ('motion2video', 'video2motion', 'text2motion', 'textimg2motion')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def status(args, stage, **values):
    value = dict(stage=stage, time=time.time(), **values)
    save_json(args.out / 'status.json', value)
    print(json.dumps(value), flush=True)


def prepare(args):
    import config
    import nymeria_joint_dataset as nj
    from dataset import humanize_caption

    root = Path(os.environ['WEKA_ROOT']) / 'nymeriaplus_kimodo_proportional'
    contract = json.loads((args.checkpoint.parent.parent / 'motion_contract.json').read_text())
    assert contract['motion_objective'] == 'x0_reconstruction_shared_waver_v1'
    assert sha(config.MOTION_STATS_MEAN) == contract['mean_sha256']
    assert sha(config.MOTION_STATS_STD) == contract['std_sha256']
    dataset = nj.NymeriaJointDataset(
        manifest_path=str(root/'video/manifest_video.jsonl'),
        split_file=str(root/'train_test_split.json'), split='test', num_frames=97,
        aligned_num_frames=97, fps=20, task_weights={'video2motion': .5, 'motimg2video': .5},
        bones_text2motion_frac=0, cfg_dropout=0, prefer_latents=False, train=False,
        uniego_root=os.environ['NYMERIA_UNIEGO_ROOT'],
        floor_calibration_json=str(root/'metadata/floor_calibration.json'), seed=0)
    split = json.loads((root/'train_test_split.json').read_text())
    candidates = sorted(enumerate(dataset._index), key=lambda p:(p[1]['uuid'],p[1]['s'],str(p[1]['cap']),p[0]))
    arrays = args.out/'inputs'
    arrays.mkdir(exist_ok=True)
    rows, rejected = [], []
    current, features, neutral = None, None, None
    # Read each compressed recording once; the original dataset still owns all
    # floor/canonicalization/normalization and feature-eligibility transforms.
    original_loader = nj.load_motion_window
    try:
        for index, row in candidates:
            assert row['uuid'] in split['test'] and row['uuid'] not in split['train']
            if row['uni'] != current:
                with np.load(row['uni'], allow_pickle=False) as z:
                    features, neutral = z['features'].astype(np.float32), z['neutral_joints'].astype(np.float32)
                current = row['uni']
            nj.load_motion_window = lambda path,s,e: (features[s:e].copy(), neutral.copy())
            try:
                motion, shape, pad = dataset._load_motion(row['uni'],row['s'],row['off'],avail=97)
                assert not pad.any() and motion.shape == (97,283)
                assert np.isfinite(motion).all() and np.isfinite(shape).all()
            except (ValueError, AssertionError) as error:
                rejected.append(dict(index=index,uuid=row['uuid'],start=row['s'],reason=str(error)))
                continue
            latent = Path(nj.latent_path(row['uuid'],row['s'],os.environ['NYMERIA_LATENT_ROOT']))
            if not latent.is_file():
                physical = hashlib.sha256(json.dumps([row['uuid'],int(row['s'])]).encode()).hexdigest()[:24]
                latent = args.out/'latents'/(physical+'.npz')
            identity = json.dumps([row['uuid'],row['s'],row['cap'],row['off'],index],sort_keys=True)
            uid = f'{len(rows):06d}_' + hashlib.sha256(identity.encode()).hexdigest()[:16]
            dest = arrays/(uid+'.npz')
            np.savez(dest, motion=motion, neutral=shape)
            rows.append(dict(id=uid,index=index,uuid=row['uuid'],start=int(row['s']),
                caption=humanize_caption(row['cap']),floor_offset=row['off'],
                bundle=str(dest),bundle_sha256=sha(dest),latent=str(latent),
                rgb=row['vis'],seed=int.from_bytes(hashlib.sha256(('naeive:'+uid).encode()).digest()[:4],'little')%(2**31)))
            if len(rows)%1000==0:
                status(args,'preparing',eligible=len(rows),rejected=len(rejected),candidates=len(candidates))
    finally:
        nj.load_motion_window = original_loader
    assert set(r['uuid'] for r in rows) == set(split['test'])
    save_json(args.out/'cases.json', rows)
    save_json(args.out/'rejections.json', rejected)
    selection = []
    for uid in sorted(set(r['uuid'] for r in rows)):
        selection.append(next(i for i,r in enumerate(rows) if r['uuid']==uid))
    chosen = [selection[i] for i in np.linspace(0,len(selection)-1,6).astype(int)]
    configuration = dict(schema='native_motion_fulltest_v1', checkpoint=str(args.checkpoint),
        checkpoint_model_metadata_sha256=sha(args.checkpoint/'model/.metadata'),
        checkpoint_step=200000, weights='EMA', training_contract=contract,
        mean=config.MOTION_STATS_MEAN,std=config.MOTION_STATS_STD,
        mean_sha256=sha(config.MOTION_STATS_MEAN),std_sha256=sha(config.MOTION_STATS_STD),
        candidates=len(candidates),eligible_cases=len(rows),rejected=len(rejected),
        recordings=len(set(r['uuid'] for r in rows)),
        physical_windows=len(set((r['uuid'],r['start']) for r in rows)),
        windows_per_recording=dict(Counter(r['uuid'] for r in rows)),
        cases_sha256=sha(args.out/'cases.json'),
        manifest_sha256=sha(root/'video/manifest_video.jsonl'),
        split_sha256=sha(root/'train_test_split.json'),
        floor_sha256=sha(root/'metadata/floor_calibration.json'),
        tasks=list(TASKS),unsupported=['image_camera_to_video','video_to_camera'],
        sampling=dict(solver='native UniPC',steps=30,shift=10.,guidance=1.,
                      seed='deterministic per-case; same across tasks',frames=97,fps=20),
        text_guidance='CFG1 matches training previews; no guidance sweep or test-based selection',
        cohort='All eligible annotation entries in the full test aligned-T97 index; all71 held-out recordings, no fixed71 subsampling. Identical cohort for all four tasks.',
        captions='Humanized annotation for text tasks only; empty for video/motion tasks.',
        video_conditioning='First causal RGB/Wan latent only for M2V and TI2M.',
        information_guard='Unknown motion tokens are zero placeholders before pure-noise sampling; unknown future video latents zeroed for M2V. GT is not provided in unobserved inputs.',
        motion_metrics='Raw decoded SOMA30, no generated grounding/alignment except explicitly labeled PA-MPJPE; body23 additionally reported.',
        video_metrics='RGB frames1-96, observed RGB frame0 excluded; GT resized exactly like native inference.',
        qualitative_indices=chosen,source_sha256=sha(__file__))
    save_json(args.out/'configuration.json',configuration)
    status(args,'prepared',cases=len(rows),recordings=configuration['recordings'],
           physical_windows=configuration['physical_windows'],rejected=len(rejected))


def load_model(args):
    import torch
    from cosmos_framework.inference.common.init import init_script
    init_script()
    from native_phase_training.compute_access import require_compute_access
    require_compute_access()
    import native_motion_phase1.experiment
    from cosmos_framework.utils.generator.model_loader import load_model_from_checkpoint
    status(args,'loading_model')
    model,cfg = load_model_from_checkpoint(
        experiment_name='native_motion_phase1_edge',checkpoint_path=str(args.checkpoint),
        config_file='native_phase_training/inference_config.py',load_ema_to_reg=True,
        parallelism_config=dict(enable_inference_mode=True,data_parallel_shard_degree=1,
                                data_parallel_replicate_degree=1,context_parallel_shard_degree=1,
                                cfg_parallel_shard_degree=1),
        compile_config=dict(enabled=False,use_cuda_graphs=False),
        experiment_opts=['model.config.rectified_flow_inference_config.scheduler_type=unipc'])
    model.eval().requires_grad_(False)
    # Direct model loading does not install the optional inference SeaCache.
    # Avoid cross-sample conditioning reuse throughout the scientific evaluation.
    model._can_reuse_inference_text_kv=lambda *a,**k:False
    return model


def sample(args):
    import torch
    from native_motion_phase1.data import inference_batch
    cfg=json.loads((args.out/'configuration.json').read_text())
    assert cfg['source_sha256']==sha(__file__)
    assert cfg['cases_sha256']==sha(args.out/'cases.json')
    assert cfg['checkpoint_model_metadata_sha256']==sha(args.checkpoint/'model/.metadata')
    rows=json.loads((args.out/'cases.json').read_text())
    model=load_model(args)
    started=time.monotonic()
    n=0
    # First finish six fixed qualitative cases on every task, then the full cohort.
    order=cfg['qualitative_indices']+[i for i in range(len(rows)) if i not in cfg['qualitative_indices']]
    if args.limit:
        order=order[:args.limit]
    for index in order:
        row=rows[index]
        assert sha(row['bundle'])==row['bundle_sha256']
        with np.load(row['bundle'],allow_pickle=False) as z:
            motion,neutral=z['motion'].copy(),z['neutral'].copy()
        if not Path(row['latent']).exists():
            from nymeria_camera_dataset import decode_window_pyav
            from native_phase_training.evaluate_inverse_forward import _resize_gt_like_native
            rgb=_resize_gt_like_native(decode_window_pyav(row['rgb'],row['start'],97,20.),256,256)
            assert rgb.shape==(97,256,256,3)
            with torch.no_grad():
                latent=model.encode(torch.from_numpy(rgb).permute(3,0,1,2)[None].to('cuda',torch.float32)/127.5-1)[0].cpu().numpy()
            p=Path(row['latent']);p.parent.mkdir(exist_ok=True)
            temp=p.with_suffix('.tmp')
            with temp.open('wb') as f:np.savez(f,latents=latent)
            temp.replace(p)
        with np.load(row['latent'],allow_pickle=False) as z:
            latent=z['latents'].copy()
        assert latent.shape==(48,25,16,16) and np.isfinite(latent).all()
        for task in TASKS:
            dest=args.out/'predictions'/task/(row['id']+'.npz')
            if dest.exists():
                with np.load(dest,allow_pickle=False) as z:
                    assert str(z['configuration_sha256'])==sha(args.out/'configuration.json')
                    expected=(97,256,256,3) if task=='motion2video' else (97,283)
                    assert z['generated'].shape==expected and np.isfinite(z['generated']).all()
                continue
            visible_motion=motion if task=='motion2video' else np.zeros_like(motion)
            visible_video=latent.copy()
            if task=='motion2video':visible_video[:,1:]=0
            batch=inference_batch(visible_motion,neutral,visible_video,task,row['caption'])
            before=time.monotonic()
            with torch.no_grad():
                result=model.generate_samples_from_batch(batch,net=model.net,seed=[row['seed']],
                    num_steps=30,shift=10.,guidance=1.)
                if task=='motion2video':
                    observed_error=float((result['vision'][0][0,:,:1].float()-torch.from_numpy(latent[:,:1]).to('cuda').float()).abs().max())
                    assert observed_error<1e-4
                    pixels=model.decode(result['vision'][0].float())
                    assert pixels.shape==(1,3,97,256,256) and torch.isfinite(pixels).all()
                    generated=((pixels[0].float().clamp(-1,1)+1)*127.5).round().byte().permute(1,2,3,0).cpu().numpy()
                else:
                    action=result['action'][0].reshape(98,-1)[:,:283].float().cpu().numpy()
                    observed_error=float(np.max(np.abs(action[0]-batch['action'][0][0,:283].numpy())))
                    assert observed_error<1e-4
                    generated=action[1:]
                    assert generated.shape==(97,283) and np.isfinite(generated).all()
            dest.parent.mkdir(parents=True,exist_ok=True)
            temp=dest.with_suffix('.tmp')
            with temp.open('wb') as f:
                np.savez(f,generated=generated,configuration_sha256=sha(args.out/'configuration.json'),
                         observed_error=observed_error,seconds=time.monotonic()-before,seed=row['seed'])
            temp.replace(dest)
            n+=1
            status(args,'sampling',case=index,id=row['id'],task=task,new_predictions=n,
                   total_predictions=len(rows)*4,elapsed_seconds=time.monotonic()-started,
                   seconds=time.monotonic()-before,peak_gib=torch.cuda.max_memory_allocated()/2**30)
            del result
    status(args,'smoke_complete' if args.limit else 'inference_complete',cases=len(order),new_predictions=n)
    if not args.limit:
        save_json(args.out/'INFERENCE_COMPLETE.json',dict(cases=len(rows),tasks=list(TASKS),configuration_sha256=sha(args.out/'configuration.json')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','sample'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    assert args.out.resolve().is_relative_to('/mnt/local')
    assert args.checkpoint.resolve().is_relative_to('/mnt/local')
    (prepare if args.action=='prepare' else sample)(args)


if __name__=='__main__':main()
