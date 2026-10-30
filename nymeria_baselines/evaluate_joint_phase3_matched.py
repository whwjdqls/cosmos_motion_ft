"""Joint-trained Phase3 V2M, using the frozen matched71 guidance protocol.

No training changes. The historical caption and guidance interventions are
imported from the exact prior evaluation and fingerprinted in the output.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

import numpy as np
import torch

from motion_expert_phase3_edge.settings import atomic_json, digest_json, require_compute_node, sha256


def read(path):
    return json.loads(Path(path).read_text())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def progress(args, stage, **values):
    row = dict(stage=stage, shard=args.shard, time=time.time(), job=os.environ['SLURM_JOB_ID'],
               node=socket.gethostname(), **values)
    atomic_json(args.output / f'shard{args.shard}_status.json', row)
    print(json.dumps(row), flush=True)


def main(args):
    require_compute_node()
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / f'shard{args.shard}.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    progress(args, 'loading_checkpoint')
    previous = read(args.reference / 'configuration.json')
    caption_cfg = read(args.reference / 'video_and_text/configuration.json')
    cases = previous['cases']
    assert len(cases) == len({r['uuid'] for r in cases}) == 71
    assert digest_json(cases) == previous['cases_sha256']
    assert caption_cfg['cases'] == cases
    assert previous['steps'] == 35 and previous['shift'] == 3
    assert previous['fps'] == 20 and previous['frames'] == 97
    assert sha256(caption_cfg['caption_implementation']) == caption_cfg['caption_implementation_sha256']
    payload = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    assert payload['step'] == 200000
    for field in ('phase1', 'phase1_metadata_sha256', 'phase2', 'phase2_contract'):
        assert payload['sources'][field] == previous['sources'][field], field
    from motion_expert_phase3_edge.visualize import configure
    contract = configure(payload)
    from motion_expert_phase3_edge_joint.model import EdgePhase3
    from motion_expert_phase3_edge_joint.sample import sample
    from motion_expert_phase3_edge.quantitative import check_bundle
    from eval_motion_recon import decode_to_joints
    from kimodo.metrics.foot_skate import FootSkateFromHeight
    from kimodo.skeleton import SOMASkeleton30
    ref = module(args.reference / 'evaluate.py', 'matched71_previous_guidance')
    captions = module(caption_cfg['caption_implementation'], 'matched71_previous_captions')
    root = Path(__file__).resolve().parents[1]
    config = dict(schema='joint_phase3_matched71_v1', checkpoint=str(args.checkpoint),
        checkpoint_sha256=sha256(args.checkpoint), step=200000,
        training_contract=read(args.checkpoint.parent.parent / 'contract.json'),
        sources=payload['sources'], cases=cases, cases_sha256=digest_json(cases),
        reference=str(args.reference), reference_config_sha256=sha256(args.reference / 'configuration.json'),
        reference_caption_config_sha256=sha256(args.reference / 'video_and_text/configuration.json'),
        source_sha256=sha256(__file__), guidance_code_sha256=sha256(args.reference / 'evaluate.py'),
        caption_code_sha256=sha256(caption_cfg['caption_implementation']),
        model_code={name: sha256(root / 'motion_expert_phase3_edge_joint' / name)
                    for name in ('model.py', 'native.py', 'sample.py', 'settings.py')},
        arms=['video', 'caption'], scales=[1., 2.], steps=35, shift=3, fps=20, frames=97,
        body23=ref.BODY23, neutral_shape='identical supplied shape from matched71 bundles',
        mode='video2motion', head_inpainting=False,
        guidance='whole-bridge-disabled reference, caption retained in both branches',
        caveat='inference intervention; not a video-dropout-trained classifier-free null',
        cohort='71 fixed T97 windows, one per held-out recording; not all test windows')
    target = args.output / 'configuration.json'
    if target.exists() and read(target) != config:
        raise RuntimeError('Evaluation identity differs; refusing incompatible resume')
    atomic_json(target, config)
    progress(args, 'loading_model')
    model = EdgePhase3(payload['sources']['phase1'], payload['sources']['phase2'],
                       device='cuda', activation_checkpointing=False)
    assert model.provenance['phase1_metadata_sha256'] == payload['sources']['phase1_metadata_sha256']
    assert model.provenance['phase2_contract'] == contract
    model.bridges.load_state_dict(payload['bridges'], strict=True)
    model.requires_grad_(False).eval()
    mean = np.load(contract['motion_stats_mean']).astype(np.float32)
    std = np.load(contract['motion_stats_std']).astype(np.float32)
    skeleton = SOMASkeleton30()
    feet = [int(j) for j in skeleton.foot_joint_idx]
    skate = FootSkateFromHeight(skeleton=skeleton, fps=20.)
    previous_metrics = {r['id']: r for r in read(args.reference / 'per_case.json')}
    selected = cases[args.shard::args.shards]
    started = time.monotonic()
    for index, row in enumerate(selected):
        motion, neutral, video = check_bundle(row, contract)
        gt = decode_to_joints(motion * std + mean)
        gt_metrics = ref.measure(gt, gt, motion, motion, skate, feet)
        assert max(abs(gt_metrics[k] - previous_metrics[row['id']]['GT'][k]) for k in gt_metrics) < 1e-7
        tensors = dict(motion=torch.from_numpy(motion)[None].cuda(),
                       neutral=torch.from_numpy(neutral)[None].cuda(), video=torch.from_numpy(video)[None].cuda())
        for arm in config['arms']:
            result = dict(id=row['id'], uuid=row['uuid'], start=row['start'], seed=row['seed'],
                          caption=row['caption'] if arm == 'caption' else '', GT=gt_metrics)
            for scale in config['scales']:
                key = ref.name(scale)
                path = args.output / arm / key / (row['id'] + '.npz')
                path.parent.mkdir(parents=True, exist_ok=True)
                progress(args, 'sampling', case=row['id'], arm=arm, scale=scale, complete=index,
                         total=len(selected), elapsed=time.monotonic()-started)
                if path.exists():
                    with np.load(path, allow_pickle=False) as saved:
                        assert str(saved['signature']) == digest_json(config)
                        assert np.array_equal(saved['gt_motion'], motion)
                        assert np.array_equal(saved['neutral'], neutral)
                        assert int(saved['seed']) == row['seed'] and float(saved['video_scale']) == scale
                        generated = saved['generated'].copy()
                else:
                    context = captions.caption_conditioning(model, row['caption']) if arm == 'caption' else nullcontext()
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16), context as calls:
                        prediction = sample(ref.GuidedPrediction(model, scale), mode='video2motion',
                            caption=row['caption'] if arm == 'caption' else '', **tensors,
                            seed=int(row['seed']), steps=35, guidance=1.)
                    if arm == 'caption':
                        expected = 35 if scale == 1 else 70
                        assert calls == dict(motion_reasoner=expected, video_reasoner=expected), calls
                    generated = prediction[0].float().cpu().numpy()
                    del prediction
                    assert generated.shape == (97, 283) and np.isfinite(generated).all()
                    if index == 0 and arm == 'video' and scale == 1:
                        # GT motion must be only a shape template: a native repeat
                        # with a zero template must match the wrapped native path.
                        probe = {**tensors, 'motion': torch.zeros_like(tensors['motion'])}
                        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                            repeat = sample(model, mode='video2motion', caption='', **probe,
                                            seed=int(row['seed']), steps=35, guidance=1.)
                        difference = float(np.max(np.abs(repeat[0].float().cpu().numpy()-generated)))
                        if difference != 0:
                            raise RuntimeError(f'Native/no-GT-motion control failed: {difference}')
                        atomic_json(args.output / f'shard{args.shard}_acceptance.json',
                                    dict(native_and_zero_motion_template_bitwise_equal=True,
                                         max_abs_difference=difference, case=row['id']))
                        del repeat
                    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}.npz')
                    np.savez(temporary, generated=generated, gt_motion=motion, neutral=neutral,
                             seed=row['seed'], video_scale=scale, caption=result['caption'],
                             signature=digest_json(config))
                    os.replace(temporary, path)
                result[key] = ref.measure(decode_to_joints(generated*std+mean), gt, generated, motion, skate, feet)
            atomic_json(args.output / arm / 'per_case' / (row['id']+'.json'), result)
        del tensors
        progress(args, 'case_complete', complete=index+1, total=len(selected),
                 elapsed=time.monotonic()-started, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    atomic_json(args.output / f'shard{args.shard}_COMPLETE.json',
                dict(cases=len(selected), ids=[r['id'] for r in selected], configuration_sha256=sha256(target)))
    progress(args, 'complete', cases=len(selected))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'reference', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=2)
    parser.add_argument('--memory-fraction', type=float, default=.35)
    args = parser.parse_args()
    try:
        main(args)
    except BaseException as error:
        progress(args, 'failed', error=str(error), traceback=traceback.format_exc())
        raise
