"""Inference-only 3-checkpoint x 4-conditioning matched reconstruction study."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import html
import json
import os
from contextlib import contextmanager
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

from motion_expert_phase3_edge.settings import atomic_json, digest_json, require_compute_node, sha256
from motion_expert_phase3_edge.visualize import configure


RUNS = {
    'baseline': 'edge_p3_p1ema200k_crosscaption200k_bridge7_T97_gbs32_wv10_yonsei_20260924',
    'joint': 'edge_p3_p1ema200k_crosscaption200k_bridge7_joint3_equal_T97_gbs32_wv10_yonsei_20261002',
    'camcond': 'edge_p3_p1ema200k_crosscaption200k_bridge7_camcond3_mix_T97_gbs32_wv10_yonsei_20261004',
}
CONDITIONS = {
    'trajectory': dict(mode='camera2motion', video=False, camera=True, text=False),
    'video_text': dict(mode='video2motion', video=True, camera=False, text=True),
    'video_trajectory': dict(mode='cameravideo2motion', video=True, camera=True, text=False),
    'video_trajectory_text': dict(mode='cameravideo2motion', video=True, camera=True, text=True),
}
LABELS = {'baseline': 'Original', 'joint': 'Joint-generation', 'camcond': 'Camera-mix'}
COLUMNS = [('MPJPE_cm', 'mpjpe_m', 100), ('PA_MPJPE_cm', 'pa_mpjpe_m', 100),
           ('Root_ADE_cm', 'root_err_m', 100), ('Root_FDE_cm', 'root_fde_m', 100),
           ('Foot_skate_cm_s', 'foot_skate_cm_s', 1), ('Foot_penetration_cm', 'foot_depth_mean_cm', 1),
           ('Airborne_pct', 'all_feet_above5cm_frame_fraction', 100), ('Accel_error_m_s2', 'accel_err_m_s2', 1)]


def read(path):
    return json.loads(Path(path).read_text())


def progress(args, stage, **fields):
    value = dict(stage=stage, time=time.time(), job=os.environ['SLURM_JOB_ID'], **fields)
    atomic_json(args.output / 'progress.json', value)
    print(json.dumps(value), flush=True)


@contextmanager
def explicit_text():
    """Preserve explicitly supplied captions in both paths, in this process only."""
    from motion_expert_phase3_edge_camcond import model as cm, native as cn
    original_motion, original_native = cm.effective_caption, cn.effective_caption
    calls = dict(motion_reasoner=0, generator_reasoner=0)

    def motion_caption(mode, caption):
        calls['motion_reasoner'] += 1
        return caption

    def native_caption(mode, caption):
        calls['generator_reasoner'] += 1
        return caption

    cm.effective_caption, cn.effective_caption = motion_caption, native_caption
    try:
        yield calls
    finally:
        cm.effective_caption, cn.effective_caption = original_motion, original_native


def load_camera(row, camera_root):
    from cosmos_framework.data.generator.action.utils.pose_utils import pose_abs_to_rel
    path = camera_root / (row['uuid'] + '.npz')
    start = int(row['start'])
    with np.load(path, allow_pickle=False) as data:
        pos = data['cam_world_pos_upright'][start:start+97].astype(np.float32)
        rot = data['cam_world_rot_upright'][start:start+97].astype(np.float32)
    assert pos.shape == (97, 3) and rot.shape == (97, 3, 3)
    poses = np.tile(np.eye(4), (97, 1, 1))
    poses[:, :3, :3], poses[:, :3, 3] = rot, pos
    actions = pose_abs_to_rel(poses, rotation_format='rot6d', pose_convention='backward_framewise')
    assert actions.shape == (96, 9) and np.isfinite(actions).all()
    padded = np.zeros((96, 64), dtype=np.float32)
    padded[:, :9] = actions
    return padded, dict(path=str(path), sha256=sha256(path),
                        action_sha256=hashlib.sha256(padded.tobytes()).hexdigest())


def tensors(motion, neutral, video, camera):
    return {k: torch.from_numpy(v)[None].to('cuda', torch.float32)
            for k, v in dict(motion=motion, neutral=neutral, video=video, camera_action=camera).items()}


@torch.no_grad()
def preflight(model, values, row, run):
    from motion_expert_phase3_edge.model import EdgePhase3 as Base
    from motion_expert_phase3_edge_joint.model import EdgePhase3 as Joint
    from motion_expert_phase3_edge_camcond.model import EdgePhase3 as Camera
    native = {'baseline': Base, 'joint': Joint, 'camcond': Camera}[run]
    generator = torch.Generator(device='cuda').manual_seed(int(row['seed']))
    noise = torch.randn(values['motion'].shape, generator=generator, device='cuda')
    result = dict(native_v2m_parity=[], modalities={})
    for sigma_value in (1.0, 0.5, 0.01):
        sigma = torch.tensor([sigma_value], device='cuda')
        inputs = dict(motion=noise, neutral=values['neutral'], video=values['video'],
                      motion_sigma=sigma, video_sigma=sigma*0, captions=[''], modes=['video2motion'])
        expected = native.forward(model, **inputs)['motion'].float()
        with explicit_text() as calls:
            actual = model(**inputs)['motion'].float()
        assert calls == dict(motion_reasoner=1, generator_reasoner=1)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        result['native_v2m_parity'].append(dict(sigma=sigma_value, bit_exact=torch.equal(actual, expected),
                                               max_abs=float((actual-expected).abs().max())))
    sigma = torch.ones(1, device='cuda')
    for name, spec in CONDITIONS.items():
        inputs = dict(motion=noise, neutral=values['neutral'], video=values['video'],
                      camera_action=values['camera_action'] if spec['camera'] else None,
                      motion_sigma=sigma, video_sigma=sigma*0,
                      captions=[row['caption'] if spec['text'] else ''], modes=[spec['mode']])
        with explicit_text() as calls:
            out = model(**inputs)
        assert calls == dict(motion_reasoner=1, generator_reasoner=1)
        assert out['vision_sample_indices'] == ([0] if spec['video'] else [])
        assert out['camera_sample_indices'] == ([0] if spec['camera'] else [])
        assert torch.isfinite(out['motion']).all()
        result['modalities'][name] = dict(**spec, caption_calls=calls,
                                         vision_samples=out['vision_sample_indices'],
                                         camera_samples=out['camera_sample_indices'])
        if name == 'trajectory':
            with explicit_text():
                changed = model(**{**inputs, 'video': torch.zeros_like(values['video'])})['motion']
            torch.testing.assert_close(changed, out['motion'], atol=0, rtol=0)
            result['trajectory_ignores_video_bit_exact'] = True
    return result


@torch.no_grad()
def sample(model, values, row, spec):
    from cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler
    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1., use_dynamic_shifting=False)
    scheduler.set_timesteps(35, device='cuda', shift=3.)
    generator = torch.Generator(device='cuda').manual_seed(int(row['seed']))
    state = torch.randn(values['motion'].shape, device='cuda', dtype=torch.float32, generator=generator)
    with explicit_text() as calls:
        for index, timestep in enumerate(scheduler.timesteps):
            sigma = timestep.float().reshape(1)/1000.
            prediction = model(motion=state, neutral=values['neutral'], video=values['video'],
                               camera_action=values['camera_action'] if spec['camera'] else None,
                               motion_sigma=sigma, video_sigma=sigma*0,
                               captions=[row['caption'] if spec['text'] else ''], modes=[spec['mode']])['motion'].float()
            actual_sigma = scheduler.sigmas[index].to(device='cuda', dtype=torch.float32).clamp_min(1e-6)
            velocity = (state-prediction)/actual_sigma
            state = scheduler.step(velocity, timestep, state, return_dict=False, generator=generator)[0].float()
    assert calls == dict(motion_reasoner=35, generator_reasoner=35)
    assert torch.isfinite(state).all()
    return state[0].cpu().numpy()


def summarize(args, rows, contract, reference):
    from decode_uniego_torch import decode_transforms
    from kimodo.metrics.foot_skate import FootSkateFromHeight
    from nymeria_baselines.evaluate_uem_phase3 import measure, BODY_JOINTS, CONTACT_JOINTS
    mean, std = np.load(contract['motion_stats_mean']), np.load(contract['motion_stats_std'])
    skate = FootSkateFromHeight(SimpleNamespace(foot_joint_idx=list(CONTACT_JOINTS)), fps=20)
    records = []
    reference_gt = {r['id']: r['metrics']['GT'] for r in read(args.reference / 'per_case.json')}
    for row in rows:
        with np.load(row['output']) as a:
            values = {k: a[k].copy() for k in ('generated', 'gt_motion')}
        positions = {k: np.take(decode_transforms(torch.from_numpy(v*std+mean)[None])[0].numpy(),
                               BODY_JOINTS, axis=1)[..., :3, 3] for k, v in values.items()}
        gt = measure(positions['gt_motion'], positions['gt_motion'], skate)
        for key in ('foot_skate_cm_s', 'foot_depth_mean_cm', 'all_feet_above5cm_frame_fraction'):
            assert abs(gt[key] - reference_gt[row['id']][key]) < 1e-6, (row['id'], key)
        records.append(dict(id=row['id'], GT=gt,
                            model=measure(positions['generated'], positions['gt_motion'], skate)))
    aggregate = {}
    for source in ('GT', 'model'):
        aggregate[source] = {}
        for key in records[0][source]:
            values = np.asarray([r[source][key] for r in records], dtype=np.float64)
            assert np.isfinite(values).all()
            aggregate[source][key] = dict(mean=float(values.mean()), median=float(np.median(values)),
                                         std=float(values.std()), min=float(values.min()), max=float(values.max()))
    return dict(n=len(rows), body_joints=list(BODY_JOINTS), aggregate=aggregate, per_sequence=records,
                protocol='Same body23, GT decoder, units and floor/skate metrics as UEM/EgoAllo/Phase3 comparison')


def comparison(args):
    summaries = {}
    for run in RUNS:
        for condition in CONDITIONS:
            path = args.output / run / condition / 'summary.json'
            if path.exists():
                summaries[run+'/'+condition] = read(path)
    atomic_json(args.output / 'comparison.json', summaries)
    with (args.output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Model', 'Step', 'Condition', 'Source', 'N'] + [c[0] for c in COLUMNS])
        for key, summary in summaries.items():
            run, condition = key.split('/')
            for source in ('GT', 'model'):
                writer.writerow([run, 200000, condition, source, summary['n']] +
                                [summary['aggregate'][source][metric]['mean']*scale for _, metric, scale in COLUMNS])
    return summaries


def render(args, records, contract):
    from PIL import Image, ImageDraw
    from decode_uniego_torch import decode_joints
    from render_motion import render_single, head_rotation_from_features, write_mp4
    mean, std = np.load(contract['motion_stats_mean']), np.load(contract['motion_stats_std'])
    gallery = args.output / 'gallery'
    gallery.mkdir(exist_ok=True)
    entries = []
    for condition, spec in CONDITIONS.items():
        inputs = ' + '.join(label for field, label in [('video', 'RGB video'), ('camera', 'camera trajectory'),
                                                      ('text', 'text')] if spec[field])
        for row in records[::6]:
            stem = condition + '__' + row['id']
            path = gallery / (stem + '.mp4')
            if not path.exists():
                with np.load(row['bundle']) as a:
                    normalized = {'GT': a['motion'].copy()}
                for run in RUNS:
                    with np.load(args.output / run / condition / (row['id']+'.npz')) as a:
                        normalized[run] = a['generated'].copy()
                panels = []
                for label, value in normalized.items():
                    raw = value*std+mean
                    joints = decode_joints(torch.from_numpy(raw)[None])[0].numpy()
                    frames = render_single(joints, caption='', width=512, height=512,
                                           color='tab:blue' if label == 'GT' else 'tab:orange',
                                           head_rotation0=head_rotation_from_features(raw))
                    decorated = []
                    for frame in frames:
                        canvas = Image.fromarray(frame)
                        draw = ImageDraw.Draw(canvas)
                        color = '#245c91' if label == 'GT' else '#a75916'
                        draw.rectangle((0, 0, 511, 511), outline=color, width=4)
                        draw.rectangle((0, 0, 511, 27), fill='#18754b')
                        draw.text((8, 8), 'Observed: '+inputs, fill='white')
                        draw.text((8, 34), 'GT reference' if label == 'GT' else LABELS[label]+' | 200k', fill=color)
                        decorated.append(np.asarray(canvas))
                    panels.append(np.stack(decorated))
                frames = np.concatenate([np.concatenate(panels[:2], axis=2), np.concatenate(panels[2:], axis=2)], axis=1)
                write_mp4(frames, str(path), fps=20)
                Image.fromarray(frames[48]).save(gallery/(stem+'.jpg'), quality=92)
            entries.append(dict(condition=condition, id=row['id'], video=str(path), caption=row['caption']))
            atomic_json(gallery / 'manifest.json', entries)
            progress(args, 'rendering', completed=len(entries), total=48)
    cards = [f'<h2>{html.escape(r["condition"])}: {html.escape(r["id"])}</h2><p>{html.escape(r["caption"])}</p>'
             f'<video controls preload="none" src="{Path(r["video"]).name}" style="width:100%"></video>' for r in entries]
    (gallery/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Phase3 conditioning matrix</title>'
        '<h1>GT / original / joint-generation / camera-mix, all 200k</h1>'
        '<p>Top row: GT, original. Bottom row: joint-generation, camera-mix. Green labels identify observed inputs; '
        'amber borders identify generated motion. Fixed 12 examples per setting, 97 frames at 20 fps. '
        'All models use native neutral shape. Novel conditioning combinations are inference diagnostics, not trained-task parity.</p>'
        + ''.join(cards))


def main(args):
    require_compute_node()
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    args.output.mkdir(parents=True, exist_ok=True)
    reference = read(args.reference / 'configuration.json')
    records = reference['cases']
    split = read(args.camera_root.parent / 'train_test_split.json')
    ids = {r['uuid'] for r in records}
    assert len(records) == len(ids) == 71 and ids <= set(split['test']) and not ids & set(split['train'])
    assert all(r['caption'].strip() for r in records)
    payloads, checkpoints = {}, {}
    for run, directory in RUNS.items():
        path = args.run_root / directory / 'checkpoints/step_000200000.pt'
        progress(args, 'loading_checkpoint', run=run, path=str(path))
        payloads[run] = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        assert payloads[run]['step'] == 200000
        checkpoints[run] = dict(path=str(path), sha256=sha256(path), step=200000)
    contract = configure(payloads['baseline'])
    for payload in payloads.values():
        assert payload['sources']['phase2_contract'] == contract == reference['phase3_contract']
        assert payload['sources']['phase1_metadata_sha256'] == payloads['baseline']['sources']['phase1_metadata_sha256']
    from motion_expert_phase3_edge import quantitative as metrics
    from motion_expert_phase3_edge_camcond.model import EdgePhase3
    cameras, camera_evidence = {}, {}
    for row in records:
        cameras[row['id']], camera_evidence[row['id']] = load_camera(row, args.camera_root)
    configuration = dict(checkpoints=checkpoints, cases=records, cases_sha256=digest_json(records),
        reference=str(args.reference), reference_config_sha256=sha256(args.reference/'configuration.json'),
        sources=payloads['baseline']['sources'], conditions=CONDITIONS, steps=35, shift=3., guidance=1.,
        frames=97, fps=20, seeds='Unchanged original per-case seeds 0..70',
        camera_inputs=camera_evidence, camera_representation='Raw native backward-framewise rot6d + translation, padded9->64, domain2; scale1',
        neutral_shape='Same native neutral shape for all checkpoints and settings',
        inference_adapter='Existing camera-capable forward with original checkpoint bridge weights; explicit caption packing in both reasoner paths',
        no_training_changes=True, no_head_inpainting=True, no_prediction_postprocessing=True,
        ood_caveats=['Camera tasks were not trained by original or joint-generation bridges.',
                    'Video+text and video+trajectory+text were not explicit training tasks for any of the three bridge runs.',
                    'Equal optimizer step does not imply equal exposure to each training task.'],
        implementation_sha256=sha256(Path(__file__)))
    config_path = args.output/'configuration.json'
    if config_path.exists():
        assert read(config_path) == configuration, 'Refusing incompatible resume'
    atomic_json(config_path, configuration)
    progress(args, 'loading_model')
    model = EdgePhase3(payloads['baseline']['sources']['phase1'], payloads['baseline']['sources']['phase2'],
                       device='cuda', activation_checkpointing=False)
    assert model.provenance['phase2_contract'] == contract
    assert model.provenance['phase1_metadata_sha256'] == configuration['sources']['phase1_metadata_sha256']
    model.eval()
    first = records[0]
    first_values = tensors(*metrics.check_bundle(first, contract), cameras[first['id']])
    audits = {}
    for run in RUNS:
        model.bridges.load_state_dict(payloads[run]['bridges'], strict=True)
        progress(args, 'preflight', run=run)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            audits[run] = preflight(model, first_values, first, run)
        atomic_json(args.output/'preflight.json', audits)
    del first_values
    for run in RUNS:
        model.bridges.load_state_dict(payloads[run]['bridges'], strict=True)
        for condition, spec in CONDITIONS.items():
            destination = args.output/run/condition
            destination.mkdir(parents=True, exist_ok=True)
            rows = []
            for index, row in enumerate(records):
                motion, neutral, video = metrics.check_bundle(row, contract)
                path = destination/(row['id']+'.npz')
                started = time.monotonic()
                if not path.exists():
                    values = tensors(motion, neutral, video, cameras[row['id']])
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                        prediction = sample(model, values, row, spec)
                    temporary = path.with_suffix('.tmp.npz')
                    np.savez(temporary, generated=prediction, gt_motion=motion, neutral=neutral,
                             seed=row['seed'], condition=condition, run=run)
                    temporary.replace(path)
                    del values, prediction
                with np.load(path) as a:
                    assert a['generated'].shape == (97, 283) and np.isfinite(a['generated']).all()
                    assert np.array_equal(a['gt_motion'], motion) and int(a['seed']) == row['seed']
                    assert str(a['condition']) == condition and str(a['run']) == run
                rows.append(dict(**row, output=str(path)))
                atomic_json(destination/'manifest.json', dict(run=run, condition=condition, cases=rows,
                                                             configuration=str(config_path)))
                progress(args, 'sampling', run=run, condition=condition, complete=index+1, total=71,
                         seconds=time.monotonic()-started)
            progress(args, 'metrics', run=run, condition=condition)
            summary = summarize(args, rows, contract, reference)
            atomic_json(destination/'summary.json', dict(**summary, run=run, condition=condition, step=200000))
            comparison(args)
    del model, payloads
    gc.collect()
    torch.cuda.empty_cache()
    summaries = comparison(args)
    assert len(summaries) == 12 and all(s['n'] == 71 for s in summaries.values())
    atomic_json(args.output/'METRICS_COMPLETE.json', dict(settings=12, cases_per_setting=71))
    render(args, records, contract)
    atomic_json(args.output/'COMPLETE.json', dict(settings=12, cases_per_setting=71, videos=48))
    progress(args, 'complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--camera-root', type=Path, required=True)
    arguments = parser.parse_args()
    try:
        main(arguments)
    except BaseException as error:
        arguments.output.mkdir(parents=True, exist_ok=True)
        progress(arguments, 'failed', error=str(error), traceback=traceback.format_exc())
        raise
