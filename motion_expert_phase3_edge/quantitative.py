"""Offline Edge Phase-3 reconstruction and video metrics with matched GT rows."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from motion_expert_phase3_edge.settings import atomic_json, digest_json, require_compute_node, sha256


def progress(args, stage, **fields):
    value = dict(stage=stage, mode=args.mode, job=os.environ.get('SLURM_JOB_ID'), time=time.time(), **fields)
    print(json.dumps(value), flush=True)
    atomic_json(args.input_root / args.mode / 'progress.json', value)


def aggregate(rows):
    keys = list(rows[0])
    result = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise FloatingPointError(f'Non-finite metric {key}')
        result[key] = dict(mean=float(values.mean()), median=float(np.median(values)),
                           std=float(values.std()), min=float(values.min()), max=float(values.max()))
    return result


def check_bundle(row, contract):
    with np.load(row['bundle'], allow_pickle=False) as bundle:
        if (str(bundle['motion_mean_sha256'].item()) != contract['motion_mean']
                or str(bundle['motion_std_sha256'].item()) != contract['motion_std']):
            raise RuntimeError('Case normalization hashes do not match the checkpoint')
        motion, neutral, video = (bundle[k].copy() for k in ('motion', 'neutral', 'video'))
    if motion.shape != (97, 283) or neutral.shape != (30, 3) or video.shape != (48, 25, 16, 16):
        raise ValueError(f'Wrong case geometry: {row["id"]}')
    if not all(np.isfinite(x).all() for x in (motion, neutral, video)):
        raise FloatingPointError(row['id'])
    return motion, neutral, video.astype(np.float32)


def sample_all(args, payload, configuration, contract, records):
    from motion_expert_phase3_edge.model import EdgePhase3
    from motion_expert_phase3_edge.sample import sample

    out = args.input_root / args.mode
    out.mkdir(parents=True, exist_ok=True)
    signature = dict(checkpoint=str(args.checkpoint), step=int(payload['step']), sources=payload['sources'],
                     cases_sha256=digest_json(records), mode=args.mode, steps=35, guidance=1.0,
                     shift=3.0 if args.mode == 'video2motion' else 10.0,
                     cohort='one first-valid aligned T97 window per sorted held-out recording')
    signature_file = out / 'signature.json'
    if signature_file.exists() and json.loads(signature_file.read_text()) != signature:
        raise RuntimeError('Existing quantitative output has a different signature')
    atomic_json(signature_file, signature)
    reused = {}
    if args.reuse_root:
        old = json.loads((args.reuse_root / args.mode / 'manifest.json').read_text())
        if (old['step'] != payload['step'] or old['sources'] != payload['sources']
                or old['steps'] != 35 or old['guidance'] != 1.0 or old['shift'] != signature['shift']):
            raise RuntimeError('Requested visualization reuse has incompatible provenance')
        reused = {(r['uuid'], int(r['start']), int(r['seed'])): r for r in old['cases']}
    model = None
    completed = []
    started = time.monotonic()
    for index, row in enumerate(records):
        path = out / (row['id'] + '.npz')
        motion, neutral, video = check_bundle(row, contract)
        old = reused.get((row['uuid'], int(row['start']), int(row['seed'])))
        source = 'existing_result' if path.exists() else 'sampled'
        if not path.exists() and old is not None:
            with np.load(old['output'], allow_pickle=False) as value:
                if not np.array_equal(value['gt_motion'], motion) or not np.array_equal(value['gt_video_latents'], video):
                    raise RuntimeError('Visualization reuse GT differs from this cohort')
                generated = value['generated'].copy()
            source = 'saved_visualization'
        elif not path.exists():
            if model is None:
                progress(args, 'loading_model', complete=index, total=len(records))
                model = EdgePhase3(payload['sources']['phase1'], payload['sources']['phase2'],
                                   device='cuda', activation_checkpointing=False)
                if (model.provenance['phase1_metadata_sha256'] != payload['sources']['phase1_metadata_sha256']
                        or model.provenance['phase2_contract'] != payload['sources']['phase2_contract']):
                    raise RuntimeError('Frozen specialist identity mismatch')
                model.bridges.load_state_dict(payload['bridges'], strict=True)
                model.eval()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = sample(model, mode=args.mode, caption=row['caption'],
                    motion=torch.from_numpy(motion)[None].cuda(), neutral=torch.from_numpy(neutral)[None].cuda(),
                    video=torch.from_numpy(video)[None].cuda(), seed=int(row['seed']), steps=35, guidance=1.0)
            generated = prediction[0].float().cpu().numpy()
            del prediction
        else:
            with np.load(path, allow_pickle=False) as value:
                generated = value['generated'].copy()
        expected = (97, 283) if args.mode == 'video2motion' else (48, 25, 16, 16)
        if generated.shape != expected or not np.isfinite(generated).all():
            raise FloatingPointError(f'Invalid generated data: {row["id"]}')
        if args.mode == 'motimg2video' and not np.array_equal(generated[:, 0], video[:, 0]):
            raise RuntimeError('Conditioned first latent changed')
        if not path.exists():
            temporary = path.with_name(path.name + f'.tmp.{os.getpid()}.npz')
            np.savez(temporary, generated=generated, gt_motion=motion, neutral=neutral, gt_video_latents=video)
            os.replace(temporary, path)
        completed.append(dict(**row, output=str(path), source=source))
        atomic_json(out / 'manifest.json', dict(**signature, cases=completed))
        elapsed = time.monotonic() - started
        progress(args, 'sampling', complete=index + 1, total=len(records), elapsed_seconds=elapsed)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return completed


def motion_metrics(args, records, contract):
    from eval_motion_recon import recon_metrics, decode_to_joints
    from phase2_quantitative import floor_metrics
    from kimodo.metrics.foot_skate import FootSkateFromHeight
    from kimodo.skeleton import SOMASkeleton30

    mean = np.load(contract['motion_stats_mean']).astype(np.float32)
    std = np.load(contract['motion_stats_std']).astype(np.float32)
    skeleton = SOMASkeleton30()
    feet = [int(j) for j in skeleton.foot_joint_idx]
    length = torch.tensor([97])
    skate = FootSkateFromHeight(skeleton=skeleton, fps=20.0)
    rows = []
    for index, row in enumerate(records):
        with np.load(row['output'], allow_pickle=False) as value:
            generated, gt = value['generated'].copy(), value['gt_motion'].copy()
        gen_joints = decode_to_joints(generated * std + mean)
        gt_joints = decode_to_joints(gt * std + mean)
        sources = {}
        for name, normalized, joints in (('model', generated, gen_joints), ('GT', gt, gt_joints)):
            result = recon_metrics(normalized, gt, mean, std)
            tensor = torch.from_numpy(joints)[None]
            result.update({key: float(val[0]) for key, val in floor_metrics(tensor, length, feet).items()})
            result['foot_skate_cm_s'] = float(skate(posed_joints=tensor, lengths=length)['foot_skate_from_height'][0]) * 100
            result['root_fde_m'] = float(np.linalg.norm(joints[-1, 0] - gt_joints[-1, 0]))
            displacement = (joints[:, 0] - joints[0, 0]) - (gt_joints[:, 0] - gt_joints[0, 0])
            result['root_relative_ade_m'] = float(np.linalg.norm(displacement, axis=-1).mean())
            result['accel_err_m_s2'] = result['accel_err'] * 400
            result['accel_m_s2'] = result['accel_pred'] * 400
            sources[name] = result
        rows.append(dict(id=row['id'], uuid=row['uuid'], start=row['start'], **sources))
        atomic_json(args.input_root / args.mode / 'per_sequence.json', rows)
        progress(args, 'metrics', complete=index + 1, total=len(records))
    return dict(n=len(rows), aggregate={name: aggregate([r[name] for r in rows]) for name in ('model', 'GT')},
                per_sequence=rows, metric_policy={
                    'normalization': 'exact checkpoint motion mean/std; no TMR evaluator involved',
                    'mpjpe': 'unaligned world-joint Euclidean error, meters',
                    'pa_mpjpe': 'one whole-clip scale/rotation/translation Umeyama fit, not per-frame alignment',
                    'root_err_m': 'unaligned root trajectory ADE in meters',
                    'accel_err': 'second difference in meters/frame^2; m/s^2 counterpart multiplies by fps^2',
                    'foot_skate': 'existing Kimodo height-based toe velocity, cm/s; height threshold 5cm',
                    'penetration': 'mean per-frame maximum negative height, cm; feet and all-joints reported separately',
                    'airborne': 'fraction of frames with all four feet above 5cm',
                    'no_postprocessing': 'no prediction grounding, canonical reset or trajectory alignment except explicit PA metric'})


def video_metrics(args, records, configuration):
    from motion_expert_phase3_edge.visualize import load_vae
    from native_phase_training.evaluate_inverse_forward import LPIPSAlex, _frame_metrics, _summarize_frame_metrics

    vae = load_vae(configuration['vae_path'], '256', 97, 'cuda')
    if vae.get_output_range() != '[-1,1]':
        raise RuntimeError('Unexpected Wan output range')
    perceptual = LPIPSAlex('cuda', batch_size=16)
    rows = []
    for index, row in enumerate(records):
        with np.load(row['output'], allow_pickle=False) as value:
            latents = {key: value[key].copy() for key in ('generated', 'gt_video_latents')}
        with np.load(row['bundle'], allow_pickle=False) as bundle:
            gt = bundle['rgb'].copy()
        if gt.shape != (97, 256, 256, 3) or gt.dtype != np.uint8:
            raise ValueError('Unexpected reference RGB')
        decoded = {}
        for name, key in (('model', 'generated'), ('VAE_GT', 'gt_video_latents')):
            with torch.no_grad():
                pixels = vae.decode(torch.from_numpy(latents[key])[None].cuda())
            if pixels.shape != (1, 3, 97, 256, 256) or not torch.isfinite(pixels).all():
                raise FloatingPointError('Invalid decoded video')
            decoded[name] = ((pixels[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()
            del pixels
        result, frame_values = {}, {}
        for name, predicted in (('model', decoded['model']), ('VAE_GT', decoded['VAE_GT']), ('GT', gt)):
            values = _frame_metrics(gt[1:], predicted[1:], perceptual)
            for key, value in values.items():
                if len(value) != 96 or not np.isfinite(value).all():
                    raise FloatingPointError(f'Invalid {name} {key}')
                frame_values[name + '_' + key] = value
            result[name] = _summarize_frame_metrics(values)
        np.savez(args.input_root / args.mode / (row['id'] + '_frame_metrics.npz'), **frame_values)
        rows.append(dict(id=row['id'], uuid=row['uuid'], start=row['start'], **result))
        atomic_json(args.input_root / args.mode / 'per_sequence.json', rows)
        progress(args, 'metrics', complete=index + 1, total=len(records))
    keys = ('psnr_db', 'ssim', 'lpips_alex')
    return dict(n=len(rows), aggregate={name: aggregate([{k: r[name][k] for k in keys} for r in rows])
                                      for name in ('model', 'VAE_GT', 'GT')},
                horizons={name: {h: aggregate([r[name]['horizons'][h] for r in rows])
                                 for h in rows[0][name]['horizons']} for name in ('model', 'VAE_GT', 'GT')},
                per_sequence=rows, metric_policy={
                    'frames': 'RGB frames 1..96 only; conditioned frame0 excluded',
                    'pixels': 'direct native Wan decode, clipped and rounded to uint8; no MP4 compression',
                    'GT': 'raw held-out resized RGB against itself; PSNR identity uses existing 100dB cap',
                    'VAE_GT': 'GT latents decoded through the same VAE against raw GT, showing compression ceiling',
                    'sampler': 'UniPC35, guidance1, motion shift3 / video shift10',
                    'interpretation': 'paired reconstruction metrics; not a distributional FID/FVD benchmark'})


def evaluate(args):
    from motion_expert_phase3_edge.visualize import configure

    torch.cuda.set_device(0)
    payload = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    contract = configure(payload)
    configuration = json.loads((args.input_root / 'configuration.json').read_text())
    if configuration['sources'] != payload['sources'] or int(configuration['step']) != int(payload['step']):
        raise RuntimeError('Prepared cohort is bound to another checkpoint')
    records = configuration['cases']
    split = json.loads(Path(configuration['latent_contract']['split_file']).read_text())
    if len(records) != args.expected_cases or len({r['uuid'] for r in records}) != args.expected_cases:
        raise RuntimeError('Wrong case count or duplicate held-out recordings')
    if not {r['uuid'] for r in records} <= set(split['test']) or {r['uuid'] for r in records} & set(split['train']):
        raise RuntimeError('Evaluation split leakage')
    sampled = sample_all(args, payload, configuration, contract, records)
    progress(args, 'metrics', complete=0, total=len(records))
    result = (motion_metrics(args, sampled, contract) if args.mode == 'video2motion'
              else video_metrics(args, sampled, configuration))
    result.update(step=int(payload['step']), checkpoint=str(args.checkpoint), mode=args.mode,
                  sources=payload['sources'], cases_sha256=digest_json(records))
    atomic_json(args.input_root / args.mode / 'summary.json', result)
    progress(args, 'complete', cases=len(records), summary=str(args.input_root / args.mode / 'summary.json'))


def summarize(args):
    motion = json.loads((args.input_root / 'video2motion/summary.json').read_text())
    video = json.loads((args.input_root / 'motimg2video/summary.json').read_text())
    configuration = json.loads((args.input_root / 'configuration.json').read_text())
    if motion['cases_sha256'] != video['cases_sha256'] or motion['n'] != video['n'] or motion['step'] != video['step']:
        raise RuntimeError('V2M/M2V cohorts or checkpoints differ')
    cohort = dict(policy='first valid aligned window per sorted test recording', n=motion['n'])
    if args.reference_windows:
        old = json.loads(args.reference_windows.read_text())
        prior = {r['uuid']: int(r['start']) for r in old}
        current = {r['uuid']: int(r['start']) for r in configuration['cases']}
        cohort.update(reference=str(args.reference_windows), same_recordings=set(prior) == set(current),
                      changed_windows=[dict(uuid=u, previous_start=prior.get(u), current_start=s)
                                       for u, s in current.items() if prior.get(u) != s])
    lines = [f'# Edge Phase 3, step {motion["step"]}: {motion["n"]} held-out Nymeria recordings', '',
             'UniPC35, guidance1, T97 at20fps. One fixed floor-calibrated valid window per recording.', '',
             '## Video to motion', '',
             '| Source | MPJPE cm | PA-MPJPE cm | Root ADE cm | Root FDE cm | Skate cm/s | Foot penetration cm | Airborne % |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for source in ('GT', 'model'):
        a = motion['aggregate'][source]
        vals = [a[k]['mean'] * scale for k, scale in (
            ('mpjpe_m',100), ('pa_mpjpe_m',100), ('root_err_m',100), ('root_fde_m',100),
            ('foot_skate_cm_s',1), ('foot_depth_mean_cm',1), ('all_feet_above5cm_frame_fraction',100))]
        lines.append('| ' + source + ' | ' + ' | '.join(f'{v:.4f}' for v in vals) + ' |')
    lines += ['', 'PA uses whole-clip similarity alignment; all other position/trajectory errors are unaligned.',
              'Skating is height-based toe velocity; penetration is mean per-frame deepest foot penetration.', '',
              '## Motion plus first image to video', '',
              '| Source | PSNR dB | SSIM | LPIPS-Alex |', '|---|---:|---:|---:|']
    for source in ('GT', 'VAE_GT', 'model'):
        a = video['aggregate'][source]
        lines.append('| ' + source + ' | ' + ' | '.join(f'{a[k]["mean"]:.5f}' for k in ('psnr_db','ssim','lpips_alex')) + ' |')
    lines += ['', 'Frames1..96 only. Direct decoded RGB, not compressed MP4. GT identity PSNR is capped at100dB.',
              'VAE_GT measures reconstruction of held-out GT latents, not a learned bridge prediction.', '',
              f'Cohort differences from historical windows: {len(cohort.get("changed_windows", []))}.',
              'Do not compare directly with historical Nano scores without matching windows and sampler steps.']
    (args.input_root / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    atomic_json(args.input_root / 'cohort_comparison.json', cohort)
    atomic_json(args.input_root / 'COMPLETE.json', dict(step=motion['step'], n=motion['n'],
                cases_sha256=motion['cases_sha256'], reports=['video2motion/summary.json', 'motimg2video/summary.json']))
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['evaluate', 'summarize'])
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--mode', choices=['video2motion', 'motimg2video'])
    parser.add_argument('--reuse-root', type=Path)
    parser.add_argument('--expected-cases', type=int, default=71)
    parser.add_argument('--reference-windows', type=Path)
    args = parser.parse_args()
    require_compute_node()
    torch.set_num_threads(1)
    if args.command == 'summarize':
        summarize(args)
        return
    if not args.mode or not args.checkpoint:
        parser.error('evaluate requires --mode and --checkpoint')
    try:
        evaluate(args)
    except BaseException as error:
        progress(args, 'failed', error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
