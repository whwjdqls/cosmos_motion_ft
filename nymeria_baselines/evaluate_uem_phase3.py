"""Native UniEgoMotion inference and matched body-only scoring of saved Phase 3."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import os
from pathlib import Path
import socket
import time
import traceback

import numpy as np
import torch

from nymeria_baselines.body_shape import BODY_JOINTS, CONTACT_JOINTS, MOTION_INDICES, PARENTS


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def announce(args, stage, **values):
    record = dict(stage=stage, time=time.time(), job=os.environ.get('SLURM_JOB_ID'), **values)
    save(args.output / 'progress.json', record)
    print(json.dumps(record), flush=True)


def sample(args):
    from yacs.config import CfgNode
    from module.uem_module import UEM_Module
    from nymeria_baselines.prepare import prepare_window

    configuration = read(args.phase3 / 'configuration.json')
    plain = read(args.phase3 / 'plain/manifest.json')
    oracle = read(args.phase3 / 'gt_head/manifest.json')
    cases = configuration['cases']
    assert len(cases) == len({r['uuid'] for r in cases}) == 71
    for manifest in (plain, oracle):
        assert [(r['id'], r['uuid'], r['start'], r['seed']) for r in manifest['cases']] == [
            (r['id'], r['uuid'], r['start'], r['seed']) for r in cases]
    assert plain['checkpoint_sha256'] == oracle['checkpoint_sha256']
    contract = configuration['sources']['phase2_contract']
    mean_path, std_path = contract['motion_stats_mean'], contract['motion_stats_std']
    assert sha(mean_path) == contract['motion_mean']
    assert sha(std_path) == contract['motion_std']
    pmean, pstd = np.load(mean_path), np.load(std_path)
    index = read(args.cache / 'index.json')
    test_rows = [r for r in index['rows'] if r['split'] == 'test']
    arrays = {r['uuid']: r['array'] for r in test_rows}
    eligible = {(r['uuid'], r['start']) for r in test_rows}
    assert set(r['uuid'] for r in cases) <= set(arrays)
    assert not set(r['uuid'] for r in cases) & {r['uuid'] for r in index['rows'] if r['split'] != 'test'}
    with np.load(args.cache / 'train_stats.npz') as stats:
        mean, std = stats['motion_mean'][MOTION_INDICES], stats['motion_std'][MOTION_INDICES]
        cmean, cstd = stats['camera_mean'], stats['camera_std']
    announce(args, 'loading_native_uem', checkpoint=str(args.checkpoint))
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = CfgNode(checkpoint['hyper_parameters'])
    assert Path(cfg.DATA.DATA_DIR).resolve() == args.cache.resolve()
    assert cfg.DATA.WINDOW == 97 and cfg.DATA.REPRE_TYPE == 'soma_body230'
    assert cfg.DATA.COND_TRAJ and not cfg.DATA.COND_BETAS and not cfg.DATA.COND_IMG_FEAT
    cfg.TRAIN.ONLY_VALIDATE = True
    model = UEM_Module(cfg)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.cuda().eval()
    policy = dict(
        cases=cases, frames=97, fps=20, body_joints=list(BODY_JOINTS),
        checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint),
        epoch=int(checkpoint['epoch']), global_step=int(checkpoint['global_step']),
        weights='state_dict, matching native run/vis_uem.py; no EMA substitution',
        sampler='unchanged UEM_Module.sample / diffusion.p_sample_loop',
        diffusion_steps=int(cfg.MODEL.DIFFUSION_STEPS), cond_scale=cfg.TRAIN.COND_SCALE,
        cache=str(args.cache), cache_index_sha256=sha(args.cache / 'index.json'),
        uem_stats_sha256=sha(args.cache / 'train_stats.npz'),
        phase3_root=str(args.phase3), phase3_checkpoint=plain['checkpoint'],
        phase3_checkpoint_sha256=plain['checkpoint_sha256'],
        phase3_manifest_sha256={k: sha(args.phase3 / k / 'manifest.json') for k in ('plain', 'gt_head')},
        phase3_contract=contract, phase3_sampler=dict(steps=35, guidance=1.0, shift=3.0),
        comparison='Matched body23 reconstruction; not equal-information conditioning or FID.',
        inputs={'UniEgoMotion': 'All 97 measured camera poses, no video/text/GT shape',
                'Phase3_V2M': 'All 97 video frames plus native neutral shape; empty caption',
                'Phase3_oracle_head': 'Video + neutral shape + oracle GT head/global features'},
        alignment='Native frame-0 head-yaw canonical motion frame; no prediction reset, fit, or grounding. Whole-clip similarity fit only for explicitly labeled PA-MPJPE.',
        shape='UEM predicts 10 SOMA shape PCA channels; saved, not supplied as input. Body positions decoded from predicted motion features.',
        qualitative='Reuse existing GT and both Phase3 panels pixel-for-pixel before final video encoding; only render new UEM panel.',
    )
    if (args.output / 'configuration.json').exists():
        assert read(args.output / 'configuration.json') == policy, 'Refusing to mix evaluation configurations'
    save(args.output / 'configuration.json', policy)
    del checkpoint
    audits = []
    started = time.monotonic()
    for number, row in enumerate(cases):
        key, start = arrays[row['uuid']], int(row['start'])
        motion = np.load(args.cache / 'records' / f'{key}.motion.npy', mmap_mode='r')[start:start+97]
        camera = np.load(args.cache / 'records' / f'{key}.pose_yup.npy', mmap_mode='r')[start:start+97]
        raw, trajectory = prepare_window(motion, camera, row['floor_offset'])
        with np.load(row['bundle']) as bundle:
            assert str(bundle['motion_mean_sha256'].item()) == contract['motion_mean']
            assert str(bundle['motion_std_sha256'].item()) == contract['motion_std']
            gt = bundle['motion'] * pstd + pmean
        delta = float(np.max(np.abs(raw - gt)))
        assert delta < 2e-4, (row['id'], 'GT/floor/canonicalization mismatch', delta)
        traj = ((trajectory - cmean) / (cstd + 1e-6)).astype(np.float32)
        assert np.isfinite(traj).all()
        path = args.output / 'predictions' / (row['id'] + '.npz')
        if not path.exists():
            torch.manual_seed(int(row['seed']))
            torch.cuda.manual_seed_all(int(row['seed']))
            y = dict(traj=torch.from_numpy(traj[None]).cuda(),
                     valid_frames=torch.ones(1, 97, dtype=torch.long, device='cuda'))
            with torch.inference_mode():
                prediction = model.sample(y, B=1, cond_scale=cfg.TRAIN.COND_SCALE)[0].cpu().numpy()
            assert prediction.shape == (97, 230) and np.isfinite(prediction).all()
            path.parent.mkdir(exist_ok=True)
            temporary = path.with_suffix('.tmp.npz')
            np.savez(temporary, normalized=prediction,
                     raw_body=prediction[:, :220] * (std + 1e-6) + mean,
                     shape_pca10=prediction[:, 220:], camera_normalized=traj,
                     gt_body=gt[:, MOTION_INDICES], seed=int(row['seed']))
            os.replace(temporary, path)
        audits.append(dict(id=row['id'], gt_feature_max_difference=delta,
                           window_in_clean_test_index=(row['uuid'], row['start']) in eligible))
        save(args.output / 'input_audit.json', audits)
        announce(args, 'sampling', complete=number+1, total=len(cases),
                 elapsed_seconds=time.monotonic()-started)
    save(args.output / 'INFERENCE_COMPLETE.json', dict(cases=len(cases)))


def measure(joints, target, skate):
    from eval_motion_recon import joint_metrics
    from phase2_quantitative import floor_metrics
    result = joint_metrics(joints, target)
    tensor, lengths = torch.from_numpy(joints)[None], torch.tensor([len(joints)])
    result.update({k: float(v[0]) for k, v in floor_metrics(tensor, lengths, list(CONTACT_JOINTS)).items()})
    result['foot_skate_cm_s'] = float(skate(posed_joints=tensor, lengths=lengths)['foot_skate_from_height'][0])*100
    result['root_fde_m'] = float(np.linalg.norm(joints[-1, 0]-target[-1, 0]))
    displacement = (joints[:, 0]-joints[0, 0])-(target[:, 0]-target[0, 0])
    result['root_relative_ade_m'] = float(np.linalg.norm(displacement, axis=-1).mean())
    result['accel_err_m_s2'] = result['accel_err']*400
    result['accel_m_s2'] = result['accel_pred']*400
    result['head_error_cm'] = float(np.linalg.norm(joints[:, 6]-target[:, 6], axis=-1).mean())*100
    other = [j for j in range(23) if j != 6]
    result['nonhead22_mpjpe_cm'] = float(np.linalg.norm(joints[:, other]-target[:, other], axis=-1).mean())*100
    assert all(np.isfinite(v) for v in result.values())
    return result


def report(args):
    from types import SimpleNamespace
    from decode_uniego_torch import decode_transforms
    from kimodo.metrics.foot_skate import FootSkateFromHeight
    from motion_expert_phase3_edge.quantitative import aggregate

    config = read(args.output / 'configuration.json')
    contract = config['phase3_contract']
    assert sha(contract['motion_stats_mean']) == contract['motion_mean']
    assert sha(contract['motion_stats_std']) == contract['motion_std']
    mean, std = np.load(contract['motion_stats_mean']), np.load(contract['motion_stats_std'])
    manifests = {}
    for name in ('plain', 'gt_head'):
        path = args.phase3 / name / 'manifest.json'
        assert sha(path) == config['phase3_manifest_sha256'][name]
        manifests[name] = {r['id']: r for r in read(path)['cases']}
    skate = FootSkateFromHeight(skeleton=SimpleNamespace(foot_joint_idx=list(CONTACT_JOINTS)), fps=20)
    records, decoded = [], {}
    for row in config['cases']:
        uid = row['id']
        with np.load(row['bundle']) as bundle:
            gt_norm = bundle['motion'].copy()
        with np.load(args.output / 'predictions' / (uid + '.npz')) as data:
            raw = data['raw_body'].copy()
        uem_tf = decode_transforms(torch.from_numpy(raw)[None], n_joints=23)[0].numpy()
        gt_tf = np.take(decode_transforms(torch.from_numpy(gt_norm*std+mean)[None])[0].numpy(),
                        BODY_JOINTS, axis=1)
        positions = {'GT': gt_tf[..., :3, 3], 'UniEgoMotion': uem_tf[..., :3, 3]}
        provenance = {}
        for name, label in [('plain', 'Phase3_V2M'), ('gt_head', 'Phase3_oracle_head')]:
            path = manifests[name][uid]['output']
            with np.load(path) as saved:
                assert np.array_equal(saved['gt_motion'], gt_norm), 'Saved Phase3 GT differs'
                pred = saved['generated']*std+mean
            transforms = decode_transforms(torch.from_numpy(pred)[None])[0].numpy()
            positions[label] = np.take(transforms[..., :3, 3], BODY_JOINTS, axis=1)
            provenance[label] = dict(path=path, sha256=sha(path))
        records.append(dict(id=uid, uuid=row['uuid'], start=row['start'], seed=row['seed'],
                            metrics={name: measure(p, positions['GT'], skate) for name, p in positions.items()},
                            phase3_reused=provenance))
        decoded[uid] = (positions['UniEgoMotion'], uem_tf[0, 6, :3, :3])
        save(args.output / 'per_case.json', records)
        announce(args, 'metrics', complete=len(records), total=len(config['cases']))
    models = list(records[0]['metrics'])
    summary = dict(n=len(records), body_joints=list(BODY_JOINTS),
                   aggregate={m: aggregate([r['metrics'][m] for r in records]) for m in models},
                   policy=config['alignment'], conditioning=config['inputs'],
                   caveat='Body23 scores are recomputed for all models; do not directly compare to old 30-joint aggregates. Oracle head is privileged conditioning, not a fair baseline.')
    save(args.output / 'summary.json', summary)
    columns = [('mpjpe_m', 100, 'MPJPE cm'), ('pa_mpjpe_m', 100, 'PA-MPJPE cm'),
               ('root_err_m', 100, 'Root ADE cm'), ('root_fde_m', 100, 'Root FDE cm'),
               ('foot_skate_cm_s', 1, 'Skate cm/s'), ('foot_depth_mean_cm', 1, 'Foot penetration cm'),
               ('all_feet_above5cm_frame_fraction', 100, 'Airborne %'),
               ('accel_err_m_s2', 1, 'Accel error m/s2'), ('nonhead22_mpjpe_cm', 1, 'Nonhead MPJPE cm')]
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(['Model']+[c[2] for c in columns])
    for m in models:
        writer.writerow([m]+[f"{summary['aggregate'][m][k]['mean']*scale:.4f}" for k, scale, _ in columns])
    (args.output / 'comparison.csv').write_text(stream.getvalue())
    print(stream.getvalue(), flush=True)
    save(args.output / 'METRICS_COMPLETE.json', dict(cases=len(records)))
    render(args, config, decoded)
    save(args.output / 'COMPLETE.json', dict(cases=len(records), qualitative_cases=12,
                                            summary=str(args.output/'summary.json')))
    announce(args, 'complete', cases=len(records), gallery=str(args.output/'gallery/index.html'))


def render(args, config, decoded):
    import cv2
    from PIL import Image, ImageDraw
    from render_motion import render_single, write_mp4

    source_rows = read(args.phase3 / 'qualitative/manifest.json')
    assert len(source_rows) == 12
    gallery = args.output / 'gallery'
    gallery.mkdir(exist_ok=True)
    entries = []
    for row in source_rows:
        uid = row['id']
        target = gallery / (uid + '.mp4')
        if not target.exists():
            joints, rotation = decoded[uid]
            uem = render_single(joints, joint_parents=PARENTS, skip_joints=[],
                                caption='UniEgoMotion | camera -> body', color='tab:red',
                                width=512, height=512, head_rotation0=rotation)
            cap = cv2.VideoCapture(row['video'])
            assert cap.isOpened() and abs(cap.get(cv2.CAP_PROP_FPS)-20) < .01
            frames = []
            for i in range(97):
                ok, old = cap.read()
                assert ok and old.shape == (512, 1536, 3), 'Saved Phase3 panel layout changed'
                old = cv2.cvtColor(old, cv2.COLOR_BGR2RGB)
                canvas = Image.new('RGB', (2048, 552), (245, 245, 245))
                draw = ImageDraw.Draw(canvas)
                panels = [old[:, :512], uem[i], old[:, 512:1024], old[:, 1024:]]
                labels = ['GT reference', 'Camera observed: all 97 frames',
                          'Video observed: all 97 frames', 'Video + ORACLE GT head/global observed']
                for col, (panel, label) in enumerate(zip(panels, labels)):
                    canvas.paste(Image.fromarray(panel), (512*col, 40))
                    draw.text((512*col+10, 12), label, fill=(20, 135, 55))
                    draw.rectangle((512*col, 40, 512*(col+1)-1, 551),
                                   outline=(70, 120, 180) if col == 0 else (225, 155, 35), width=3)
                frames.append(np.asarray(canvas))
            assert not cap.read()[0], 'Unexpected extra saved Phase3 frames'
            cap.release()
            write_mp4(np.stack(frames), str(target), fps=20)
            Image.fromarray(np.concatenate([frames[i] for i in (0, 24, 48, 72, 96)], axis=0)).save(
                gallery / (uid + '.jpg'), quality=92)
        entries.append(dict(id=uid, video=str(target), reused_phase3_video=row['video'],
                            source_sha256=sha(row['video']), phase3_rerendered=False))
        save(gallery / 'manifest.json', entries)
        cards = ['<!doctype html><meta charset="utf-8"><title>UniEgoMotion / Phase3</title>',
                 '<h1>GT | UniEgoMotion | Phase3 V2M | Phase3 oracle head</h1>',
                 '<p>71 matched T97 windows; these are the 12 existing qualitative cases. '
                 'Green labels denote observed inputs; amber borders denote generated bodies. '
                 'Different conditioning, not equal-information inference. Metrics use body23; '
                 'reused Phase3 panels retain their original skeleton rendering. No grounding or alignment correction.</p>',
                 '<p><a href="../comparison.csv">Quantitative comparison (includes GT)</a></p>']
        for item in entries:
            name = html.escape(item['id'])
            cards.append(f'<h2>{name}</h2><video controls preload="none" style="width:100%" src="{name}.mp4"></video><p><a href="{name}.jpg">Contact sheet</a></p>')
        (gallery / 'index.html').write_text('\n'.join(cards))
        announce(args, 'rendering', complete=len(entries), total=12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['sample', 'report'])
    for name in ('output', 'phase3', 'cache', 'checkpoint'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    host = socket.gethostname().split('.')[0]
    if not os.environ.get('SLURM_JOB_ID') or not host.startswith('node') or host == 'node37':
        raise RuntimeError('Use an owned permitted compute allocation')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    try:
        (sample if args.mode == 'sample' else report)(args)
    except BaseException as error:
        announce(args, 'failed', error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
