"""Evaluate body-only EgoAllo on the saved 71-case UEM/Phase3 comparison."""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nymeria_baselines.visualize_official_baselines import (
    PARENTS, Y_FROM_Z, decode, sha,
)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def calibrate(cache):
    """Fit the SOMA camera/head extrinsics using training recordings only."""
    rows = read(cache / 'index.json')['rows']
    keys = sorted({r['array'] for r in rows if r['split'] == 'train'})
    offsets, rotations = [], []
    for key in keys:
        joints = np.load(cache / 'egoallo_metadata_v1' / f'{key}.joints_wrt_cpf.npy', mmap_mode='r')
        offsets.append(np.asarray(joints[::20, 5], dtype=np.float64).mean(axis=0))
        motion = np.load(cache / 'records' / f'{key}.motion.npy', mmap_mode='r')
        camera = np.load(cache / 'records' / f'{key}.pose_yup.npy', mmap_mode='r')
        _, world_r = decode(motion[:1], 30)
        rotations.append(camera[0, :3, :3].T @ world_r[0, 6])
    u, _, vt = np.linalg.svd(np.mean(rotations, axis=0))
    rotation = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    error = float(np.max(np.abs(np.asarray(rotations) - rotation)))
    assert error < 2e-3, ('Nonconstant camera/head rotation', error)
    return dict(head_offset_camera=np.mean(offsets, axis=0).tolist(),
                head_rotation_in_camera=rotation.tolist(), train_recordings=len(keys),
                rotation_calibration_max_error=error,
                policy='Train-only equal-record mean at 1 Hz; no test shape or pose fitting')


def infer(args):
    import torch
    import yaml
    from safetensors.torch import load_file
    from egoallo.data.nymeria_soma import NymeriaSomaDataset
    from egoallo.network import EgoDenoiser
    from egoallo.transforms import SE3
    from motion_expert_joint_attention.camera_head_recanonicalization import yaw_rotation_y
    from nymeria_baselines.visualize_official_baselines import native_ego_sampler

    reference = read(args.reference / 'configuration.json')
    cache = Path(reference['cache'])
    assert len(reference['cases']) == 71
    assert sha(cache / 'index.json') == reference['cache_index_sha256']
    dataset = NymeriaSomaDataset(cache, 'test')
    lookup = {(r['uuid'], r['start']): i for i, r in enumerate(dataset.rows)}
    assert all((c['uuid'], c['start']) in lookup for c in reference['cases'])
    model_config = args.checkpoint.parent / 'model_config.yaml'
    cfg = yaml.load(model_config.read_text(), Loader=yaml.Loader)
    assert not cfg.include_hands and cfg.body_joint_count == 22 and cfg.contact_joint_count == 4
    model = EgoDenoiser(cfg)
    model.load_state_dict(load_file(str(args.checkpoint / 'model.safetensors')), strict=True)
    device = torch.device('cuda:0')
    model.to(device).eval()
    sampler, sampler_sha = native_ego_sampler()
    calibration = calibrate(cache)
    extrinsic_r = np.asarray(calibration['head_rotation_in_camera'])
    extrinsic_p = np.asarray(calibration['head_offset_camera'])
    config = dict(cases=reference['cases'], checkpoint=str(args.checkpoint),
                  checkpoint_sha256=sha(args.checkpoint / 'model.safetensors'),
                  model_config_sha256=sha(model_config),
                  step=int(args.checkpoint.name.rsplit('_', 1)[1]),
                  reference=str(args.reference), reference_config_sha256=sha(args.reference / 'configuration.json'),
                  cache=str(cache), cache_index_sha256=sha(cache / 'index.json'),
                  shape_basis_sha256=sha(cache / 'body_shape16.npz'), calibration=calibration,
                  sampler='Native run_sampling_with_stitching; unchanged function body; body22/contact4 unpack adapter',
                  sampler_sha256=sampler_sha, guidance='off', hands=False,
                  frames=97, fps=20, body_joints=reference['body_joints'],
                  conditioning='All 97 measured camera poses; no RGB, text, GT shape, or GT motion',
                  shape='Predicted 16D SOMA shape PCA, decoded with the training basis',
                  alignment='Camera plus train-only head extrinsics supplies world placement and frame-0 head-yaw origin; no GT fit or prediction grounding',
                  caveat='Camera-conditioned reconstruction versus video-conditioned Phase3; oracle-head Phase3 is privileged conditioning')
    args.output.mkdir(parents=True, exist_ok=True)
    predictions = args.output / 'predictions'
    predictions.mkdir(exist_ok=True)
    if (args.output / 'configuration.json').exists():
        assert read(args.output / 'configuration.json') == config, 'Refusing incompatible resume'
    save(args.output / 'configuration.json', config)
    shape = np.load(cache / 'body_shape16.npz')
    audits = []
    print('INFERENCE_START', json.dumps(dict(step=config['step'], cases=71)), flush=True)
    for number, case in enumerate(config['cases']):
        output = predictions / (case['id'] + '.npz')
        row_index = lookup[(case['uuid'], case['start'])]
        row = dataset.rows[row_index]
        assert abs(float(row['ground_offset_y']) - float(case['floor_offset'])) < 1e-5
        data = dataset[row_index]
        camera = SE3(data.T_world_cpf).as_matrix().numpy()
        camera_r = Y_FROM_Z @ camera[:, :3, :3]
        camera_p = np.einsum('ij,tj->ti', Y_FROM_Z, camera[:, :3, 3])
        forward = (camera_r[0] @ extrinsic_r)[:, 2]
        assert np.linalg.norm(forward[[0, 2]]) > 1e-2
        yaw = yaw_rotation_y(np.asarray(np.arctan2(forward[0], forward[2])))
        head_p = camera_p + np.einsum('tij,j->ti', camera_r, extrinsic_p)
        origin = np.array([head_p[0, 0], 0., head_p[0, 2]])
        with np.load(args.reference / 'predictions' / (case['id'] + '.npz')) as previous:
            target, target_r = decode(previous['gt_body'], 23)
        # Audit the independently prepared GT only; never use this GT anchor
        # to place the prediction. Camera-derived origin above is independent.
        gt_world = camera_p[:, None] + np.einsum('tij,tkj->tki', camera_r, data.joints_wrt_cpf.numpy())
        gt_origin = np.array([gt_world[0, 5, 0], 0., gt_world[0, 5, 2]])
        gt_canonical = np.einsum('ij,tkj->tki', yaw.T, gt_world - gt_origin)
        error = float(np.max(np.abs(gt_canonical - target[:, 1:])))
        assert error < 2e-3, (case['id'], 'GT frame mismatch', error)
        audits.append(dict(id=case['id'], cross_adapter_gt_max_m=error))
        if not output.exists():
            torch.manual_seed(case['seed'])
            np.random.seed(case['seed'])
            cpf = data.T_world_cpf.to(device)
            delta = data.T_cpf_tm1_cpf_t.to(device)
            previous_pose = (SE3(cpf[0]) @ SE3(delta[0]).inverse()).parameters()
            transforms = torch.cat([previous_pose[None], cpf], dim=0)
            with torch.inference_mode():
                pred = sampler(model, None, 'off', False, False, transforms, 0., None, None,
                               1, device, guidance_verbose=False)
            local_r = pred.body_rotmats[0].detach().cpu().numpy()
            betas = pred.betas[0].detach().cpu().numpy()
            assert local_r.shape == (97, 22, 3, 3) and betas.shape == (97, 16)
            assert np.isfinite(local_r).all() and np.isfinite(betas).all()
            neutral = (shape['mean'] + (betas * shape['scale']) @ shape['components']).reshape(97, 23, 3)
            neutral = neutral - neutral[:, :1]
            fk_r = np.broadcast_to(np.eye(3), (97, 23, 3, 3)).copy()
            fk_p = np.zeros((97, 23, 3))
            for j in range(1, 23):
                parent = PARENTS[j]
                fk_r[:, j] = fk_r[:, parent] @ local_r[:, j-1]
                fk_p[:, j] = fk_p[:, parent] + np.einsum('tij,tj->ti', fk_r[:, parent], neutral[:, j]-neutral[:, parent])
            root_r = camera_r @ extrinsic_r @ np.swapaxes(fk_r[:, 6], -1, -2)
            root_p = head_p - np.einsum('tij,tj->ti', root_r, fk_p[:, 6])
            world = root_p[:, None] + np.einsum('tij,tkj->tki', root_r, fk_p)
            joints = np.einsum('ij,tkj->tki', yaw.T, world-origin).astype(np.float32)
            rotations = yaw.T @ root_r[:, None] @ fk_r
            temporary = output.with_suffix('.tmp.npz')
            np.savez(temporary, joints=joints, rotations=rotations, gt=target.astype(np.float32),
                     gt_rotations=target_r, betas=betas, local_rotations=local_r,
                     contacts=pred.contacts[0].cpu().numpy(), seed=case['seed'],
                     camera_p=np.einsum('ij,tj->ti', yaw.T, camera_p-origin))
            temporary.replace(output)
        save(args.output / 'progress.json', dict(stage='inference', completed=number+1, total=71, last=case['id']))
        print('SAMPLED', number+1, '/71', case['id'], flush=True)
    save(args.output / 'input_audit.json', audits)
    save(args.output / 'INFERENCE_COMPLETE.json', dict(cases=71, step=config['step']))


def report(args):
    import torch
    from decode_uniego_torch import decode_transforms
    from kimodo.metrics.foot_skate import FootSkateFromHeight
    from nymeria_baselines.evaluate_uem_phase3 import measure, CONTACT_JOINTS
    config = read(args.output / 'configuration.json')
    assert sha(args.reference / 'configuration.json') == config['reference_config_sha256']
    assert read(args.output / 'INFERENCE_COMPLETE.json')['cases'] == 71
    reference = read(args.reference / 'configuration.json')
    contract = reference['phase3_contract']
    assert sha(contract['motion_stats_mean']) == contract['motion_mean']
    assert sha(contract['motion_stats_std']) == contract['motion_std']
    mean = np.load(contract['motion_stats_mean'])
    std = np.load(contract['motion_stats_std'])
    cases = {case['id']: case for case in reference['cases']}
    rows = read(args.reference / 'per_case.json')
    assert [r['id'] for r in rows] == [c['id'] for c in config['cases']]
    skate = FootSkateFromHeight(SimpleNamespace(foot_joint_idx=list(CONTACT_JOINTS)), fps=20)
    for row in rows:
        with np.load(cases[row['id']]['bundle']) as bundle:
            gt_norm = bundle['motion'].copy()
        gt_tf = np.take(decode_transforms(torch.from_numpy(gt_norm*std+mean)[None])[0].numpy(),
                        reference['body_joints'], axis=1)
        gt = gt_tf[..., :3, 3]
        with np.load(args.output / 'predictions' / (row['id'] + '.npz')) as a:
            gt_metrics = measure(gt, gt, skate)
            for key in ('foot_skate_cm_s', 'foot_depth_mean_cm', 'all_feet_above5cm_frame_fraction'):
                assert abs(gt_metrics[key] - row['metrics']['GT'][key]) < 1e-3, (row['id'], key, gt_metrics[key], row['metrics']['GT'][key])
            row['metrics']['EgoAllo'] = measure(a['joints'], gt, skate)
    save(args.output / 'metric_protocol.json', dict(
        gt_source='Exact saved Phase3 bundles, normalization and Torch decoder from the original comparison',
        decoder='decode_uniego_torch.decode_transforms',
        mean_path=contract['motion_stats_mean'], mean_sha256=contract['motion_mean'],
        std_path=contract['motion_stats_std'], std_sha256=contract['motion_std'],
        reference_config_sha256=config['reference_config_sha256'],
        original_gt_physical_metrics_checked=True, prediction_files_unchanged=True))
    summary = read(args.reference / 'summary.json')
    summary['aggregate']['EgoAllo'] = {}
    for key in rows[0]['metrics']['EgoAllo']:
        values = np.asarray([r['metrics']['EgoAllo'][key] for r in rows], dtype=np.float64)
        assert np.isfinite(values).all(), key
        summary['aggregate']['EgoAllo'][key] = dict(mean=float(values.mean()), median=float(np.median(values)),
                                                  std=float(values.std()), min=float(values.min()), max=float(values.max()))
    summary['conditioning']['EgoAllo'] = config['conditioning']
    summary['egoallo_step'] = config['step']
    summary['egoallo_alignment'] = config['alignment']
    save(args.output / 'per_case.json', rows)
    save(args.output / 'summary.json', summary)
    columns = [('MPJPE_cm', 'mpjpe_m', 100), ('PA_MPJPE_cm', 'pa_mpjpe_m', 100),
               ('Root_ADE_cm', 'root_err_m', 100), ('Root_FDE_cm', 'root_fde_m', 100),
               ('Foot_skate_cm_s', 'foot_skate_cm_s', 1), ('Foot_penetration_cm', 'foot_depth_mean_cm', 1),
               ('Airborne_pct', 'all_feet_above5cm_frame_fraction', 100), ('Accel_error_m_s2', 'accel_err_m_s2', 1)]
    with (args.output / 'comparison.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Model'] + [x[0] for x in columns])
        for model in ('GT', 'UniEgoMotion', 'EgoAllo', 'Phase3_V2M', 'Phase3_oracle_head'):
            values = [summary['aggregate'][model][key]['mean'] * scale for _, key, scale in columns]
            writer.writerow([model] + values)
            print('METRICS', model, dict(zip([x[0] for x in columns], values)), flush=True)
    save(args.output / 'METRICS_COMPLETE.json', dict(cases=71))
    render(args, config)


def render(args, config):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    from motion_expert_joint_attention import render_motion as r
    gallery = args.output / 'gallery'
    gallery.mkdir(exist_ok=True)
    qualitative = read(Path(read(args.reference / 'configuration.json')['phase3_root']) / 'qualitative/manifest.json')
    entries = []
    for case in qualitative:
        identifier = case['id']
        reader = imageio.get_reader(str(args.reference / 'gallery' / (identifier + '.mp4')))
        width, height = reader.get_meta_data()['size']
        assert width % 4 == 0
        panel_width, header = width // 4, 40
        with np.load(args.output / 'predictions' / (identifier + '.npz')) as a:
            joints, gt = r._to_display(a['joints']), r._to_display(a['gt'])
            rotation = a['rotations'][0, 6]
        extent = r._world_extent([gt, joints])
        fig = r.plt.figure(figsize=(panel_width/100, (height-header)/100), dpi=100)
        frames, previews = [], []
        try:
            for t, previous in enumerate(reader):
                assert t < 97
                fig.clf()
                ax = fig.add_subplot(1, 1, 1, projection='3d')
                r._setup_axes(ax, r._viewport_for_center((gt[t, 0, 0], gt[t, 0, 2]),
                              r.DEFAULT_VIEW_HALF, extent['y_top'], extent['y_bottom']),
                              f'EgoAllo {config["step"]//1000}k | camera -> body')
                r._draw_floor_and_grid(ax, extent)
                r._draw_trajectory(ax, joints[:, 0], t, 'tab:purple')
                r._draw_skel(ax, joints[t], PARENTS, 'tab:purple', skip_joints=[])
                r._draw_head_reference(ax, joints[0], rotation)
                fig.subplots_adjust(left=.01, right=.99, bottom=.04, top=.92)
                fig.canvas.draw()
                canvas = np.asarray(fig.canvas.buffer_rgba())[..., :3]
                right = Image.new('RGB', (panel_width, height), '#f5f5f5')
                right.paste(Image.fromarray(canvas), (0, header))
                draw = ImageDraw.Draw(right)
                draw.text((8, 10), 'Camera observed all 97 frames', fill='#238b45')
                draw.rectangle((1, header, panel_width-2, height-2), outline='#d99a22', width=3)
                combined = np.concatenate([previous, np.asarray(right)], axis=1)
                frames.append(combined)
                if t in (0, 24, 48, 72, 96):
                    previews.append(combined)
            assert len(frames) == 97
            r.write_mp4(np.stack(frames), str(gallery / (identifier + '.mp4')), fps=20)
            imageio.imwrite(gallery / (identifier + '.jpg'), np.concatenate(previews, axis=0))
        finally:
            reader.close()
            r.plt.close(fig)
        entries.append(dict(id=identifier, caption=case['caption'], video=str(gallery / (identifier+'.mp4'))))
        print('RENDERED', len(entries), '/12', identifier, flush=True)
    blocks = [f'<h2>{html.escape(x["id"])}</h2><p>{html.escape(x["caption"])}</p>'
              f'<video controls loop preload="metadata" src="{x["id"]}.mp4" style="width:100%"></video>' for x in entries]
    (gallery / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>EgoAllo / UEM / Phase3</title>'
        '<h1>GT / UniEgoMotion-traj / Phase3 V2M / Phase3 oracle-head / EgoAllo</h1>'
        '<p>97 frames at 20 fps. Green labels: observed inputs. Amber borders: generated motion. '
        'The first four panels reuse the saved comparison. Camera and video conditioning are not equal-information inputs. '
        'EgoAllo uses predicted SOMA shape and training-only head calibration, without GT alignment or floor correction.</p>'
        + ''.join(blocks))
    save(args.output / 'qualitative.json', entries)
    save(args.output / 'COMPLETE.json', dict(cases=71, videos=len(entries), step=config['step'], gallery=str(gallery)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('infer', 'report'))
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.uname().nodename.startswith('node'):
        raise RuntimeError('Run on a compute allocation, never the login node')
    import torch
    torch.set_num_threads(4)
    (infer if args.stage == 'infer' else report)(args)


if __name__ == '__main__':
    main()
