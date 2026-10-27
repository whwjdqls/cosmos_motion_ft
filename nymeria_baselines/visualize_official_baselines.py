"""Matched, camera-conditioned qualitative samples without touching training.

Use native UEM sampling and the verbatim EgoAllo no-guidance sampler. The latter
is extracted by AST to avoid importing unused SMPL/JAX/Aria guidance packages;
only its trajectory unpack interface is adapted to 22 body joints/4 contacts.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

OPS = Path('/scratch2/whwjdqls99/cosmos/operations/nymeria_official_baselines_20261015')
CACHE = Path('/scratch2/whwjdqls99/cosmos/caches/nymeria_soma_official_baselines_T97_20fps_camhead_v1_20261015')
UEM_RUN = OPS / 'runs/uniegomotion_nymeria_soma_body230_b64_pinned_20261015'
EGO_RUN = OPS / 'upstream/egoallo/experiments/nymeria_soma_body_t97_shape16_floor_b256_pinned_20261015/v0'
Y_FROM_Z = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
PARENTS = [-1, 0, 1, 2, 3, 4, 5, 3, 7, 8, 9, 3, 11, 12, 13, 0, 15, 16, 17, 0, 19, 20, 21]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def decode(features, joints):
    from motion_expert_joint_attention.camera_head_recanonicalization import cont6d_to_matrix
    x = np.asarray(features, dtype=np.float64)
    local = x[:, :9*joints].reshape(-1, joints, 9)
    delta = x[:, 9*joints:9*joints+9]
    dr = cont6d_to_matrix(delta[:, :6])
    cr, ct = np.empty_like(dr), np.empty((len(x), 3))
    cr[0], ct[0] = dr[0], delta[0, 6:9]
    for t in range(1, len(x)):
        ct[t] = ct[t-1] + cr[t-1] @ delta[t, 6:9]
        cr[t] = cr[t-1] @ dr[t]
    return (ct[:, None] + np.einsum('tij,tkj->tki', cr, local[..., 6:9]),
            np.einsum('tij,tkjl->tkil', cr, cont6d_to_matrix(local[..., :6])))


def prepare(out):
    from dataset.nymeria_soma_dataset import NymeriaSOMADataset
    from egoallo.data.nymeria_soma import NymeriaSomaDataset
    from egoallo.transforms import SE3
    out.mkdir(parents=True, exist_ok=False)
    (out / 'inputs').mkdir()
    (out / 'snapshots').mkdir()
    ud, ed = NymeriaSOMADataset(str(CACHE), 'test'), NymeriaSomaDataset(CACHE, 'test')
    assert ud.rows == ed.rows
    first = {}
    for idx, row in enumerate(ud.rows):
        first.setdefault(row['uuid'], idx)
    names = sorted(first)
    chosen = [first[names[i]] for i in np.linspace(0, len(names)-1, 8, dtype=int)]
    manifest = json.loads((CACHE / 'index.json').read_text())
    train_arrays = sorted({r['array'] for r in manifest['rows'] if r['split'] == 'train'})
    # Equal-record weighting, sampling at 1 Hz. Never fit to test/validation poses.
    offsets, head_rotations_in_camera = [], []
    for key in train_arrays:
        a = np.load(CACHE / 'egoallo_metadata_v1' / f'{key}.joints_wrt_cpf.npy', mmap_mode='r')
        offsets.append(np.asarray(a[::20, 5], dtype=np.float64).mean(axis=0))
        motion = np.load(CACHE / 'records' / f'{key}.motion.npy', mmap_mode='r')
        camera = np.load(CACHE / 'records' / f'{key}.pose_yup.npy', mmap_mode='r')
        _, world_rotations = decode(motion[:1], 30)
        head_rotations_in_camera.append(camera[0, :3, :3].T @ world_rotations[0, 6])
    head_offset = np.mean(offsets, axis=0)
    assert np.isfinite(head_offset).all()
    ur, _, vr = np.linalg.svd(np.mean(head_rotations_in_camera, axis=0))
    head_rotation_in_camera = ur @ np.diag([1., 1., np.linalg.det(ur @ vr)]) @ vr
    calibration_error = float(np.max(np.abs(np.asarray(head_rotations_in_camera) - head_rotation_in_camera)))
    assert calibration_error < 2e-3, ('Camera/head rotation is not a fixed calibration', calibration_error)
    ck = max((p for p in UEM_RUN.glob('*.ckpt') if p.name != 'last.ckpt'), key=lambda p: p.stat().st_mtime_ns)
    shutil.copy2(ck, out / 'snapshots/uem.ckpt')
    # Copy only fully saved, nonzero native checkpoints, newest first.
    candidates = sorted(EGO_RUN.glob('checkpoints_*'), key=lambda p: int(p.name.rsplit('_', 1)[1]), reverse=True)
    saved_log = (OPS / 'egoallo_b256_pinned_train.log').read_text()
    ec = next(p for p in candidates if f'Saved checkpoint to {p}\n' in saved_log)
    shutil.copy2(ec / 'model.safetensors', out / 'snapshots/egoallo.safetensors')
    shutil.copy2(EGO_RUN / 'model_config.yaml', out / 'snapshots/egoallo_model_config.yaml')
    cases = []
    for number, idx in enumerate(chosen):
        u, e, row = ud[idx], ed[idx], ud.rows[idx]
        traj = u['y']['traj'].numpy()
        raw_camera = traj * ud.camera_std + ud.camera_mean
        cam_p, cam_r = decode(raw_camera, 1)
        cp = SE3(e.T_world_cpf).as_matrix().numpy()
        target_r = Y_FROM_Z @ cp[:, :3, :3]
        target_p = np.einsum('ij,tj->ti', Y_FROM_Z, cp[:, :3, 3])
        # Camera and motion each reset their own canonical frame at window start.
        # Check camera decoding independently; its frame is NOT the motion frame.
        camera_align_r = target_r[0] @ cam_r[0, 0].T
        camera_align_t = target_p[0] - camera_align_r @ cam_p[0, 0]
        cam_err = np.max(np.abs(np.einsum('ij,tj->ti', camera_align_r, cam_p[:, 0]) + camera_align_t - target_p))
        rot_err = np.max(np.abs(camera_align_r @ cam_r[:, 0] - target_r))
        assert cam_err < 2e-3 and rot_err < 2e-3, (row, cam_err, rot_err)
        from motion_expert_joint_attention.camera_head_recanonicalization import yaw_rotation_y
        head_r0 = target_r[0] @ head_rotation_in_camera
        forward = head_r0[:, 2]
        assert np.linalg.norm(forward[[0, 2]]) > 1e-2, ('Cannot recover head yaw from this camera input', row)
        align_r = yaw_rotation_y(np.asarray(np.arctan2(forward[0], forward[2])))
        predicted_head0 = target_p[0] + target_r[0] @ head_offset
        align_t = np.array([predicted_head0[0], 0., predicted_head0[2]])
        # Exact GT placement is for the GT panel/check only. It is not used to
        # align predictions; both models use the camera-only anchor above.
        other_gt = np.einsum('tij,tkj->tki', target_r, e.joints_wrt_cpf.numpy()) + target_p[:, None]
        gt_head0 = other_gt[0, 5]
        gt_align_t = np.array([gt_head0[0], 0., gt_head0[2]])
        raw = u['misc']['motion'].numpy()[:, :220] * ud.motion_std + ud.motion_mean
        gt, gt_r = decode(raw, 23)
        gt = np.einsum('ij,tkj->tki', align_r, gt) + gt_align_t
        gt_r = align_r @ gt_r
        # Independently verify body coordinates against EgoAllo's corrected metadata.
        gt_error = float(np.max(np.abs(gt[:, 1:] - other_gt)))
        assert gt_error < 2e-3, (row, gt_error)
        case = f'case_{number:02d}'
        np.savez(out / 'inputs' / f'{case}.npz', traj=traj,
                 motion_mean=ud.motion_mean, motion_std=ud.motion_std,
                 align_r=align_r, align_t=align_t, gt=gt, gt_r=gt_r,
                 camera_p=target_p, camera_r=target_r,
                 T_world_cpf=e.T_world_cpf.numpy(), T_cpf_tm1_cpf_t=e.T_cpf_tm1_cpf_t.numpy())
        cases.append(dict(id=case, dataset_index=idx, **row,
                          camera_alignment_max_m=float(cam_err), camera_rotation_max=float(rot_err),
                          cross_adapter_gt_max_m=gt_error, seed=62+number))
    config = dict(cases=cases, fps=20, frames=97,
                  cache=str(CACHE), manifest_sha256=sha(CACHE/'index.json'),
                  uem_checkpoint=str(ck), egoallo_checkpoint=str(ec),
                  uem_snapshot_sha256=sha(out/'snapshots/uem.ckpt'),
                  egoallo_snapshot_sha256=sha(out/'snapshots/egoallo.safetensors'),
                  uem_weights='state_dict, matching upstream vis_uem.py',
                  egoallo_head_offset_camera=head_offset.tolist(),
                  head_rotation_in_camera=head_rotation_in_camera.tolist(),
                  head_rotation_calibration_max_error=calibration_error,
                  prediction_anchor='Camera input plus train-only head extrinsics; no GT motion alignment',
                  head_offset_fit='Equal-record mean, 1 Hz samples from train recordings only',
                  head_offset_train_records=len(train_arrays),
                  egoallo_guidance='off; body-only SOMA, no SMPL/hand guidance',
                  floor='Existing per-window metadata; no prediction-dependent regrounding',
                  status='prepared')
    save_json(out / 'config.json', config)
    print('PREPARED', json.dumps(config), flush=True)


def sample_uem(out, device):
    from yacs.config import CfgNode
    from module.uem_module import UEM_Module
    config = json.loads((out/'config.json').read_text())
    ck = torch.load(out/'snapshots/uem.ckpt', map_location='cpu', weights_only=False)
    cfg = CfgNode(ck['hyper_parameters'])
    cfg.TRAIN.ONLY_VALIDATE = True
    model = UEM_Module(cfg)
    model.load_state_dict(ck['state_dict'], strict=True)
    model.to(device).eval()
    result = {'epoch': ck['epoch'], 'global_step': ck['global_step'],
              'diffusion_steps': cfg.MODEL.DIFFUSION_STEPS, 'cond_scale': cfg.TRAIN.COND_SCALE,
              'torch': torch.__version__, 'cases': []}
    del ck
    for case in config['cases']:
        torch.manual_seed(case['seed'])
        a = np.load(out/'inputs'/f"{case['id']}.npz")
        y = dict(traj=torch.as_tensor(a['traj'][None], device=device),
                 valid_frames=torch.ones((1, 97), dtype=torch.long, device=device))
        with torch.inference_mode():
            x = model.sample(y, B=1, cond_scale=cfg.TRAIN.COND_SCALE)[0].cpu().numpy()
        assert np.isfinite(x).all()
        raw = x[:, :220] * a['motion_std'] + a['motion_mean']
        joints, rotations = decode(raw, 23)
        joints = np.einsum('ij,tkj->tki', a['align_r'], joints) + a['align_t']
        rotations = a['align_r'] @ rotations
        np.savez(out/f"{case['id']}_uem.npz", prediction=x, joints=joints, rotations=rotations)
        result['cases'].append(case['id'])
        print('UEM sampled', case['id'], flush=True)
    save_json(out/'uem_result.json', result)


def native_ego_sampler():
    from egoallo import network
    from egoallo.tensor_dataclass import TensorDataclass
    from egoallo.transforms import SE3
    from tqdm.auto import tqdm
    class BodyTrajectory:
        @staticmethod
        def unpack(x, **kwargs):
            return network.EgoDenoiseTraj.unpack(x, **kwargs, body_joint_count=22, contact_joint_count=4)
    path = OPS/'upstream/egoallo/src/egoallo/sampling.py'
    tree = ast.parse(path.read_text())
    names = {'quadratic_ts', 'CosineNoiseScheduleConstants', 'run_sampling_with_stitching'}
    nodes = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == '__future__'
             or isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    namespace = dict(torch=torch, np=np, time=time, tqdm=tqdm, TensorDataclass=TensorDataclass,
                     SE3=SE3, network=SimpleNamespace(EgoDenoiseTraj=BodyTrajectory))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['run_sampling_with_stitching'], sha(path)


def sample_ego(out, device):
    import yaml
    from safetensors.torch import load_file
    from egoallo.network import EgoDenoiser
    from egoallo.transforms import SE3
    config = json.loads((out/'config.json').read_text())
    cfg = yaml.load((out/'snapshots/egoallo_model_config.yaml').read_text(), Loader=yaml.Loader)
    model = EgoDenoiser(cfg)
    model.load_state_dict(load_file(str(out/'snapshots/egoallo.safetensors')), strict=True)
    model.to(device).eval()
    sampler, source_sha = native_ego_sampler()
    shape = np.load(CACHE/'body_shape16.npz')
    result = dict(torch=torch.__version__, sampler_sha256=source_sha, cases=[])
    for case in config['cases']:
        torch.manual_seed(case['seed'])
        a = np.load(out/'inputs'/f"{case['id']}.npz")
        cpf = torch.as_tensor(a['T_world_cpf'], device=device)
        delta = torch.as_tensor(a['T_cpf_tm1_cpf_t'], device=device)
        previous = (SE3(cpf[0]) @ SE3(delta[0]).inverse()).parameters()
        transforms = torch.cat([previous[None], cpf], dim=0)
        pred = sampler(model, None, 'off', False, False, transforms, 0., None, None,
                       1, device, guidance_verbose=False)
        local_r = pred.body_rotmats[0].detach().cpu().numpy()
        betas = pred.betas[0].detach().cpu().numpy()
        assert np.isfinite(local_r).all() and np.isfinite(betas).all()
        neutral = (shape['mean'] + (betas * shape['scale']) @ shape['components']).reshape(97, 23, 3)
        neutral -= neutral[:, :1]
        fk_r = np.broadcast_to(np.eye(3), (97, 23, 3, 3)).copy()
        fk_p = np.zeros((97, 23, 3))
        for j in range(1, 23):
            p = PARENTS[j]
            fk_r[:, j] = fk_r[:, p] @ local_r[:, j-1]
            fk_p[:, j] = fk_p[:, p] + np.einsum('tij,tj->ti', fk_r[:, p], neutral[:, j]-neutral[:, p])
        world_root_r = a['camera_r'] @ np.asarray(config['head_rotation_in_camera']) @ np.swapaxes(fk_r[:, 6], -1, -2)
        head_p = a['camera_p'] + np.einsum('tij,j->ti', a['camera_r'], config['egoallo_head_offset_camera'])
        world_root_p = head_p - np.einsum('tij,tj->ti', world_root_r, fk_p[:, 6])
        joints = world_root_p[:, None] + np.einsum('tij,tkj->tki', world_root_r, fk_p)
        rotations = world_root_r[:, None] @ fk_r
        np.savez(out/f"{case['id']}_egoallo.npz", joints=joints, rotations=rotations,
                 betas=betas, local_rotations=local_r, contacts=pred.contacts[0].cpu().numpy())
        result['cases'].append(dict(id=case['id'], min_rotation_det=float(np.linalg.det(local_r).min())))
        print('EGO sampled', case['id'], flush=True)
    save_json(out/'egoallo_result.json', result)


def render(out):
    import html
    from motion_expert_joint_attention import render_motion as r
    config = json.loads((out/'config.json').read_text())
    gallery = out/'gallery'
    gallery.mkdir(exist_ok=True)
    entries = []
    for case in config['cases']:
        a = np.load(out/'inputs'/f"{case['id']}.npz")
        u = np.load(out/f"{case['id']}_uem.npz")
        e = np.load(out/f"{case['id']}_egoallo.npz")
        arrays = [a['gt'], u['joints'], e['joints']]
        rotations = [a['gt_r'][0, 6], u['rotations'][0, 6], e['rotations'][0, 6]]
        displayed = [r._to_display(x) for x in arrays]
        camera = r._to_display(a['camera_p'])
        extent = r._world_extent(displayed, extra_pts=[camera])
        fig = r.plt.figure(figsize=(15, 5), dpi=100)
        frames = []
        try:
            for t in range(97):
                fig.clf()
                for panel, (joints, rotation, title, color) in enumerate(zip(
                    displayed, rotations, ['Ground truth', 'UniEgoMotion', 'EgoAllo'],
                    ['tab:blue', 'tab:red', 'tab:orange']), 1):
                    ax = fig.add_subplot(1, 3, panel, projection='3d')
                    r._setup_axes(ax, r._viewport_for_center((camera[t, 0], camera[t, 2]),
                                  r.DEFAULT_VIEW_HALF, extent['y_top'], extent['y_bottom']), title)
                    r._draw_floor_and_grid(ax, extent)
                    r._draw_trajectory(ax, joints[:, 0], t, color)
                    r._draw_skel(ax, joints[t], PARENTS, color, skip_joints=[])
                    r._draw_head_reference(ax, joints[0], rotation)
                    ax.plot(camera[:, 0], camera[:, 2], camera[:, 1], color='forestgreen', linewidth=1., alpha=.6)
                    ax.scatter(camera[t, 0], camera[t, 2], camera[t, 1], color='forestgreen', s=15, depthshade=False)
                fig.suptitle('Camera-only conditioning | green: observed camera | axes: first-frame head', fontsize=10)
                fig.subplots_adjust(left=.01, right=.99, top=.90, bottom=.02, wspace=.03)
                fig.canvas.draw()
                frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            r.write_mp4(np.stack(frames), str(gallery/f"{case['id']}.mp4"), fps=20)
            import imageio.v3 as iio
            iio.imwrite(gallery/f"{case['id']}.png", frames[0])
        finally:
            r.plt.close(fig)
        entries.append(f'<h2>{html.escape(case["uuid"])}</h2><p>{html.escape(case["caption"])}</p>'
                       f'<video controls loop src="{case["id"]}.mp4" poster="{case["id"]}.png" style="width:100%"></video>')
        print('RENDERED', case['id'], flush=True)
    (gallery/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Nymeria baseline comparison</title>'
                                    '<h1>GT / UniEgoMotion / EgoAllo</h1><p>Eight deterministic held-out cases. '
                                    'Predicted motions; all camera frames observed. No prediction-dependent floor adjustment. '
                                    'EgoAllo uses a training-only fitted SOMA camera/head offset.</p>'+''.join(entries))
    save_json(out/'complete.json', dict(status='complete', cases=len(entries), gallery=str(gallery)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'uem', 'egoallo', 'render'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.mode == 'prepare':
        prepare(args.output)
    elif args.mode == 'uem':
        sample_uem(args.output, torch.device(args.device))
    elif args.mode == 'egoallo':
        sample_ego(args.output, torch.device(args.device))
    else:
        render(args.output)
