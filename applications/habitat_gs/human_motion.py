"""Generated RGB -> evaluated caption-conditioned baseline Phase3 -> replay data."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch
from .camera import sha256
from .human_geometry import map_motion_to_scene
from .rollout import atomic_json, validate_video
from .runtime import require_compute


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--start', type=Path, required=True)
    p.add_argument('--prompt', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    require_compute()
    cfg = json.loads(args.config.read_text())
    reference = json.loads(Path(cfg['motion_reference']).read_text())
    checkpoint = Path(reference['checkpoint'])
    if sha256(checkpoint) != reference['checkpoint_sha256']:
        raise ValueError('Evaluation checkpoint hash changed')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    from motion_expert_phase3_edge.visualize import configure, load_vae, make_pipeline, encode_window
    contract = configure(payload)
    frames = np.asarray(validate_video(args.video))
    np.save(args.out / 'generated_rgb.npy', frames)
    # Actual encoded RGB matches the evaluated preparation, rather than
    # assuming that a generation pipeline's private latents are equivalent.
    vae = load_vae(Path(os.environ['WAN_VAE_PATH']), '256', 97, 'cuda')
    latents, image_size = encode_window(vae, make_pipeline(), '256', frames, 'cuda')
    np.save(args.out / 'video_latents.npy', latents)
    del vae
    torch.cuda.empty_cache()
    from motion_expert_phase3_edge.model import EdgePhase3
    from motion_expert_phase3_edge.sample import sample
    model = EdgePhase3(payload['sources']['phase1'], payload['sources']['phase2'],
                       device='cuda', activation_checkpointing=False)
    if (model.provenance['phase2_contract'] != contract or
            model.provenance['phase1_metadata_sha256'] != payload['sources']['phase1_metadata_sha256']):
        raise ValueError('Frozen specialist provenance differs from evaluation')
    model.bridges.load_state_dict(payload['bridges'], strict=True)
    # Reuse the exact text intervention used by the saved evaluation, including
    # both reasoners' packing. The native 35-step sampler is unmodified.
    intervention = Path(cfg['caption_implementation'])
    spec = importlib.util.spec_from_file_location('habitat_evaluated_caption', intervention)
    captions = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(captions)
    shape = np.load(cfg['neutral_shape']).astype(np.float32)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        with captions.caption_conditioning(model, args.prompt) as calls:
            prediction = sample(model, mode='video2motion', caption=args.prompt,
                motion=torch.zeros((1, 97, 283), device='cuda'),
                neutral=torch.from_numpy(shape)[None].cuda(),
                video=torch.from_numpy(latents.astype(np.float32))[None].cuda(),
                seed=args.seed, steps=reference['steps'], guidance=reference['guidance'])
    if calls != {'motion_reasoner': 35, 'video_reasoner': 35}:
        raise ValueError(f'Caption conditioning was not applied at all steps: {calls}')
    normalized = prediction[0].float().cpu().numpy()
    if normalized.shape != (97, 283) or not np.isfinite(normalized).all():
        raise ValueError('Invalid sampled motion')
    mean, std = (np.load(contract[k]).astype(np.float32) for k in ('motion_stats_mean', 'motion_stats_std'))
    raw = normalized * std + mean
    from motion_expert_joint_attention.decode_uniego_torch import decode_transforms
    decoded = decode_transforms(torch.from_numpy(raw)[None])[0].numpy()
    calibration = json.loads(Path(cfg['head_calibration']).read_text())
    if calibration['split'] != 'train':
        raise ValueError('Replay requires train-only head/camera calibration')
    head_camera = np.eye(4)
    head_camera[:3, :3] = calibration['rotation_head_to_upright_camera']
    head_camera[:3, 3] = calibration['camera_origin_in_head_m']
    start = np.asarray(json.loads(args.start.read_text()))
    mapped = map_motion_to_scene(decoded, start, head_camera, cfg['scene_units_per_meter'])
    np.savez_compressed(args.out / 'motion_arrays.npz', features_normalized=normalized,
        features_unnormalized=raw, decoded_joint_transforms=decoded, neutral_joints=shape,
        joint_poses_scene=mapped['joint_poses_scene'], joints_scene=mapped['joints_scene'],
        scene_from_decoded=mapped['scene_from_decoded'], head_to_camera_meters=head_camera,
        frame_times_local_s=np.arange(97) / 20)
    np.savez(args.out / 'camera_trajectory.npz', camera_to_world_optical=mapped['camera_to_world_optical'],
             frame_times_s=np.arange(97) / 20)
    # Web clients can replay directly without NumPy or a model installation.
    atomic_json(args.out / 'replay.json', dict(schema='habitat_human_replay_v1', fps=20, frames=97,
        prompt=args.prompt, seed=args.seed, units='scene units', matrix_layout='row-major; column-vector transforms',
        camera_to_world_optical=mapped['camera_to_world_optical'].tolist(),
        joint_poses_scene=mapped['joint_poses_scene'].tolist(), joints_scene=mapped['joints_scene'].tolist(),
        scene_from_decoded=mapped['scene_from_decoded'].tolist(), head_to_camera_meters=head_camera.tolist(),
        frame_times_local_s=(np.arange(97) / 20).tolist(), skeleton='../assets/skeleton.json'))
    atomic_json(args.out / 'end.pose.json', mapped['camera_to_world_optical'][-1].tolist())
    for key in ('motion_stats_mean', 'motion_stats_std'):
        shutil.copy2(contract[key], args.out / (key + '.npy'))
    atomic_json(args.out / 'inference.json', dict(checkpoint=str(checkpoint), checkpoint_sha256=reference['checkpoint_sha256'],
        checkpoint_step=int(payload['step']), sources=payload['sources'], sampler='official UniPC',
        steps=reference['steps'], guidance=reference['guidance'], shift=reference['shift'], seed=args.seed,
        prompt=args.prompt, caption_paths=calls, caption_implementation_sha256=sha256(intervention),
        observed_video=str(args.video), observed_video_sha256=sha256(args.video),
        decoded_joints='exact evaluated torch UniEgo decoder, no grounding or smoothing',
        neutral_shape_sha256=sha256(cfg['neutral_shape']),
        camera_mapping='H_scene(t) = A H_decoded(t); C_scene(t) = H_scene(t) X_train; A = C_start inv(H_decoded(0) X_train)',
        rotation_projection_max_abs=mapped['rotation_projection_max_abs'],
        scene_alignment_world_up_tilt_deg=mapped['world_up_tilt_deg'],
        head_calibration_sha256=sha256(cfg['head_calibration']), camera_is_phase1_inverse_dynamics=False,
        implementation_sha256={p.name:sha256(p) for p in (Path(__file__), Path(__file__).with_name('human_geometry.py'))},
        caveat='Caption-conditioned V2M is the saved inference-only evaluation intervention; no GT motion is supplied.'))


if __name__ == '__main__':
    main()
