"""Native coupled sampling with camera velocity output and no GT camera input."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motion_expert_phase3_edge_joint.settings import add_paths, atomic_json, require_compute_node
from .model import EdgePhase3
from .policy import CAMERA_POLICY, configure
from motion_expert_phase3_edge_image_joint.policy import captions_for_modes


@torch.no_grad()
def sample(model, *, mode, caption="", motion, neutral, video, seed=0, steps=35, guidance=1.0):
    from cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler
    if mode not in {"video2motion", "motimg2video", "joint"} or guidance != 1.0:
        raise ValueError("This camera-prediction ablation uses the three native scale1 tasks")
    caption = captions_for_modes([caption], [mode])[0]
    device = motion.device

    def scheduler(shift):
        result = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1., use_dynamic_shifting=False)
        result.set_timesteps(steps, device=device, shift=shift)
        return result

    ms, vs = scheduler(3.), scheduler(10.)
    cs = scheduler(3. if mode == "video2motion" else 10.)
    generator = torch.Generator(device=device).manual_seed(seed)
    # Keep the baseline's motion/video random draw order exactly. Camera uses
    # a private generator and never consumes a baseline diffusion RNG draw.
    state_m = torch.randn(motion.shape, device=device, dtype=torch.float32, generator=generator) if mode != 'motimg2video' else motion
    state_v = torch.randn(video.shape, device=device, dtype=torch.float32, generator=generator) if mode != 'video2motion' else video
    camera_generator = torch.Generator(device=device).manual_seed(131071 + int(seed))
    state_c = torch.randn((motion.shape[0], 96, 64), device=device, dtype=torch.float32, generator=camera_generator)
    if mode != 'video2motion':
        state_v[:, :, 0] = video[:, :, 0]
    model.eval()
    for index in range(steps):
        motion_sigma = ms.timesteps[index].float().reshape(1)/1000. if mode != 'motimg2video' else torch.zeros(1, device=device)
        video_sigma = vs.timesteps[index].float().reshape(1)/1000. if mode != 'video2motion' else torch.zeros(1, device=device)
        camera_sigma = cs.timesteps[index].float().reshape(1)/1000.
        output = model(motion=state_m, neutral=neutral, video=state_v, motion_sigma=motion_sigma,
                       video_sigma=video_sigma, captions=[caption], modes=[mode],
                       camera_state=state_c, camera_sigma=camera_sigma)
        # All predictions see the same pre-step states before any state advances.
        if mode != 'motimg2video':
            sigma = ms.sigmas[index].to(device=device, dtype=torch.float32).clamp_min(1e-6)
            velocity = (state_m-output['motion'].float())/sigma
            state_m = ms.step(velocity, ms.timesteps[index], state_m, return_dict=False, generator=generator)[0].float()
        if mode != 'video2motion':
            state_v = vs.step(output['video'].float(), vs.timesteps[index], state_v, return_dict=False, generator=generator)[0].float()
            state_v[:, :, 0] = video[:, :, 0]
        state_c = cs.step(output['camera_velocity'].float(), cs.timesteps[index], state_c,
                          return_dict=False, generator=camera_generator)[0].float()
    result = dict(motion=state_m, video=state_v, camera=state_c)
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("Nonfinite motion/video/camera sample")
    return result


def main():
    configure()
    parser = add_paths(argparse.ArgumentParser())
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--mode', choices=['video2motion','motimg2video','joint'], required=True)
    parser.add_argument('--steps', type=int, default=35)
    args = parser.parse_args()
    require_compute_node()
    if not args.out.resolve().is_relative_to('/mnt/local'):
        raise ValueError('Inference output must be on node-local NVMe')
    payload = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if payload['scheduler']['recipe'].get('camera_prediction') != CAMERA_POLICY:
        raise ValueError('Not this camera-prediction checkpoint')
    model = EdgePhase3(args.phase1,args.phase2,device='cuda',activation_checkpointing=False)
    if payload['sources']['phase1_metadata_sha256'] != model.provenance['phase1_metadata_sha256'] or payload['sources']['phase2_contract'] != model.provenance['phase2_contract']:
        raise ValueError('Checkpoint specialist identity mismatch')
    model.bridges.load_state_dict(payload['bridges'],strict=True)
    model.requires_grad_(False).eval()
    args.out.mkdir(parents=True,exist_ok=True)
    records = json.loads(args.cases.read_text())
    for row in records:
        with np.load(row['bundle'],allow_pickle=False) as bundle:
            contract=model.provenance['phase2_contract']
            if str(bundle['motion_mean_sha256']) != contract['motion_mean'] or str(bundle['motion_std_sha256']) != contract['motion_std']:
                raise ValueError('Bundle normalization mismatch')
            values={k:torch.from_numpy(bundle[k].copy())[None].cuda().float() for k in ['motion','neutral','video']}
        with torch.autocast('cuda',dtype=torch.bfloat16):
            result=sample(model,mode=args.mode,caption=row['caption'],**values,seed=int(row['seed']),steps=args.steps)
        target=args.out/(Path(row['id']).name+'.npz')
        if target.exists():
            raise FileExistsError(target)
        np.savez(target,**{k:v[0].cpu().numpy() for k,v in result.items()},seed=row['seed'])
        atomic_json(args.out/'manifest.json',dict(checkpoint=str(args.checkpoint),step=payload['step'],
                    mode=args.mode,steps=args.steps,camera_input='pure_noise_no_GT_camera_at_inference',
                    camera_prediction=CAMERA_POLICY))


if __name__=='__main__':
    main()
