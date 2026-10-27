"""Held-out offline Phase-3 samples. Does not change training or model code."""
from __future__ import annotations

import argparse
import gc
import html
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

def make_pipeline():
    from cosmos_framework.data.generator.action.utils.transforms import VideoResize

    # Only the spatial stage is needed. Edge's action-task planner does not
    # accept Nano's image2video mode, and must not rewrite Phase-3 conditioning.
    return VideoResize(pad_keys=['video'], keep_aspect_ratio=True)


def load_vae(vae_path, resolution, num_frames, device):
    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import Wan2pt2VAEInterface

    if device != 'cuda' or str(resolution) != '256' or num_frames != 97:
        raise ValueError('This Phase-3 recipe requires CUDA, resolution tier 256 and T97')
    return Wan2pt2VAEInterface(
        bucket_name='', vae_path=str(vae_path), chunk_duration=93,
        keep_decoder_cache=False, use_streaming_encode=False,
        encode_chunk_frames={'256': 68, '480': 24, '720': 12},
        encode_exact_durations=[97], spatial_compression_factor=16,
        temporal_compression_factor=4,
    )


@torch.no_grad()
def encode_window(vae, pipe, resolution, frames, device):
    # Same resize/pad, pixel normalization and latent crop as the training cache.
    transformed = pipe(dict(video=torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous(),
                            mode='image2video', ai_caption=''), resolution)
    pixels = transformed['video'][None].to(device=device, dtype=torch.float32) / 127.5 - 1.0
    latent = vae.encode(pixels)[0]
    image_size = transformed['image_size'].float()
    h = max(int(image_size[2].item()) // vae.spatial_compression_factor, 1)
    w = max(int(image_size[3].item()) // vae.spatial_compression_factor, 1)
    return latent[:, :, :h, :w].contiguous().half().cpu().numpy(), image_size.cpu().numpy()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    os.replace(temporary, path)


def status(stage, **fields):
    value = dict(stage=stage, time=time.time(), job=os.environ['SLURM_JOB_ID'], **fields)
    print(json.dumps(value, default=str), flush=True)
    save(OPS / (ARGS.command + ('_' + ARGS.mode if ARGS.mode else '') + '_status.json'), value)


def configure(payload):
    contract = payload['sources']['phase2_contract']
    for key, env in [('motion_stats_mean', 'MOTION_STATS_MEAN'),
                     ('motion_stats_std', 'MOTION_STATS_STD'),
                     ('nymeria_uniego_root', 'NYMERIA_UNIEGO_ROOT')]:
        os.environ[env] = contract[key]
    from motion_expert_phase3_edge.settings import sha256
    if sha256(os.environ['MOTION_STATS_MEAN']) != contract['motion_mean']:
        raise RuntimeError('Checkpoint motion mean hash mismatch')
    if sha256(os.environ['MOTION_STATS_STD']) != contract['motion_std']:
        raise RuntimeError('Checkpoint motion std hash mismatch')
    if payload.get('schema') != 1 or int(payload['step']) <= 0:
        raise RuntimeError('Expected a trained Phase-3 bridge checkpoint')
    return contract


def prepare(payload, contract):
    import config
    from dataset import humanize_caption
    from nymeria_joint_dataset import NymeriaJointDataset, decode_window_pyav, latent_path
    from native_phase_training.latent_cache_contract import load_latent_cache_contract
    from render_motion import write_mp4

    root = ARGS.nymeria_root
    split_file = root / 'train_test_split.json'
    split = json.loads(split_file.read_text())
    if set(split['train']) & set(split['test']):
        raise RuntimeError('Train/test recordings overlap')
    latent_root = root / 'joint_latents_T97_256tier_256'
    latent_contract = load_latent_cache_contract(latent_root)
    if (latent_contract.expected_latent_shape != (48, 25, 16, 16)
            or latent_contract.expected_image_hw != (256, 256)
            or latent_contract.num_frames != 97 or latent_contract.fps != 20):
        raise RuntimeError('Wrong training latent geometry')
    vae_path = ARGS.vae_path or Path(os.environ.get('WAN_VAE_PATH', latent_contract.vae_path))
    dataset = NymeriaJointDataset(
        manifest_path=str(root / 'video/manifest_video.jsonl'),
        split_file=str(split_file), split='test', num_frames=97, aligned_num_frames=97,
        fps=20, task_weights={'video2motion': 0.5, 'motimg2video': 0.5},
        bones_text2motion_frac=0, cfg_dropout=0, prefer_latents=True,
        latent_root=str(latent_root), train=False, uniego_root=config.NYMERIA_UNIEGO_ROOT,
        floor_calibration_json=str(root / 'metadata/floor_calibration.json'), seed=0,
    )
    # Deterministic distinct held-out recordings, fixed before generation.
    chosen, seen, rejected = [], set(), []
    for row in sorted(dataset._index, key=lambda r: (r['uuid'], r['s'])):
        if row['uuid'] in seen:
            continue
        try:
            motion, neutral, pad = dataset._load_motion(row['uni'], row['s'], row['off'], avail=97)
            if pad.any() or not np.isfinite(motion).all() or not np.isfinite(neutral).all():
                raise ValueError('Invalid held-out motion')
        except (ValueError, OSError, KeyError) as error:
            rejected.append(dict(uuid=row['uuid'], start=row['s'], reason=str(error)))
            continue
        chosen.append((row, motion, neutral))
        seen.add(row['uuid'])
        if len(chosen) == ARGS.num_cases:
            break
    if len(chosen) != ARGS.num_cases:
        raise RuntimeError(f'Could not select {ARGS.num_cases} valid distinct test recordings')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'inputs').mkdir(exist_ok=True)
    save(OPS / 'selection.json', dict(rows=[r for r, _, _ in chosen], rejected=rejected,
         policy='first valid aligned window in each of the first sorted distinct test recordings'))
    pipe, vae = make_pipeline(), None
    records = []
    for slot, (row, motion, neutral) in enumerate(chosen):
        assert row['uuid'] in set(split['test']) and row['uuid'] not in set(split['train'])
        case = f"case_{slot:02d}_{row['uuid'].replace('/', '__')}_{row['s']}"
        status('prepare_case', case=case, complete=len(records), total=ARGS.num_cases)
        frames = decode_window_pyav(row['vis'], int(row['s']), 97, 20.0)
        transformed = pipe({'video': torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous(),
                            'mode': 'image2video', 'ai_caption': ''},
                           latent_contract.spatial_transform_resolution)
        rgb = transformed['video'].permute(1, 2, 3, 0).contiguous().numpy()
        if rgb.shape != (97, 256, 256, 3) or rgb.dtype != np.uint8:
            raise RuntimeError(f'Unexpected held-out RGB shape/dtype: {rgb.shape}/{rgb.dtype}')
        cached = Path(latent_path(row['uuid'], row['s'], str(latent_root)))
        if cached.is_file():
            with np.load(cached, allow_pickle=False) as value:
                video = value['latents'].copy()
            latent_source = str(cached)
        else:
            if vae is None:
                vae = load_vae(vae_path,
                               latent_contract.spatial_transform_resolution, 97, 'cuda')
            video, _ = encode_window(vae, pipe, latent_contract.spatial_transform_resolution, frames, 'cuda')
            latent_source = 'independently_encoded_held_out_rgb_with_training_pipeline'
        if video.shape != (48, 25, 16, 16) or not np.isfinite(video).all():
            raise RuntimeError('Invalid held-out video latents')
        bundle = OUT / 'inputs' / (case + '.npz')
        if bundle.exists():
            raise FileExistsError(bundle)
        np.savez(bundle, motion=motion, neutral=neutral, video=video.astype(np.float16), rgb=rgb,
                 motion_mean_sha256=contract['motion_mean'], motion_std_sha256=contract['motion_std'])
        gt_mp4 = OUT / 'inputs' / (case + '_gt_video.mp4')
        write_mp4(rgb, str(gt_mp4), fps=20)
        records.append(dict(id=case, caption=humanize_caption(row['cap']), bundle=str(bundle),
                            seed=slot, uuid=row['uuid'], start=row['s'], split='test',
                            gt_video=str(gt_mp4), latent_source=latent_source,
                            motion_source=row['uni'], floor_offset=row['off']))
    save(OPS / 'cases.json', records)
    save(OUT / 'configuration.json', dict(checkpoint=str(CHECKPOINT), step=STEP,
         cases=records, steps=35, guidance=1.0, motion_shift=3.0, video_shift=10.0,
         frames=97, fps=20, sources=payload['sources'], latent_contract=latent_contract.to_dict(),
         vae_path=str(vae_path),
         note='Qualitative held-out samples only. No generated-motion grounding or recanonicalization.'))
    status('complete', cases=len(records), output=OUT)


def predict(payload, contract):
    from motion_expert_phase3_edge.model import EdgePhase3
    from motion_expert_phase3_edge.sample import sample

    mode = ARGS.mode
    records = json.loads((OPS / 'cases.json').read_text())
    destination = OUT / mode
    destination.mkdir(parents=True, exist_ok=True)
    sources = payload['sources']
    status('loading_model', mode=mode)
    model = EdgePhase3(sources['phase1'], sources['phase2'], device='cuda', activation_checkpointing=False)
    if (model.provenance['phase1_metadata_sha256'] != sources['phase1_metadata_sha256']
            or model.provenance['phase2_contract'] != sources['phase2_contract']):
        raise RuntimeError('Frozen specialist provenance differs from bridge checkpoint')
    model.bridges.load_state_dict(payload['bridges'], strict=True)
    model.eval()
    completed = []
    for row in records:
        path = destination / (row['id'] + '.npz')
        if path.exists():
            raise FileExistsError(path)
        with np.load(row['bundle'], allow_pickle=False) as bundle:
            if str(bundle['motion_mean_sha256'].item()) != contract['motion_mean']:
                raise RuntimeError('Bundle mean hash mismatch')
            if str(bundle['motion_std_sha256'].item()) != contract['motion_std']:
                raise RuntimeError('Bundle std hash mismatch')
            motion = torch.from_numpy(bundle['motion'].copy()).to('cuda', torch.float32)[None]
            neutral = torch.from_numpy(bundle['neutral'].copy()).to('cuda', torch.float32)[None]
            video = torch.from_numpy(bundle['video'].copy()).to('cuda', torch.float32)[None]
        status('sampling', mode=mode, case=row['id'], complete=len(completed), total=len(records))
        started = time.monotonic()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            generated = sample(model, mode=mode, caption=row['caption'], motion=motion,
                               neutral=neutral, video=video, seed=row['seed'], steps=35, guidance=1.0)
        np.savez(path, generated=generated[0].float().cpu().numpy(),
                 gt_motion=motion[0].cpu().numpy(), gt_video_latents=video[0].cpu().numpy(),
                 neutral=neutral[0].cpu().numpy())
        completed.append(dict(**row, output=str(path), sampling_seconds=time.monotonic() - started))
        save(destination / 'manifest.json', dict(mode=mode, step=STEP, cases=completed,
             steps=35, guidance=1.0, shift=3.0 if mode == 'video2motion' else 10.0,
             checkpoint=str(CHECKPOINT), sources=sources))
        del generated, motion, neutral, video
    del model
    gc.collect()
    torch.cuda.empty_cache()
    status('rendering', mode=mode, cases=len(completed))
    render(completed, mode, contract, destination)
    status('complete', mode=mode, cases=len(completed), output=destination)


def labeled_panel(frames, label, size=512):
    from PIL import Image, ImageDraw
    output = []
    for frame in frames:
        canvas = Image.fromarray(frame).resize((size, size), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, size, 26), fill='white')
        draw.text((10, 7), label, fill='black')
        output.append(np.asarray(canvas))
    return np.stack(output)


def render(records, mode, contract, destination):
    from PIL import Image
    from decode_uniego_torch import decode_joints
    from render_motion import render_sidebyside, render_single, head_rotation_from_features, write_mp4
    mean = np.load(contract['motion_stats_mean']).astype(np.float32)
    std = np.load(contract['motion_stats_std']).astype(np.float32)
    vae = None
    if mode == 'motimg2video':
        configuration = json.loads((OUT / 'configuration.json').read_text())
        vae = load_vae(configuration['vae_path'], '256', 97, 'cuda')
        if vae.get_output_range() != '[-1,1]':
            raise RuntimeError('Unexpected native Wan decoder output range')
    rendered = []
    for row in records:
        with np.load(row['bundle'], allow_pickle=False) as bundle:
            rgb = bundle['rgb'].copy()
        with np.load(row['output'], allow_pickle=False) as arrays:
            gt_raw = arrays['gt_motion'] * std + mean
            generated = arrays['generated'].copy()
        with torch.no_grad():
            gt_joints = decode_joints(torch.from_numpy(gt_raw)[None])[0].numpy()
        gt_head = head_rotation_from_features(gt_raw)
        if mode == 'video2motion':
            raw = generated * std + mean
            with torch.no_grad():
                joints = decode_joints(torch.from_numpy(raw)[None])[0].numpy()
            skeleton = render_sidebyside(gt_joints, joints, caption=row['caption'], width=1024,
                       height=512, gen_head_rotation0=head_rotation_from_features(raw), gt_head_rotation0=gt_head)
            frames = np.concatenate((labeled_panel(rgb, 'Input video (held-out GT)'), skeleton), axis=2)
            np.savez(destination / (row['id'] + '_joints.npz'), generated_joints=joints, gt_joints=gt_joints,
                     generated_features=raw, gt_features=gt_raw)
        else:
            with torch.no_grad():
                pixels = vae.decode(torch.from_numpy(generated)[None].to('cuda', torch.float32))
            if pixels.shape != (1, 3, 97, 256, 256) or not torch.isfinite(pixels).all():
                raise RuntimeError(f'Unexpected decoded video: {pixels.shape}')
            decoded = ((pixels[0].float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
            decoded = decoded.permute(1, 2, 3, 0).cpu().numpy()
            del pixels
            write_mp4(decoded, str(destination / (row['id'] + '_generated_video.mp4')), fps=20)
            skeleton = render_single(gt_joints, caption='Conditioning GT motion', width=512, height=512,
                                     color='tab:blue', head_rotation0=gt_head)
            frames = np.concatenate((skeleton, labeled_panel(rgb, 'GT video'),
                                     labeled_panel(decoded, 'Generated video (motion + first frame + caption)')), axis=2)
        mp4 = destination / (row['id'] + '_comparison.mp4')
        preview = destination / (row['id'] + '_preview.jpg')
        write_mp4(frames, str(mp4), fps=20)
        Image.fromarray(frames[0]).save(preview, quality=92)
        rendered.append(dict(id=row['id'], caption=row['caption'], comparison=str(mp4), preview=str(preview)))
        save(destination / 'renders.json', rendered)
        status('rendered_case', mode=mode, complete=len(rendered), total=len(records), video=mp4)


def gallery():
    sections, counts = [], {}
    for mode in ('video2motion', 'motimg2video'):
        rows = json.loads((OUT / mode / 'renders.json').read_text())
        counts[mode] = len(rows)
        sections.append(f'<h2>{html.escape(mode)}</h2>')
        for row in rows:
            video = Path(row['comparison']).relative_to(OUT)
            preview = Path(row['preview']).relative_to(OUT)
            sections.append(f'<p>{html.escape(row["id"])}<br>{html.escape(row["caption"])}</p>'
                            f'<video controls preload="none" width="1200" poster="{preview}" src="{video}"></video>')
    configuration = json.loads((OUT / 'configuration.json').read_text())
    step = int(configuration['step'])
    (OUT / 'index.html').write_text(f'<!doctype html><meta charset="utf-8"><title>Phase 3: {step}</title>'
        '<style>body{font:16px Georgia;margin:32px;background:#faf8f2}video{max-width:100%}</style>'
        f'<h1>Phase 3, step {step}: held-out Nymeria</h1>'
        f'<p>{len(configuration["cases"])} fixed test recordings. UniPC35, guidance1, motion shift3 / video shift10. '
        'V2M has no caption input. M2V conditions on motion, caption and first frame. '
        'No generated-motion grounding corrections. Ground truth accompanies every example.</p>' + ''.join(sections))
    save(OUT / 'COMPLETE.json', dict(step=step, counts=counts, output=str(OUT), checkpoint=str(CHECKPOINT)))
    status('complete', output=OUT / 'index.html', counts=counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'predict', 'render', 'gallery'])
    parser.add_argument('--mode', choices=['video2motion', 'motimg2video'])
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--nymeria-root', type=Path, default=Path(os.environ.get(
        'WEKA_ROOT', '/lustre/whwjdqls99/cosmos/weka')) / 'nymeriaplus_kimodo_proportional')
    parser.add_argument('--vae-path', type=Path)
    parser.add_argument('--num-cases', type=int, default=4)
    ARGS = parser.parse_args()
    if ARGS.num_cases < 1:
        parser.error('--num-cases must be positive')
    if ARGS.command in {'predict', 'render'} and not ARGS.mode:
        parser.error('--mode is required for predict/render')
    OUT = ARGS.out.resolve()
    OPS = (ARGS.work_dir or OUT / '_operations').resolve()
    CHECKPOINT = ARGS.checkpoint.resolve()
    from motion_expert_phase3_edge.settings import require_compute_node
    require_compute_node()
    torch.set_num_threads(1)
    try:
        if ARGS.command == 'gallery':
            gallery()
        else:
            torch.cuda.set_device(0)
            payload = torch.load(CHECKPOINT, map_location='cpu', mmap=True, weights_only=False)
            contract = configure(payload)
            STEP = int(payload['step'])
            if ARGS.command == 'prepare':
                prepare(payload, contract)
            elif ARGS.command == 'render':
                manifest = json.loads((OUT / ARGS.mode / 'manifest.json').read_text())
                render(manifest['cases'], ARGS.mode, contract, OUT / ARGS.mode)
                status('complete', mode=ARGS.mode, output=OUT / ARGS.mode)
            else:
                predict(payload, contract)
    except BaseException as error:
        status('failed', error=str(error), traceback=traceback.format_exc())
        raise
