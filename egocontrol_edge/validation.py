"""Fixed held-out monitoring: native EMA sampling, paired metrics, W&B videos.

The cases are from the existing test71 cohort, not a new model-selection set.
No train batch is consumed; Python/NumPy/Torch RNG and module modes are restored.
All ranks enter every model forward, including EMA/FSDP collectives.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from cosmos_framework.utils.callback import Callback
from motion_expert_phase3_edge.settings import atomic_json, sha256, require_compute_node


class FixedValidation(Callback):
    def __init__(self, configuration, indices=(0, 23, 46, 70), every=5000, first_step=1):
        super().__init__()
        self.configuration = Path(configuration)
        self.indices = list(indices)
        self.every, self.first_step = int(every), int(first_step)
        if self.every <= 0 or not self.indices or len(set(self.indices)) != len(self.indices):
            raise ValueError('Validation requires a positive interval and unique fixed cases')

    def on_train_start(self, model, iteration=0):
        require_compute_node()
        configuration = json.loads(self.configuration.read_text())
        self.rows = [configuration['cases'][i] for i in self.indices]
        if any(row['split'] != 'test' for row in self.rows):
            raise ValueError('Expected the established held-out test71 cohort')
        train_index = json.loads((Path(os.environ['EGOCONTROL_CACHE']) / 'index.json').read_text())
        train_uuids = {r['uuid'] for r in train_index['rows']}
        if any(row['uuid'] in train_uuids for row in self.rows):
            raise ValueError('Validation recording overlaps the training cache')
        if model.config.rectified_flow_inference_config.scheduler_type != 'unipc':
            raise ValueError('Fixed validation must use the native UniPC sampler')
        self.mean = np.load(os.environ['EGOCONTROL_MOTION_MEAN'])
        self.std = np.load(os.environ['EGOCONTROL_MOTION_STD'])
        self.hashes = {k: sha256(os.environ['EGOCONTROL_MOTION_' + k.upper()]) for k in ('mean', 'std')}
        job = self.config.job
        self.output = Path(os.environ['IMAGINAIRE_OUTPUT_ROOT']) / job.project / job.group / job.name / 'validation'
        self.output.mkdir(parents=True, exist_ok=True)
        contract = dict(configuration=str(self.configuration), configuration_sha256=sha256(self.configuration),
                        indices=self.indices, cases=self.rows, every=self.every, first_step=self.first_step,
                        weights='EMA', sampler='native UniPC', steps=35, shift=10., guidance=1.,
                        captions=False, observed_frames=[0], scored_frames=[1, 96], fps=20,
                        normalization_sha256=self.hashes,
                        purpose='Fixed test71 subset for monitoring only; no model selection or tuning')
        if dist.get_rank() == 0:
            path = self.output / 'configuration.json'
            if path.exists() and json.loads(path.read_text()) != contract:
                raise RuntimeError('Fixed validation contract changed on resume')
            atomic_json(path, contract)
        print(f'[egocontrol/validation] cases={self.indices} every={self.every} first={self.first_step} '
              'EMA UniPC35 shift10 CFG1; held-out monitoring, not model selection', flush=True)

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        if iteration != self.first_step and (iteration <= 0 or iteration % self.every):
            return
        rank = dist.get_rank()
        destination = self.output / f'iter_{iteration:09d}'
        destination.mkdir(parents=True, exist_ok=True)
        python_state, numpy_state = random.getstate(), np.random.get_state()
        modes = [(module, module.training) for net in (model.net, model.net_ema) for module in net.modules()]
        poses = [getattr(net, '_egocontrol_pose', None) for net in (model.net, model.net_ema)]
        started = time.monotonic()
        print(f'[egocontrol/validation] start iteration={iteration} rank={rank}', flush=True)
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
                model.net.eval()
                # The native EMA storage is FP32, without the training net's
                # FSDP mixed-precision policy. Use the official EMA scope so
                # inference retains BF16 Q/K/V and restores training weights.
                # Cache training weights on CPU to avoid a second VRAM copy.
                with model.ema_scope('egocontrol_validation', is_cpu=True):
                    self._generate(model, destination, iteration, rank)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            for module, training in modes:
                module.training = training
            for net, pose in zip((model.net, model.net_ema), poses):
                net._egocontrol_pose = pose
        dist.barrier()
        print(f'[egocontrol/validation] complete iteration={iteration} rank={rank} '
              f'seconds={time.monotonic() - started:.1f} output={destination}', flush=True)

    def _generate(self, model, destination, iteration, rank):
        from cosmos_framework.data.generator.sequence_packing import SequencePlan
        from native_phase_training.latent_nymeria_dataset import _format_prompt_for_mode
        from native_phase_training.evaluate_inverse_forward import LPIPSAlex, _frame_metrics, _summarize_frame_metrics
        from native_phase_training.visualize_checkpoint import _make_video_comparison
        from egocontrol_edge.representation import from_normalized_motion
        from render_motion import write_mp4
        import wandb

        perceptual = LPIPSAlex('cuda', batch_size=16) if rank == 0 else None
        records, log_values = [], {}
        for row in self.rows:
            with np.load(row['bundle'], allow_pickle=False) as bundle:
                for key in ('mean', 'std'):
                    if str(bundle['motion_' + key + '_sha256'].item()) != self.hashes[key]:
                        raise ValueError('Validation motion normalization mismatch: ' + row['id'])
                motion, latent, rgb = (bundle[k].copy() for k in ('motion', 'video', 'rgb'))
            if latent.shape != (48, 25, 16, 16) or rgb.shape != (97, 256, 256, 3) or rgb.dtype != np.uint8:
                raise ValueError('Unexpected fixed validation shapes')
            pose = torch.from_numpy(from_normalized_motion(motion, self.mean, self.std))[None]
            sample = dict(video=torch.empty((3, 97, 1, 1), dtype=torch.uint8),
                          # Native image_size is [padded_h,padded_w,original_h,original_w].
                          image_size=torch.tensor([256., 256., 256., 256.]), conditioning_fps=torch.tensor(20), ai_caption='')
            sample = _format_prompt_for_mode(sample, 'image2video', None)
            batch = dict(video=[sample['video'][None]], video_latents=[torch.from_numpy(latent)],
                         image_size=sample['image_size'][None], conditioning_fps=torch.tensor([20]),
                         ai_caption=[sample['ai_caption']], egocontrol_pose=pose,
                         sequence_plan=[SequencePlan(has_text=True, has_vision=True, has_action=False,
                                                     condition_frame_indexes_vision=[0])])
            # Native FSDP/denoiser already apply the model precision. An outer
            # autocast would promote our LayerNorm/modulation to FP32 and split
            # UND/GEN dtypes, unlike the working native training/sampling path.
            result = model.generate_samples_from_batch(batch, net=model.net, seed=[int(row['seed'])],
                                                       num_steps=35, shift=10., guidance=1.)
            generated = result['vision'][0]
            reference = torch.from_numpy(latent)[None].to(generated.device)
            if generated.shape != reference.shape or not torch.isfinite(generated).all():
                raise FloatingPointError('Invalid generated validation latents')
            prefix_error = float((generated[:, :, :1].float() - reference[:, :, :1].float()).abs().max())
            # UniPC combines FP32 states, so an exactly zero conditioned velocity
            # can still accumulate roundoff. Keep the native output unmodified.
            print(f'[egocontrol/validation] observed_latent_max_abs_error={prefix_error:.9g}', flush=True)
            if prefix_error > 1e-4:
                raise RuntimeError(f'Sampler changed the observed first latent frame: max_abs={prefix_error}')
            # Only rank0 decodes/writes; all ranks still run the same denoiser calls.
            if rank == 0:
                decoded = {}
                for name, value in (('model', generated), ('VAE_GT', reference)):
                    pixels = model.decode(value.float())
                    if pixels.shape != (1, 3, 97, 256, 256) or not torch.isfinite(pixels).all():
                        raise FloatingPointError('Invalid validation decode')
                    decoded[name] = ((pixels[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()
                frame_values, metrics = {}, {}
                for name, predicted in (('model', decoded['model']), ('VAE_GT', decoded['VAE_GT']), ('GT', rgb)):
                    values = _frame_metrics(rgb[1:], predicted[1:], perceptual)
                    if not all(np.isfinite(v).all() for v in values.values()):
                        raise FloatingPointError('Non-finite validation metrics')
                    metrics[name] = _summarize_frame_metrics(values)
                    frame_values.update({name + '_' + k: v for k, v in values.items()})
                np.savez(destination / (row['id'] + '_frame_metrics.npz'), **frame_values)
                np.savez(destination / (row['id'] + '_latents.npz'), generated=generated[0].float().cpu().numpy())
                gt_path, pred_path, pair_path = [destination / (row['id'] + suffix + '.mp4')
                                                for suffix in ('_GT', '_generated', '_comparison')]
                write_mp4(rgb, str(gt_path), fps=20)
                write_mp4(decoded['model'], str(pred_path), fps=20)
                _make_video_comparison(gt_video=gt_path, generated_video=pred_path, output=pair_path,
                                       label=f'EgoControl Edge {len(model.conditioning_layers)} blocks / {iteration}', prefix_length=1)
                records.append(dict(id=row['id'], seed=row['seed'], uuid=row['uuid'], start=row['start'],
                                    observed_latent_max_abs_error=prefix_error, **metrics))
                log_values['validation_preview/' + row['id']] = wandb.Video(str(pair_path), format='mp4', fps=20)
                atomic_json(destination / 'per_case.json', records)
                print(f'[egocontrol/validation] case={row["id"]} model={metrics["model"]} GT={metrics["GT"]}', flush=True)
            dist.barrier()
        if rank == 0:
            summary = {name: {key: float(np.mean([r[name][key] for r in records]))
                              for key in ('psnr_db', 'ssim', 'lpips_alex')}
                       for name in ('model', 'VAE_GT', 'GT')}
            atomic_json(destination / 'summary.json', dict(step=iteration, n=len(records), aggregate=summary,
                        observed_latent_max_abs_error=max(r['observed_latent_max_abs_error'] for r in records),
                        observed_latent_tolerance=1e-4, cohort='fixed test71 subset; monitoring only'))
            for name, metrics in summary.items():
                log_values.update({f'validation/{name}/{key}': value for key, value in metrics.items()})
            if wandb.run is not None and not wandb.run.disabled:
                wandb.log(log_values, step=iteration)
                print(f'[egocontrol/validation] W&B samples+metrics logged at step={iteration}', flush=True)
