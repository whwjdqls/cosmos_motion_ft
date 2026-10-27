"""Small runtime audit for the explicitly requested GPU acceptance gate."""
import os
import json
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor
from cosmos_framework.utils.callback import Callback
from motion_expert_phase3_edge.settings import atomic_json, sha256


class ComparisonContract(Callback):
    def on_train_start(self, model, iteration=0):
        if int(os.environ.get('RANK', '0')):
            return
        root = Path(os.environ['REPO_ROOT'])
        cache = Path(os.environ['EGOCONTROL_CACHE'])
        index = json.loads((cache / 'index.json').read_text())
        initial = Path(os.environ['EGOCONTROL_INITIAL_CHECKPOINT'])
        named = list(model.net.named_parameters())
        from egocontrol_edge.representation import BODY23, SOMA_BODY_NAMES
        contract = {
            'method': 'EgoControl adapted to Cosmos3-Edge',
            'upstream_commit': '5dcfb82ef898315d415f8f04c94337c83d9a099d',
            'task': 'first RGB frame + body motion -> video',
            'conditioning_rgb_frames': [0], 'semantic_captions': False,
            'camera_action_input': False, 'motion_auxiliary_loss': False,
            'frames': 97, 'fps': 20, 'resolution': [256, 256],
            'pose_frames': 96, 'pose_features': 138, 'body_joints': 23,
            'conditioning_layout': model.conditioning_layout,
            'conditioning_layers_zero_based': list(model.conditioning_layers),
            'final_pose_modulation': model.final_pose_modulation,
            'soma_joint_indices': list(BODY23), 'soma_joint_names': list(SOMA_BODY_NAMES),
            'phase3_cache': str(cache), 'cache_fingerprint': index['fingerprint'],
            'full_training_windows': len(index['rows']),
            'acceptance_subset': os.environ.get('NYMERIA_MAX_SAMPLES'),
            'normalization_sha256': {k: sha256(os.environ['EGOCONTROL_MOTION_' + k.upper()])
                                     for k in ('mean', 'std')},
            'initial_phase1': str(initial), 'initial_weights': 'EMA',
            'initial_metadata_sha256': sha256(initial / 'model' / '.metadata'),
            'fresh_optimizer_scheduler': True,
            'global_batch': int(os.environ.get('WORLD_SIZE', '1')) * int(os.environ['NATIVEP1_CLIPS_PER_GPU']),
            'target_updates': 200000, 'm2v_task_probability': 1.,
            'pose_parameters': sum(p.numel() for n, p in named if 'egocontrol' in n),
            'trainable_parameters': sum(p.numel() for n, p in named if p.requires_grad),
            'training_recipe': 'Phase1; resolved config.yaml is authoritative',
            'comparison_limits': [
                'Different motion representation and parameter count',
                'Ours has additional Phase2 motion pretraining',
                'Historical Phase3 M2V training used captions; evaluate both without them',
                'M2V-only receives more M2V exposures per update; also compare100k to original Phase3 200k',
            ],
            'evaluation_policy': {
                'cohort': 'same fixed held-out clean71 (or same full-window set) for both',
                'caption': '', 'observed_frames': [0], 'scored_frames': [1, 96],
                'sampling': 'identical native solver, steps, shift, CFG and seeds for both',
                'metrics': ['PSNR', 'SSIM', 'LPIPS'], 'report_gt_reference': True,
                'model_selection': 'fixed100k and200k; no test-driven tuning',
            },
            'source_sha256': {str(p.relative_to(root)): sha256(p)
                              for p in sorted((root / 'egocontrol_edge').glob('*'))
                              if p.suffix in ('.py', '.sh', '.toml')},
        }
        job = self.config.job
        output = Path(os.environ['IMAGINAIRE_OUTPUT_ROOT']) / job.project / job.group / job.name
        path = output / 'egocontrol_contract.json'
        if path.exists():
            prior = json.loads(path.read_text())
            # Contracts written before the seven-site variant describe all28
            # adapters plus final modulation. Never infer them as seven-site.
            if 'conditioning_layout' not in prior:
                prior.update(conditioning_layout='legacy_all28',
                             conditioning_layers_zero_based=list(range(28)), final_pose_modulation=True)
            for key in ('cache_fingerprint', 'normalization_sha256', 'initial_metadata_sha256', 'global_batch',
                        'soma_joint_indices', 'conditioning_layout', 'conditioning_layers_zero_based',
                        'final_pose_modulation'):
                if prior[key] != contract[key]:
                    raise RuntimeError('Resume comparison contract differs: ' + key)
        else:
            atomic_json(path, contract)
        print(f'[egocontrol/contract] {path} global_batch={contract["global_batch"]} '
              f'trainable={contract["trainable_parameters"]}', flush=True)


class AcceptanceAudit(Callback):
    def on_train_start(self, model, iteration=0):
        import torch.distributed as dist
        from egocontrol_edge.layers import EgoControlDecoderLayer
        from cosmos_framework.model.generator.mot.unified_mot import MoTDecoderLayer
        layers = [m for m in model.net.modules() if isinstance(m, MoTDecoderLayer)]
        selected = [i for i, m in enumerate(layers) if isinstance(m, EgoControlDecoderLayer)]
        if len(layers) != 28 or selected != list(model.conditioning_layers):
            raise RuntimeError('Unexpected motion conditioning placement: ' + str(selected))
        for i, layer in enumerate(layers):
            # FSDP may wrap the native class; its forward must still be native.
            if i not in selected and (type(layer).forward is not MoTDecoderLayer.forward or hasattr(layer, 'egocontrol')):
                raise RuntimeError('An unselected native layer was changed')
        if (model.net.egocontrol.final_modulation is not None) != model.final_pose_modulation:
            raise RuntimeError('Unexpected final-output pose conditioning')
        print(f'[egocontrol/acceptance] selected_layers={selected} native_layers={28-len(selected)} '
              f'final_pose_modulation={model.final_pose_modulation}', flush=True)
        if dist.is_initialized() and dist.get_world_size() > 1:
            signatures = []
            for name, parameter in model.net.named_parameters():
                if 'egocontrol' not in name:
                    continue
                local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
                flat = local.detach().flatten()
                signatures.extend((flat.sum(), flat.norm(), flat[0], flat[-1]))
            signature = torch.stack(signatures)
            gathered = [torch.empty_like(signature) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, signature)
            if any(not torch.equal(gathered[0], other) for other in gathered[1:]):
                raise RuntimeError('Pose parameter signatures differ between replicated ranks')
            print('[egocontrol/acceptance] replicated_pose_parameter_signatures_equal=True', flush=True)
        # Restored adapters are already nonzero and must not be tested as fresh.
        if iteration == 0:
            from egocontrol_edge.layers import EgoControlDecoderLayer
            for layer in model.net.modules():
                if isinstance(layer, EgoControlDecoderLayer):
                    layer._acceptance_initial_noop = True
        print(f'[egocontrol/acceptance] loaded_iteration={iteration} '
              f'rank={os.environ.get("RANK", "0")}', flush=True)

    def on_before_optimizer_step(self, model, optimizer, scheduler, grad_scaler, iteration=0):
        # Keep the reference branch present through activation-checkpoint
        # recomputation, then disable it only after backward has completed.
        from egocontrol_edge.layers import EgoControlDecoderLayer
        for layer in model.net.modules():
            if isinstance(layer, EgoControlDecoderLayer):
                layer._acceptance_initial_noop = False
        groups = {'lora': [], 'pose': [], 'pose_keys': []}
        for name, parameter in model.net.named_parameters():
            if not parameter.requires_grad:
                if parameter.grad is not None:
                    raise RuntimeError('Frozen base received a gradient: ' + name)
                continue
            grad = parameter.grad
            if grad is None:
                continue
            grad = grad.to_local() if isinstance(grad, DTensor) else grad
            norm = torch.linalg.vector_norm(grad.float()).detach()
            groups['pose' if 'egocontrol' in name else 'lora'].append(norm)
            if 'egocontrol.k_proj' in name:
                groups['pose_keys'].append(norm)
        totals = {name: float(torch.stack(norms).sum()) if norms else 0.
                  for name, norms in groups.items()}
        if not all(torch.isfinite(torch.tensor(v)) for v in totals.values()):
            raise RuntimeError('Non-finite gradients: ' + str(totals))
        if totals['lora'] == 0 or totals['pose'] == 0:
            raise RuntimeError('Inactive trainable branch: ' + str(totals))
        # K/V are zero-gated at initialization; they receive gradients after
        # the first optimizer update opens the pose-attention gate.
        if iteration > 0 and totals['pose_keys'] == 0:
            raise RuntimeError('Pose cross-attention keys still receive no gradient')
        print(f'[egocontrol/acceptance] gradients iteration={iteration} '
              f'rank={os.environ.get("RANK", "0")} {totals}', flush=True)
