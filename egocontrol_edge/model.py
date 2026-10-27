"""Native Phase1 objective/EMA/checkpoint lifecycle with isolated pose conditioning."""
from __future__ import annotations

import os
import torch

from native_phase_training.latent_omni_model_edge import LatentOmniMoTModelEdge
from cosmos_framework.data.generator.sequence_packing.runtime import (
    from_und_gen_splits, get_gen_seq, get_und_seq,
)
from egocontrol_edge.layers import PoseEmbedding, PoseBlock, EgoControlDecoderLayer


class EgoControlOmniModel(LatentOmniMoTModelEdge):
    def __init__(self, config, adaptation_mode='global_lora', model_family='edge',
                 conditioning_layout='phase3_seven'):
        if conditioning_layout not in ('phase3_seven', 'legacy_all28'):
            raise ValueError('Unknown conditioning layout: ' + conditioning_layout)
        self.conditioning_layout = conditioning_layout
        self.conditioning_layers = (3, 7, 11, 15, 19, 23, 27) if conditioning_layout == 'phase3_seven' else tuple(range(28))
        self.final_pose_modulation = conditioning_layout == 'legacy_all28'
        super().__init__(config, adaptation_mode=adaptation_mode, model_family=model_family)

    def install_attention_dispatch(self, net):
        super().install_attention_dispatch(net)
        with torch.device(next(net.parameters()).device):
            net.egocontrol = PoseEmbedding(final_modulation=self.final_pose_modulation)
            if len(net.language_model.model.layers) != 28:
                raise ValueError('Conditioning schedule requires the 28-layer Edge backbone')
            for index in self.conditioning_layers:
                layer = net.language_model.model.layers[index]
                layer.__class__ = EgoControlDecoderLayer
                layer.egocontrol = PoseBlock()
        for name, parameter in net.named_parameters():
            parameter.requires_grad_('lora_' in name or 'egocontrol' in name)

        def before_language(module, args, kwargs):
            pose = getattr(net, '_egocontrol_pose', None)
            if pose is None:
                raise RuntimeError('Motion conditioning is required; refusing unconditional training')
            kwargs['attention_mask'].egocontrol_context = net.egocontrol(pose)
            return args, kwargs

        def after_language(module, args, kwargs, output):
            packed, metadata = output
            context = kwargs['attention_mask'].egocontrol_context
            return from_und_gen_splits(get_und_seq(packed),
                net.egocontrol.final(get_gen_seq(packed), context), packed), metadata

        net.language_model.register_forward_pre_hook(before_language, with_kwargs=True)
        if self.final_pose_modulation:
            net.language_model.register_forward_hook(after_language, with_kwargs=True)
        print(f'[egocontrol/architecture] layers={self.conditioning_layers} '
              f'final_pose_modulation={self.final_pose_modulation}', flush=True)

    def _can_reuse_inference_text_kv(self, *args, **kwargs):
        # Pose adapters implement the native full UND/GEN path. Do not enter
        # the optional cached-reasoner/gen_only fast path during previews.
        return False

    def _init_lora_weights_post_materialization(self, network):
        super()._init_lora_weights_post_materialization(network)
        for module in network.modules():
            if isinstance(module, (PoseEmbedding, PoseBlock)):
                module.reset_parameters()

    def _validate_and_log_trainable_parameters(self):
        named = [(name, p) for name, p in self.net.named_parameters() if p.requires_grad]
        unexpected = [name for name, _ in named if 'lora_' not in name and 'egocontrol' not in name]
        if unexpected:
            raise RuntimeError('Unexpected trainable weights: ' + str(unexpected[:8]))
        counts = {key: sum(p.numel() for name, p in named if key in name) for key in ['lora_', 'egocontrol']}
        if not all(counts.values()):
            raise RuntimeError('Both LoRA and pose-conditioning parameters must train')
        print('[egocontrol/trainable]', counts, flush=True)

    def get_data_and_condition(self, data_batch, **kwargs):
        pose = data_batch.get('egocontrol_pose')
        if isinstance(pose, list):
            pose = torch.stack([p.squeeze(0) if p.ndim == 3 else p for p in pose])
        if not isinstance(pose, torch.Tensor) or tuple(pose.shape[1:]) != (96, 138):
            raise ValueError(f'Lost/malformed motion conditioning: {type(pose)} {getattr(pose, "shape", None)}')
        if data_batch.get('action') is not None:
            raise RuntimeError('Camera actions are not an input to this baseline')
        for net in (self.net, getattr(self, 'net_ema', None)):
            if net is not None:
                net._egocontrol_pose = pose.to(device=self.tensor_kwargs['device'], dtype=torch.float32)
        return super().get_data_and_condition(data_batch, **kwargs)

    def training_step(self, data_batch, iteration):
        outputs, loss = super().training_step(data_batch, iteration)
        if os.environ.get('EGOCONTROL_ACCEPTANCE') == '1':
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite acceptance loss')
            print(f'[egocontrol/acceptance] iteration={iteration} loss={float(loss.detach()):.8g} '
                  f'vision_tokens={outputs["vision_token_length"]} action_tokens={outputs["action_token_length"]}', flush=True)
        return outputs, loss
