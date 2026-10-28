"""CPU regressions for the Edge projection-only ablations (use the Edge env)."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from native_phase_training.latent_omni_model_edge import LatentOmniMoTModelEdge, OmniMoTModel
from native_phase_training.run_contract import NativePhase1RunContract, SCHEMA_VERSION


class ProjectionModeTests(unittest.TestCase):
    def model(self, mode):
        net = nn.Module()
        for name in ('action2llm', 'llm2action', 'vae2llm', 'llm2vae',
                     'vae2llm_lidar', 'llm2vae_lidar', 'time_embedder', 'backbone'):
            net.add_module(name, nn.Linear(4, 4))
        net.register_parameter('action_modality_embed', nn.Parameter(torch.zeros(4)))
        model = object.__new__(LatentOmniMoTModelEdge)
        object.__setattr__(model, 'adaptation_mode', mode)
        object.__setattr__(model, 'net', net)
        with patch.object(OmniMoTModel, 'install_attention_dispatch'):
            model.install_attention_dispatch(net)
        return model

    def test_action_only_still_freezes_video(self):
        model = self.model('action_only')
        model._validate_and_log_trainable_parameters()
        self.assertEqual({n.split('.')[0] for n, p in model.net.named_parameters() if p.requires_grad},
                         {'action2llm', 'llm2action', 'action_modality_embed'})

    def test_video_projections_exclude_backbone_timestep_and_other_modalities(self):
        model = self.model('action_video_projections')
        model._validate_and_log_trainable_parameters()
        self.assertEqual({n.split('.')[0] for n, p in model.net.named_parameters() if p.requires_grad},
                         {'action2llm', 'llm2action', 'action_modality_embed', 'vae2llm', 'llm2vae'})
        model.net.backbone.weight.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, 'unexpected trainable'):
            model._validate_and_log_trainable_parameters()

    def test_contract_round_trip_and_rejects_lora_or_nano(self):
        contract = NativePhase1RunContract(
            schema_version=SCHEMA_VERSION, adaptation_mode='action_video_projections',
            active_modes=('forward_dynamics', 'inverse_dynamics'),
            dropped_modes=('image2video', 'policy'), lora_enabled=False,
            lora_target_modules=('q_proj_moe_gen', 'k_proj_moe_gen', 'v_proj_moe_gen', 'o_proj_moe_gen'),
            training_prefix_lengths=(1,), model_resolution='256', num_frames=97,
            training_shift=3.0, inference_shift=10.0, model_family='edge',
            conditioning_fps=20.0, base_fps=20.0, vision_loss_scale=10.0,
            image_loss_scale=None, action_loss_weight=10.0)
        self.assertEqual(NativePhase1RunContract.from_dict(contract.to_dict()), contract)
        with self.assertRaisesRegex(ValueError, 'cannot enable LoRA'):
            replace(contract, lora_enabled=True)
        with self.assertRaisesRegex(ValueError, 'requires Edge'):
            replace(contract, model_family='nano')


if __name__ == '__main__':
    unittest.main()
