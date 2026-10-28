"""The projection ablation adds exactly four tensors to baseline trainability."""
import tempfile
import unittest
from unittest.mock import patch

import torch

from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.utils.generator.optimizer import _build_params_with_metadata
from native_phase_training.latent_omni_model_edge import LatentOmniMoTModelEdge
from native_phase_training.run_contract import contract_from_config, persist_run_contract


def network():
    net = torch.nn.Module()
    for name in ("vae2llm", "llm2vae", "action2llm", "llm2action", "backbone", "time_embedder"):
        net.add_module(name, torch.nn.Linear(4, 4))
    net.action_modality_embed = torch.nn.Embedding(2, 4)
    for name in ("q_proj_moe_gen", "k_proj_moe_gen", "v_proj_moe_gen", "o_proj_moe_gen"):
        layer = torch.nn.Module()
        layer.lora_A, layer.lora_B = torch.nn.Linear(4, 2, bias=False), torch.nn.Linear(2, 4, bias=False)
        net.add_module(name, layer)
    return net


class VideoIOTests(unittest.TestCase):
    def test_trainability_and_optimizer_add_only_full_video_projections(self):
        selections = {}
        for mode in ("global_lora", "global_lora_video_io"):
            model = LatentOmniMoTModelEdge.__new__(LatentOmniMoTModelEdge)
            torch.nn.Module.__init__(model)
            model.adaptation_mode = mode
            model.net = network()
            with patch.object(OmniMoTModel, "install_attention_dispatch"):
                model.install_attention_dispatch(model.net)
            model._validate_and_log_trainable_parameters()
            keys = ["lora_A", "lora_B", "action2llm", "llm2action", "action_modality_embed"]
            if mode == "global_lora_video_io": keys += ["vae2llm", "llm2vae"]
            selected = _build_params_with_metadata(model, keys, {"action2llm": 4, "llm2action": 4, "action_modality_embed": 4}, 5e-5, False)
            ids = {id(p): meta for p, meta in selected}
            names = {n for n, p in model.net.named_parameters() if id(p) in ids}
            selections[mode] = names
            for n, p in model.net.named_parameters():
                self.assertEqual(p.requires_grad, n in names)
                if n.startswith(("vae2llm.", "llm2vae.")) and mode == "global_lora_video_io":
                    self.assertEqual(ids[id(p)].lr, 5e-5)
        self.assertEqual(selections["global_lora_video_io"] - selections["global_lora"],
                         {f"{m}.{p}" for m in ("vae2llm", "llm2vae") for p in ("weight", "bias")})

    def test_saved_contract_rejects_resume_as_standard_lora(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = dict(job=dict(path_local=temp),
                model=dict(adaptation_mode="global_lora_video_io", model_family="edge",
                    config=dict(lora_enabled=True,
                        lora_target_modules="q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen",
                        resolution="256", diffusion_expert_config=dict(base_fps=24.),
                        rectified_flow_training_config=dict(shift={"256": 3.}, loss_scale=10.,
                                                           image_loss_scale=None, action_loss_weight=10.))),
                dataloader_train=dict(dataloaders={task:dict(dataloader=dict(dataset=dict(
                    prefix_lengths=[1], num_frames=97, fps=20.)))
                    for task in ("forward_dynamics", "inverse_dynamics")}))
            contract = contract_from_config(cfg)
            self.assertEqual(contract.adaptation_mode, "global_lora_video_io")
            persist_run_contract(cfg)
            cfg["model"]["adaptation_mode"] = "global_lora"
            with self.assertRaises(RuntimeError): persist_run_contract(cfg)


if __name__ == "__main__":
    unittest.main()
