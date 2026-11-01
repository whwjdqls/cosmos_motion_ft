"""Shape-free inference must be invariant to all externally supplied shapes."""
import os
import unittest
from unittest import mock
from types import SimpleNamespace
import torch
import torch.nn as nn
import ablation_settings as settings
import config
import model as module


class NoShapeTest(unittest.TestCase):
    def make_model(self, mode):
        backbone=nn.Module();backbone.layers=nn.ModuleList();backbone.norm=nn.LayerNorm(32)
        edge=SimpleNamespace(backbone=backbone, geometry=SimpleNamespace(
            hidden_size=32,num_attention_heads=4,num_key_value_heads=2,head_dim=8),
            rope=lambda pos:(torch.zeros(pos.shape[1],8),torch.zeros(pos.shape[1],8)))
        with mock.patch.object(settings,'SHAPE_CONDITIONING',mode):
            value=module.EdgePhase2MotionExpert(device='cpu',dtype=torch.float32,
                verbose=False,frozen_edge=edge)
        value.attention_mode='joint'
        with torch.no_grad():value.heads.llm2motion.weight.normal_()
        return value

    def test_shape_is_never_read_and_only_frame_tokens_are_packed(self):
        torch.manual_seed(21);value=self.make_model('none')
        batch=dict(reasoner_inputs=[dict(inputs_embeds=torch.randn(2,32),
                   position_ids=torch.zeros(3,2,dtype=torch.long)) for _ in range(2)],
                   x_sigma=torch.randn(2,5,283),sigma=torch.tensor([.3,.8]),
                   motion_pad_mask=torch.tensor([[False]*3+[True]*2,[False]*5]))
        positions=[]
        def mrope(n,text_length,device):
            result=torch.arange(n+1,device=device)[None].expand(3,-1)
            positions.append(result[:,1:].clone());return result
        with mock.patch.object(config,'HIDDEN_SIZE',32),mock.patch.object(module,'_motion_positions',side_effect=mrope), \
             mock.patch.object(value.heads,'encode_shape',side_effect=AssertionError('shape was read')):
            none=value(**batch,neutral_joints=None)
            neutral=torch.randn(2,30,3,requires_grad=True)
            changed=value(**batch,neutral_joints=neutral)
            invalid=value(**batch,neutral_joints=torch.full_like(neutral,float('nan')))
        self.assertTrue(torch.equal(none,changed));self.assertTrue(torch.equal(none,invalid))
        self.assertEqual(tuple(none.shape),(2,5,283));self.assertTrue(torch.isfinite(none).all())
        self.assertEqual(torch.count_nonzero(none[0,3:]).item(),0)
        changed.square().mean().backward();self.assertIsNone(neutral.grad)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in value.heads.shape2llm.parameters()))
        self.assertFalse(value.heads.shape_type_embed.requires_grad)
        self.assertGreater(value.heads.motion2llm.weight.grad.abs().sum().item(),0)

    def test_default_shape_contract_and_trainability_are_unchanged(self):
        value=self.make_model('neutral_joints')
        self.assertTrue(all(p.requires_grad for p in value.heads.shape2llm.parameters()))
        self.assertTrue(value.heads.shape_type_embed.requires_grad)
        with mock.patch.object(settings,'CODEC','uniego283'),mock.patch.object(settings,'OBJECTIVE','x0'), \
             mock.patch.object(settings,'SHAPE_CONDITIONING','neutral_joints'):
            self.assertIsNone(settings.contract())

    def test_shape_mode_is_restored_and_explicit_checkpoint_conflicts_fail(self):
        saved=dict(version=1,feature_codec='uniego283',objective='x0',shape_conditioning='none')
        with mock.patch.dict(os.environ,clear=False),mock.patch.object(settings,'CODEC','uniego283'), \
             mock.patch.object(settings,'OBJECTIVE','x0'),mock.patch.object(settings,'SHAPE_CONDITIONING','neutral_joints'):
            for k in ['EDGE_PHASE2_SHAPE_CONDITIONING','EDGE_PHASE2_FEATURE_CODEC','EDGE_PHASE2_OBJECTIVE']:os.environ.pop(k,None)
            settings.configure_checkpoint(saved);self.assertEqual(settings.SHAPE_CONDITIONING,'none')
            self.assertEqual(settings.contract()['shape_conditioning'],'none')
            os.environ['EDGE_PHASE2_SHAPE_CONDITIONING']='none'
            with self.assertRaisesRegex(RuntimeError,'conflicts'):settings.configure_checkpoint(None)
            os.environ.pop('EDGE_PHASE2_SHAPE_CONDITIONING')
            settings.configure_checkpoint(None);self.assertEqual(settings.SHAPE_CONDITIONING,'neutral_joints')


if __name__=='__main__':unittest.main()
