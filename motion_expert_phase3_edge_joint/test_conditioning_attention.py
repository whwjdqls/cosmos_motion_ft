"""Actual cross-modal influence, legacy parity and execution-cache correctness."""
import copy
import json
import subprocess
import sys
import types
import unittest
from pathlib import Path

import torch
from modality_bridge import BridgeMeta, LocalModalityBridge
from scripts.phase3_performance import attention_bias, bridge_forward
from motion_expert_phase3_edge_joint import settings


def metadata(mode):
    # Latent 0 is frame 0; latent 1 is frames 1..4, including two spatial rows.
    return BridgeMeta(mode, torch.tensor([0, 1, 1]),
        torch.ones(3, dtype=torch.bool) if mode == 'video2motion' else torch.tensor([True, False, False]),
        torch.tensor([-1, 0, 1, 4, 5]),
        torch.tensor([True, False, False, False, False]) if mode == 'video2motion' else torch.ones(5, dtype=torch.bool),
        torch.tensor([0, 1, 1]), torch.tensor([0, 4, 4]))


class ConditioningTests(unittest.TestCase):
    def test_default_preserves_legacy_outputs_with_nonzero_gates(self):
        old=types.ModuleType('_legacy_bridge');sys.modules[old.__name__]=old
        source=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[1]),'show','f9043d3:motion_expert_joint_attention/modality_bridge.py'],text=True)
        exec(compile(source,'legacy_bridge.py','exec'),old.__dict__)
        for mode in ['video2motion','motimg2video']:
            torch.manual_seed(12);a=old.LocalModalityBridge(8,2,4)
            torch.manual_seed(12);b=LocalModalityBridge(8,2,4)
            with torch.no_grad():a.gate_g.fill_(.2);a.gate_m.fill_(.3)
            b.load_state_dict(a.state_dict());g=torch.randn(3,8);m=torch.randn(5,8);meta=metadata(mode)
            for x,y in zip(a(g,m,meta),b(g,m,meta)):self.assertTrue(torch.equal(x,y))

    def test_bidirectional_reverses_local_edges_and_shape_edges(self):
        for mode in ['video2motion','motimg2video']:
            b=LocalModalityBridge(8,2,4,'bidirectional');meta=metadata(mode);mask=b._attention_mask(meta)
            self.assertTrue(torch.equal(mask,mask.T))
            self.assertTrue(mask[1,3+2]);self.assertTrue(mask[3+2,1])
            self.assertFalse(mask[1,3+4]);self.assertFalse(mask[3+4,1])
            self.assertTrue(mask[:3,3].all());self.assertTrue(mask[3,:3].all())

    def test_clean_queries_receive_target_information_only_in_bidirectional_mode(self):
        for mode in ['video2motion','motimg2video']:
            torch.manual_seed(7);g=torch.randn(3,8);m=torch.randn(5,8);meta=metadata(mode)
            for policy in ['directional','bidirectional']:
                b=LocalModalityBridge(8,2,4,policy)
                with torch.no_grad():b.gate_g.fill_(.2);b.gate_m.fill_(.3)
                ga,ma=b(g,m,meta);gp=g.clone();mp=m.clone();gp[:,0]+=2;mp[:,0]+=2
                other=b(g,mp,meta)[0] if mode=='video2motion' else b(gp,m,meta)[1]
                original=ga if mode=='video2motion' else ma
                if policy=='directional':self.assertTrue(torch.equal(original,other))
                else:self.assertGreater((original-other).abs().max().item(),1e-7)

    def test_cached_forward_and_backward_match_both_modes_on_shared_metadata(self):
        meta=metadata('video2motion')
        for policy in ['directional','bidirectional','directional']:
            torch.manual_seed(4);a=LocalModalityBridge(8,2,4,policy)
            with torch.no_grad():a.gate_g.fill_(.2);a.gate_m.fill_(.3)
            b=copy.deepcopy(a);g=torch.randn(3,8);m=torch.randn(5,8)
            expected=a(g,m,meta);actual=bridge_forward(b,g,m,meta)
            for x,y in zip(expected,actual):self.assertTrue(torch.equal(x,y))
            sum(x.square().sum() for x in expected).backward();sum(x.square().sum() for x in actual).backward()
            for x,y in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(x.grad,y.grad))
            bias=attention_bias(meta,torch.device('cpu'),policy)
            self.assertTrue(torch.equal(bias==0,a._attention_mask(meta)))

    def test_two_task_sampling_and_checkpoint_policy(self):
        weights=settings.task_probabilities(json.dumps(dict(video2motion=.5,motimg2video=.5,joint=0)))
        self.assertEqual(weights,dict(video2motion=.5,motimg2video=.5))
        old=copy.deepcopy(settings.RECIPE)
        try:
            payload={'scheduler':{'recipe':dict(conditioning_attention='bidirectional',clean_motion_timestep='zero',tasks=weights)}}
            settings.configure_checkpoint_conditioning(payload)
            self.assertEqual(settings.RECIPE['conditioning_attention'],'bidirectional')
            self.assertEqual(settings.RECIPE['clean_motion_timestep'],'zero')
            with self.assertRaisesRegex(ValueError,'conflicts'):
                settings.configure_checkpoint_conditioning(payload,attention='directional')
            settings.configure_checkpoint_conditioning({})
            self.assertEqual(settings.RECIPE['conditioning_attention'],'directional')
            self.assertEqual(settings.RECIPE['clean_motion_timestep'],'omit')
        finally:settings.RECIPE.clear();settings.RECIPE.update(old)


if __name__=='__main__':unittest.main()
