import copy
import unittest
import torch
import modality_bridge as bridge
from motion_expert_phase3_edge_joint.model import bridge_meta
from motion_expert_phase3_edge_joint.settings import RECIPE
from .attention import attention_scope, global_attention_mask, global_joint_attention
from .policy import configure, TASKS

class ExperimentTests(unittest.TestCase):
    def tearDown(self):
        configure('temporal_local', 'directional')

    def test_recipes_only_differ_in_scope_and_identity(self):
        local=copy.deepcopy(configure('temporal_local'))
        global_=copy.deepcopy(configure('global'))
        self.assertEqual({k for k in set(local)|set(global_) if local.get(k)!=global_.get(k)}, {'experiment','bridge_attention_scope'})
        self.assertEqual(global_['tasks'],TASKS)
        self.assertEqual(global_['video_weight'],1.)
        self.assertEqual(global_['conditioning_attention'],'directional')
        self.assertTrue(all(v=='empty' for v in global_['caption_policy'].values()))
        self.assertEqual(global_['camera_prediction']['loss_weight'],0.)

    def test_global_far_edges_direction_and_shape(self):
        for mode in ['video2motion','motimg2video']:
            meta=bridge_meta(mode,'cpu');mask=global_attention_mask(meta,'directional')
            local=bridge.temporally_local_attention_mask(meta,'directional')
            self.assertTrue(mask[0,1599]);self.assertFalse(local[0,1599])
            self.assertTrue(mask[1601,-1]);self.assertFalse(local[1601,-1])
            self.assertFalse(mask[1600,:1600].any());self.assertFalse(mask[:1600,1600].any())
            self.assertTrue(mask[1600,1600])
            if mode=='video2motion':
                self.assertFalse(mask[:1600,1600:].any());self.assertTrue(mask[-1,0])
            else:
                self.assertFalse(mask[1601:,:1600].any());self.assertTrue(mask[1599,1601])

    def test_checkpoint_restores_scope_instead_of_environment_default(self):
        from .policy import configure_checkpoint
        payload={'scheduler':{'recipe':copy.deepcopy(configure('global'))}}
        configure('temporal_local')
        configure_checkpoint(payload)
        self.assertEqual(RECIPE['bridge_attention_scope'],'global')

    def test_bidirectional_local_recipe_checkpoint_and_mask(self):
        from .policy import configure_checkpoint
        directional=copy.deepcopy(configure('temporal_local', 'directional'))
        bidirectional=copy.deepcopy(configure('temporal_local', 'bidirectional'))
        self.assertEqual({k for k in directional if directional[k]!=bidirectional[k]},
                         {'experiment','conditioning_attention'})
        payload={'scheduler':{'recipe':bidirectional}}
        configure('global', 'directional')
        configure_checkpoint(payload)
        self.assertEqual(RECIPE['bridge_attention_scope'], 'temporal_local')
        self.assertEqual(RECIPE['conditioning_attention'], 'bidirectional')
        for mode in ['video2motion', 'motimg2video']:
            meta=bridge_meta(mode, 'cpu')
            mask=bridge.temporally_local_attention_mask(meta, 'bidirectional')
            self.assertTrue(torch.equal(mask, mask.T))
            self.assertTrue(mask[64, 1602])  # latent1 and motion1
            self.assertTrue(mask[1602, 64])
            self.assertFalse(mask[64, 1606])  # motion5 belongs to latent2
            self.assertFalse(mask[0, 64])
            self.assertFalse(mask[1600, :1600].any())

    def test_invalid_attention_is_rejected(self):
        with self.assertRaises(ValueError):configure('temporal_local', 'unknown')

    def test_global_math_and_gradients_match_explicit_dense_reference(self):
        meta=bridge.BridgeMeta('motimg2video',torch.tensor([0,1,2]),torch.tensor([1,0,0],dtype=torch.bool),
                              torch.tensor([0,4,8]),torch.ones(3,dtype=torch.bool),torch.tensor([0,1,5]),torch.tensor([0,4,8]))
        torch.manual_seed(4)
        q,k,v=[torch.randn(2,6,4,requires_grad=True) for _ in range(3)]
        mask=global_attention_mask(meta,'directional')
        ref=torch.softmax((q@k.transpose(-2,-1))*.5+torch.where(mask,0.,torch.finfo(torch.float32).min),-1)@v
        actual=global_joint_attention(q,k,v,meta,'directional')
        torch.testing.assert_close(actual,ref,rtol=0,atol=0)
        ga=torch.autograd.grad(actual.square().sum(),[q,k,v],retain_graph=True)
        gr=torch.autograd.grad(ref.square().sum(),[q,k,v])
        for a,r in zip(ga,gr):torch.testing.assert_close(a,r,rtol=0,atol=0)

    def test_scope_context_restores_canonical_local_behavior(self):
        original=bridge.temporally_local_attention_mask;grouped=bridge.grouped_joint_attention
        with attention_scope('global'):
            self.assertIs(bridge.temporally_local_attention_mask,global_attention_mask)
            self.assertIs(bridge.grouped_joint_attention,global_joint_attention)
            with attention_scope('global'):pass
            self.assertIs(bridge.temporally_local_attention_mask,global_attention_mask)
        self.assertIs(bridge.temporally_local_attention_mask,original)
        self.assertIs(bridge.grouped_joint_attention,grouped)

    def test_global_bridge_far_frame_effect_clean_direction_and_shape_bypass(self):
        meta=bridge.BridgeMeta('video2motion',torch.tensor([0,1,2]),torch.ones(3,dtype=torch.bool),
                              torch.tensor([-1,0,4,8]),torch.tensor([1,0,0,0],dtype=torch.bool),
                              torch.tensor([0,1,5]),torch.tensor([0,4,8]))
        torch.manual_seed(9)
        module=bridge.LocalModalityBridge(8,2,4,conditioning_attention='directional',parameterization='mmdit')
        with torch.no_grad():module.gate_g.fill_(.2);module.gate_m.fill_(.2)
        g,m=torch.randn(3,8),torch.randn(4,8)
        changed_g=g.clone();changed_g[-1,0]+=3
        with torch.no_grad():
            local_a=module(g,m,meta)[1];local_b=module(changed_g,m,meta)[1]
            torch.testing.assert_close(local_a[1],local_b[1],rtol=0,atol=0)
            with attention_scope('global'):
                a,b=module(g,m,meta);c,d=module(changed_g,m,meta)
                self.assertGreater(float((b[1]-d[1]).abs().max()),1e-6)
                perturbed=m.clone();perturbed[1:,0]+=2
                clean,_=module(g,perturbed,meta)
                torch.testing.assert_close(a,clean,rtol=0,atol=0)
                torch.testing.assert_close(b[0],m[0],rtol=0,atol=0)

if __name__=='__main__':unittest.main()
