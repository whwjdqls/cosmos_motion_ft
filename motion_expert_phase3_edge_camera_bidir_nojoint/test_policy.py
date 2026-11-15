import copy
import unittest
import torch
import modality_bridge
from motion_expert_phase3_edge_joint.model import bridge_meta
from motion_expert_phase3_edge_joint.settings import RECIPE
from motion_expert_phase3_edge_camera_directional_nojoint.policy import configure as directional
from .policy import configure,configure_checkpoint

class PolicyTests(unittest.TestCase):
    def tearDown(self):directional('temporal_local')

    def test_only_attention_and_identity_change(self):
        a=copy.deepcopy(directional('temporal_local'));b=copy.deepcopy(configure('temporal_local'))
        self.assertEqual({k for k in set(a)|set(b) if a.get(k)!=b.get(k)},{'experiment','conditioning_attention'})
        self.assertEqual(b['tasks'],{'video2motion':.5,'motimg2video':.5,'joint':0.})
        self.assertEqual(b['video_weight'],1.)
        self.assertEqual(b['camera_prediction']['loss_weight'],0.)

    def test_clean_queries_gain_local_cross_edges_without_far_edges_or_shape(self):
        for mode in ['video2motion','motimg2video']:
            meta=bridge_meta(mode,'cpu');a=modality_bridge.temporally_local_attention_mask(meta,'directional');b=modality_bridge.temporally_local_attention_mask(meta,'bidirectional')
            temporal=torch.cat((meta.gen_frame>=0,meta.motion_frame>=0))
            self.assertTrue(torch.equal(b[temporal][:,temporal],b[temporal][:,temporal].T))
            self.assertTrue(b[:1600,1601:].any());self.assertTrue(b[1601:,:1600].any())
            self.assertFalse(b[0,-1]);self.assertFalse(b[1601,1599]);self.assertFalse(b[1600,:1600].any())
            self.assertFalse(torch.equal(a,b))

    def test_saved_bidirectional_policy_restores_and_directional_is_rejected(self):
        saved=copy.deepcopy(configure('temporal_local'));directional('temporal_local')
        configure_checkpoint({'scheduler':{'recipe':saved}})
        self.assertEqual(RECIPE['conditioning_attention'],'bidirectional')
        bad=copy.deepcopy(directional('temporal_local'))
        with self.assertRaises(ValueError):configure_checkpoint({'scheduler':{'recipe':bad}})

if __name__=='__main__':unittest.main()
