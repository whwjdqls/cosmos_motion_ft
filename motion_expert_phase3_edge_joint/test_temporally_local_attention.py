"""Wan-group locality, directional conditioning and absence of shape-mediated leakage."""
from unittest.mock import patch
import unittest

import torch
from torch.nn import functional as F

from modality_bridge import (
    BRIDGE_PARAMETERIZATIONS, BridgeMeta, LocalModalityBridge,
    temporally_local_attention_mask,
)
from scripts.phase3_performance import attention_bias, execution_policy


def metadata(mode='joint', *, groups=25, spatial=64, shape=True):
    gf=torch.arange(groups).repeat_interleave(spatial)
    mf=torch.arange(-1 if shape else 0,4*(groups-1)+1)
    return BridgeMeta(mode,gf,
        torch.ones_like(gf,dtype=torch.bool) if mode=='video2motion' else gf==0,
        mf,torch.ones_like(mf,dtype=torch.bool) if mode=='motimg2video' else mf<0,
        torch.where(gf==0,gf,4*gf-3),4*gf)


def oracle(mode, policy, *, groups=25, spatial=64):
    """Explicit groups, independent of the vectorized mask implementation."""
    ng=groups*spatial
    expected=torch.eye(ng+1+4*(groups-1)+1,dtype=torch.bool)
    for group in range(groups):
        video=list(range(group*spatial,(group+1)*spatial))
        frames=[0] if group==0 else list(range(4*group-3,4*group+1))
        motion=[ng+1+frame for frame in frames]
        for query in video:
            expected[query,video]=True
            if policy=='bidirectional' or (mode!='video2motion' and group!=0):
                expected[query,motion]=True
        for query in motion:
            expected[query,motion]=True
            if policy=='bidirectional' or mode!='motimg2video':
                expected[query,video]=True
    return expected


def opened_bridge(parameterization='mmdit', policy='bidirectional'):
    bridge=LocalModalityBridge(8,2,4,policy,parameterization)
    with torch.no_grad():
        bridge.gate_g.fill_(.3);bridge.gate_m.fill_(.2)
    return bridge


class TemporallyLocalAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(311)

    def test_complete_1698_mask_matches_explicit_groups_for_all_tasks_and_policies(self):
        for mode in ('video2motion','motimg2video','joint'):
            for policy in ('directional','bidirectional'):
                meta=metadata(mode)
                actual=temporally_local_attention_mask(meta,policy)
                self.assertEqual(actual.shape,(1698,1698))
                self.assertTrue(torch.equal(actual,oracle(mode,policy)))
                self.assertTrue(actual.diag().all())
                self.assertTrue(actual.any(dim=-1).all())

    def test_group_two_first_last_and_frame_boundaries(self):
        mask=temporally_local_attention_mask(metadata(),'bidirectional')
        for group,frames in [(0,[0]),(2,[5,6,7,8]),(24,[93,94,95,96])]:
            video=list(range(group*64,(group+1)*64))
            motion=[1601+frame for frame in frames]
            expected=set(video+motion)
            for query in [video[0],video[-1],motion[0],motion[-1]]:
                self.assertEqual(set(mask[query].nonzero().flatten().tolist()),expected)
        self.assertTrue(mask[1601+1,1601+4])
        self.assertFalse(mask[1601+4,1601+5])
        self.assertFalse(mask[1601,1602])
        self.assertFalse(mask[64,128])

    def test_shape_has_no_temporal_attention_edges_in_both_modes(self):
        for mode in ('video2motion','motimg2video','joint'):
            for policy in ('directional','bidirectional'):
                meta=metadata(mode)
                mask=temporally_local_attention_mask(meta,policy)
                self.assertEqual(mask[1600].nonzero().flatten().tolist(),[1600])
                self.assertFalse(mask[1601:,1600].any())
                self.assertFalse(mask[:1600,1600].any())

    def test_policies_only_change_permitted_cross_modal_edges(self):
        for mode in ('video2motion','motimg2video','joint'):
            meta=metadata(mode)
            a=temporally_local_attention_mask(meta,'directional')
            b=temporally_local_attention_mask(meta,'bidirectional')
            self.assertTrue(torch.equal(a[:1600,:1600],b[:1600,:1600]))
            self.assertTrue(torch.equal(a[1600:,1600:],b[1600:,1600:]))
            self.assertTrue((a<=b).all())
            self.assertTrue(b[:1600,1601:][meta.gen_clean].any())
            self.assertFalse(a[:1600,1601:][meta.gen_clean].any())
            self.assertFalse(a[1601:,:1600][meta.motion_clean[1:]].any())

    def test_plain_and_cached_paths_match_and_ignore_old_global_cache(self):
        for policy in ('directional','bidirectional'):
            meta=metadata()
            # A legacy cache must not silently restore global same-stream edges.
            old_key='_phase3_fp32_bias_v1' if policy=='directional' else '_phase3_fp32_bias_bidirectional_v1'
            setattr(meta,old_key,torch.zeros(1698,1698))
            bias=attention_bias(meta,torch.device('cpu'),policy)
            self.assertTrue(torch.equal(bias==0,temporally_local_attention_mask(meta,policy)))
            self.assertTrue(torch.isfinite(bias).all())
            self.assertEqual(bias.data_ptr(),attention_bias(meta,torch.device('cpu'),policy).data_ptr())
            self.assertLess(bias[64,128],0)

    def test_one_joint_softmax_over_both_modalities_and_shape(self):
        bridge=LocalModalityBridge(8,1,8,'bidirectional')
        with torch.no_grad():
            bridge.q.weight.zero_();bridge.k.weight.zero_()
            bridge.v.weight.copy_(torch.eye(8));bridge.o.weight.copy_(torch.eye(8))
            bridge.gate_g.fill_(1.);bridge.gate_m.fill_(1.)
        meta=metadata(groups=1,spatial=2)
        g,m=torch.randn(2,8),torch.randn(2,8)
        normalized=bridge.norm(torch.cat((g,m[1:])))
        expected=normalized.mean(dim=0)  # Both temporal modalities, without shape.
        original_softmax=F.softmax
        with patch('torch.nn.functional.softmax',wraps=original_softmax) as softmax:
            go,mo=bridge(g,m,meta)
        self.assertEqual(softmax.call_count,1)
        torch.testing.assert_close(go-g,expected.expand_as(g))
        torch.testing.assert_close(mo[1]-m[1],expected)
        self.assertTrue(torch.equal(mo[0],m[0]))

    def test_no_cross_group_forward_leak_even_through_repeated_shape_updates(self):
        for parameterization in BRIDGE_PARAMETERIZATIONS:
            for policy in ('directional','bidirectional'):
                for mode in ('video2motion','motimg2video','joint'):
                    meta=metadata(mode,groups=3,spatial=2)
                    g,m=torch.randn(6,8),torch.randn(10,8)
                    gp,mp=g.clone(),m.clone()
                    # Change only temporal group2, keeping shape fixed.
                    if mode=='motimg2video':mp[6:,0]+=3
                    else:gp[4:,1]-=3
                    layers=[opened_bridge(parameterization,policy) for _ in range(3)]
                    with execution_policy():
                        for bridge in layers:
                            g,m=bridge(g,m,meta);gp,mp=bridge(gp,mp,meta)
                            self.assertTrue(torch.equal(g[:4],gp[:4]))
                            self.assertTrue(torch.equal(m[:6],mp[:6]))  # Includes shape.
                    other=(g-gp)[4:] if mode=='motimg2video' else (m-mp)[6:]
                    self.assertGreater(other.abs().max().item(),1e-7)

    def test_blocked_groups_have_zero_input_gradient_and_shape_never_collects_frames(self):
        for parameterization in BRIDGE_PARAMETERIZATIONS:
            bridge=opened_bridge(parameterization)
            meta=metadata(groups=3,spatial=2)
            g=torch.randn(6,8,requires_grad=True);m=torch.randn(10,8,requires_grad=True)
            go,mo=bridge(g,m,meta)
            gg,gm=torch.autograd.grad(go[2:4].square().sum(),(g,m),retain_graph=True)
            self.assertEqual(gg[:2].count_nonzero(),0);self.assertEqual(gg[4:].count_nonzero(),0)
            self.assertEqual(gm[1].count_nonzero(),0);self.assertEqual(gm[6:].count_nonzero(),0)
            self.assertGreater(gm[2:6].abs().max().item(),0)
            self.assertEqual(gm[0].count_nonzero(),0)
            gg,gm=torch.autograd.grad(mo[0].square().sum(),(g,m))
            self.assertEqual(gg.count_nonzero(),0);self.assertEqual(gm[1:].count_nonzero(),0)

    def test_locality_with_no_shape_and_unidentified_rows_has_no_global_fallback(self):
        meta=metadata(groups=3,spatial=2,shape=False)
        meta.gen_frame[0]=-1
        mask=temporally_local_attention_mask(meta,'bidirectional')
        self.assertEqual(mask[0].nonzero().flatten().tolist(),[0])
        self.assertEqual(mask[:,0].nonzero().flatten().tolist(),[0])
        self.assertTrue(mask.diag().all())


if __name__=='__main__':unittest.main()
