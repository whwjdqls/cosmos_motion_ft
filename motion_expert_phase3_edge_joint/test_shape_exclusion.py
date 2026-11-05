"""Shape bypasses bridge normalization, QKV, attention and residuals entirely."""
from contextlib import contextmanager
from unittest.mock import patch
import unittest

import torch

from modality_bridge import BRIDGE_PARAMETERIZATIONS, BridgeMeta, LocalModalityBridge
from scripts.phase3_performance import execution_policy


def metadata(task='joint'):
    # Deliberately put shape in the middle, rather than relying on row zero.
    gf=torch.tensor([0,1,1]); mf=torch.tensor([0,-1,1,4,5])
    return BridgeMeta(task,gf,torch.ones(3,dtype=torch.bool) if task=='video2motion' else gf==0,
        mf,torch.ones(5,dtype=torch.bool) if task=='motimg2video' else mf<0,
        torch.tensor([0,1,1]),torch.tensor([0,4,4]))


@contextmanager
def policy(path):
    if path=='dense': yield
    elif path=='nested':
        with execution_policy(grouped_attention=True):
            with execution_policy(): yield
    else:
        with execution_policy(grouped_attention=path=='grouped'): yield


def bridge(kind,conditioning):
    b=LocalModalityBridge(8,2,4,conditioning,kind)
    with torch.no_grad(): b.gate_g.fill_(.3); b.gate_m.fill_(-.2)
    return b


class ShapeExclusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(42)

    def test_shape_is_bit_exact_and_cannot_affect_any_bridge_output(self):
        for kind in BRIDGE_PARAMETERIZATIONS:
            for conditioning in ('directional','bidirectional'):
                b=bridge(kind,conditioning)
                for task in ('video2motion','motimg2video','joint'):
                    meta=metadata(task)
                    for dtype in (torch.float32,torch.bfloat16):
                        g,m=torch.randn(3,8).to(dtype),torch.randn(5,8).to(dtype)
                        m[1,0]=-0.0
                        changed=m.clone(); changed[1].fill_(float('nan'))
                        for path in ('dense','cached','grouped','nested'):
                            with self.subTest(kind=kind,conditioning=conditioning,task=task,dtype=dtype,path=path):
                                with policy(path):
                                    go,mo=b(g,m,meta); gp,mp=b(g,changed,meta)
                                self.assertTrue(torch.equal(mo[1].view(torch.uint8),m[1].view(torch.uint8)))
                                self.assertTrue(torch.equal(mp[1].view(torch.uint8),changed[1].view(torch.uint8)))
                                self.assertTrue(torch.equal(go,gp))
                                self.assertTrue(torch.equal(mo[[0,2,3,4]],mp[[0,2,3,4]]))

    def test_mmdit_shape_never_enters_motion_norm_qkv_or_attention(self):
        for path in ('dense','cached','grouped','nested'):
            b=bridge('mmdit','bidirectional'); seen={}
            handles=[]
            for name in ('norm_m','q_m','k_m','v_m','o_m'):
                handles.append(getattr(b,name).register_forward_pre_hook(
                    lambda module,args,key=name: seen.__setitem__(key,args[0].detach().clone())))
            g,m=torch.randn(3,8),torch.randn(5,8)
            with policy(path):
                original=b._joint_attention;calls=[]
                def attention(*args,**kwargs):
                    calls.append(args)
                    return original(*args,**kwargs)
                # Preserve the grouped dispatch marker when instrumenting it.
                attention.__func__=getattr(original,'__func__',None)
                with patch.object(b,'_joint_attention',attention):
                    b(g,m,metadata())
                    self.assertEqual(len(calls),1)
                    q,k,v,meta=calls[0]
                    self.assertEqual(q.shape[1],7); self.assertEqual(k.shape[1],7); self.assertEqual(v.shape[1],7)
                    self.assertTrue((meta.motion_frame>=0).all())
            for handle in handles: handle.remove()
            self.assertTrue(torch.equal(seen['norm_m'],m[[0,2,3,4]]))
            self.assertEqual({name:value.shape[0] for name,value in seen.items()},
                             {name:4 for name in ('norm_m','q_m','k_m','v_m','o_m')})

    def test_shape_bypass_has_identity_gradient_and_no_bridge_parameter_gradient(self):
        for kind in BRIDGE_PARAMETERIZATIONS:
            for path in ('dense','cached','grouped'):
                b=bridge(kind,'bidirectional')
                g=torch.randn(3,8,requires_grad=True);m=torch.randn(5,8,requires_grad=True)
                with policy(path): go,mo=b(g,m,metadata())
                gg,gm,*parameters=torch.autograd.grad(mo[1].sum(),(g,m,*b.parameters()),
                                                     allow_unused=True,retain_graph=True)
                self.assertTrue(gg is None or gg.count_nonzero()==0)
                expected=torch.zeros_like(m);expected[1]=1
                self.assertTrue(torch.equal(gm,expected))
                self.assertTrue(all(p is None or p.count_nonzero()==0 for p in parameters))
                gm=torch.autograd.grad(go.square().sum()+mo[[0,2,3,4]].square().sum(),m)[0]
                self.assertEqual(gm[1].count_nonzero(),0)
                self.assertGreater(gm[[0,2,3,4]].abs().max().item(),0)

    def test_no_shape_and_shape_only_inputs(self):
        b=bridge('mmdit','bidirectional');g,m=torch.randn(3,8),torch.randn(5,8)
        for path in ('dense','cached','grouped'):
            meta=metadata(); meta.motion_frame=torch.arange(5)
            with policy(path):
                actual=b(g,m,meta);expected=b._forward_frames(g,m,meta)
            for a,e in zip(actual,expected): self.assertTrue(torch.equal(a,e))
            meta=metadata();meta.motion_frame.fill_(-1)
            with policy(path): go,mo=b(g,m,meta)
            self.assertTrue(torch.equal(go,g));self.assertTrue(torch.equal(mo,m))


if __name__=='__main__': unittest.main()
