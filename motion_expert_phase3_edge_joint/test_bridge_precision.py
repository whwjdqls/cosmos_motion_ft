"""Opt-in BF16 compute preserves FP32 master weights and the shape bypass."""
import copy
from functools import partial
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.multiprocessing as mp

from modality_bridge import LocalModalityBridge, BRIDGE_PARAMETERIZATIONS
from scripts.phase3_performance import execution_policy
from .test_shape_exclusion import metadata, bridge


def precision_ddp_worker(rank, rendezvous):
    from . import test_mmdit_bridge as tests
    tests.execution_policy=partial(execution_policy,grouped_attention=True,precision='bf16_bridge')
    tests.mmdit_ddp_worker(rank,rendezvous)


class BridgePrecisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(177)

    def test_actual_operator_dtypes_and_fp32_master_gradients(self):
        for precision in ('fp32','bf16_attention','bf16_bridge'):
            b=bridge('mmdit','bidirectional'); seen={}; products=[]; handles=[]
            for name,module in b.named_modules():
                if isinstance(module,(nn.Linear,nn.LayerNorm)):
                    handles.append(module.register_forward_hook(
                        lambda module,args,out,label=name: seen.__setitem__(label,out.dtype)))
            original=torch.matmul
            def matmul(a,c,*args,**kwargs):
                products.append((a.dtype,c.dtype));return original(a,c,*args,**kwargs)
            g,m=torch.randn(3,8),torch.randn(5,8)
            with execution_policy(grouped_attention=True,precision=precision):
                with patch('torch.matmul',side_effect=matmul):
                    out=b(g,m,metadata())
                sum(x.square().mean() for x in out).backward()
            for h in handles:h.remove()
            expected=torch.float32 if precision=='fp32' else torch.bfloat16
            self.assertEqual(products,[(expected,expected),(expected,expected)])
            self.assertEqual(seen['norm_g'],torch.float32);self.assertEqual(seen['norm_m'],torch.float32)
            for name,dtype in seen.items():
                if not name.startswith('norm'):
                    self.assertEqual(dtype,torch.bfloat16 if precision=='bf16_bridge' else torch.float32)
            self.assertTrue(all(p.dtype==torch.float32 and p.grad.dtype==torch.float32 for p in b.parameters()))

    def test_shape_bypass_zero_gates_and_finite_gradients_all_bridges(self):
        for precision in ('bf16_attention','bf16_bridge'):
            for kind in BRIDGE_PARAMETERIZATIONS:
                for policy in ('directional','bidirectional'):
                    b=bridge(kind,policy)
                    for task in ('video2motion','motimg2video','joint'):
                        meta=metadata(task)
                        for dtype in (torch.float32,torch.bfloat16):
                            g=torch.randn(3,8,dtype=dtype,requires_grad=True)
                            m=torch.randn(5,8,dtype=dtype,requires_grad=True)
                            with torch.no_grad():m[1,0]=-0.
                            with execution_policy(grouped_attention=True,precision=precision):
                                with execution_policy(): # Existing trainer's nested cache policy.
                                    go,mo=b(g,m,meta)
                                    self.assertTrue(torch.equal(mo[1].view(torch.uint8),m[1].view(torch.uint8)))
                                    gm=torch.autograd.grad(go.square().sum()+mo[[0,2,3,4]].square().sum(),m)[0]
                                    self.assertEqual(gm[1].count_nonzero(),0)
                                    self.assertTrue(torch.isfinite(gm).all())
                                poisoned=m.detach().clone();poisoned[1].fill_(float('nan'))
                                pg,pm=b(g.detach(),poisoned,meta)
                                self.assertTrue(torch.equal(go.detach(),pg));self.assertTrue(torch.equal(mo[[0,2,3,4]].detach(),pm[[0,2,3,4]]))
                    with torch.no_grad():b.gate_g.zero_();b.gate_m.zero_()
                    with execution_policy(grouped_attention=True,precision=precision):
                        go,mo=b(g,m,metadata())
                    self.assertTrue(torch.equal(go,g));self.assertTrue(torch.equal(mo,m))

    def test_loss_and_gradients_are_close_to_fp32_and_adam_states_stay_fp32(self):
        for precision in ('bf16_attention','bf16_bridge'):
            a=bridge('mmdit','bidirectional');b=copy.deepcopy(a)
            g,m=torch.randn(3,8),torch.randn(5,8)
            with execution_policy(grouped_attention=True,precision='fp32'):
                ref=a(g,m,metadata());loss=sum(x.square().mean() for x in ref);loss.backward()
            optimizer=torch.optim.AdamW(b.parameters(),lr=2e-4,betas=(.9,.99),weight_decay=0)
            with execution_policy(grouped_attention=True,precision=precision):
                out=b(g,m,metadata());other=sum(x.square().mean() for x in out);other.backward()
                self.assertLess(float((other-loss).detach().abs()/loss.detach()),.01)
                for p,q in zip(a.parameters(),b.parameters()):
                    relative=(p.grad-q.grad).norm()/p.grad.norm().clamp_min(1e-8)
                    self.assertLess(float(relative),.05)
                torch.nn.utils.clip_grad_norm_(b.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
            self.assertTrue(all(v.dtype==torch.float32 for state in optimizer.state.values()
                                for v in state.values() if isinstance(v,torch.Tensor)))

    def test_context_restores_state_and_rejects_invalid_modes(self):
        original=LocalModalityBridge._forward_frames
        self.assertFalse(hasattr(LocalModalityBridge,'_execution_precision'))
        with execution_policy(grouped_attention=True,precision='bf16_bridge'):
            with execution_policy():self.assertEqual(LocalModalityBridge._execution_precision,'bf16_bridge')
        self.assertIs(LocalModalityBridge._forward_frames,original)
        self.assertFalse(hasattr(LocalModalityBridge,'_execution_precision'))
        for kwargs in (dict(precision='float16'),dict(precision='bf16_attention')):
            with self.assertRaises(ValueError):
                with execution_policy(**kwargs):pass

    def test_bf16_projection_ddp_and_checkpoint_recomputation(self):
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(precision_ddp_worker,args=(str(Path(root)/'gloo'),),nprocs=2,join=True)


if __name__=='__main__':unittest.main()
