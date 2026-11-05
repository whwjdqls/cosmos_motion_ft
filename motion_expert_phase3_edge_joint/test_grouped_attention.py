"""Grouped FP32 parity and gradients for the existing local-attention bridge."""
import copy
from functools import partial
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

import torch
import torch.multiprocessing as mp
from torch.nn import functional as F

from modality_bridge import (
    BRIDGE_PARAMETERIZATIONS, LocalModalityBridge, grouped_attention_plan,
    grouped_joint_attention, temporally_local_attention_mask,
)
from scripts.phase3_performance import execution_policy
from .test_temporally_local_attention import metadata, opened_bridge


def reference(q,k,v,meta,policy):
    mask=temporally_local_attention_mask(meta,policy)
    scores=q@k.transpose(-2,-1)*(q.shape[-1]**-.5)
    scores=scores.masked_fill(~mask,torch.finfo(torch.float32).min)
    return F.softmax(scores,dim=-1)@v


def grouped_ddp_worker(rank,rendezvous):
    from . import test_mmdit_bridge as tests
    tests.execution_policy=partial(execution_policy,grouped_attention=True)
    tests.mmdit_ddp_worker(rank,rendezvous)


class GroupedAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(7)

    def test_full_plan_reconstructs_exact_mask_all_tasks_and_policies(self):
        for mode in ('video2motion','motimg2video','joint'):
            for policy in ('directional','bidirectional'):
                meta=metadata(mode)
                plan=grouped_attention_plan(meta,'cpu',policy)
                full=torch.zeros(1698,1698,dtype=torch.bool)
                nq=plan.query_indexes.shape[1]
                for query,position in enumerate(plan.restore_indexes.tolist()):
                    block,row=divmod(position,nq)
                    keys=plan.key_indexes[block][plan.bias[block,0,row]==0]
                    full[query,keys]=True
                self.assertTrue(torch.equal(full,temporally_local_attention_mask(meta,policy)))
                self.assertEqual(plan.bias.shape,(26,1,68,68))

    def test_attention_and_input_gradients_match_dense_reference(self):
        for mode in ('video2motion','motimg2video','joint'):
            for policy in ('directional','bidirectional'):
                meta=metadata(mode,groups=4,spatial=3)
                n=meta.gen_frame.numel()+meta.motion_frame.numel()
                a=[torch.randn(2,n,8,requires_grad=True) for _ in range(3)]
                b=[x.detach().clone().requires_grad_() for x in a]
                expected=reference(*a,meta,policy)
                actual=grouped_joint_attention(*b,meta,policy)
                torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
                expected.square().sum().backward();actual.square().sum().backward()
                for x,y in zip(a,b):
                    torch.testing.assert_close(x.grad,y.grad,rtol=5e-5,atol=5e-6)

    def test_entire_bridge_parameters_and_input_gradients_match(self):
        for mode in ('video2motion','motimg2video','joint'):
            for policy in ('directional','bidirectional'):
                for kind in BRIDGE_PARAMETERIZATIONS:
                    meta=metadata(mode,groups=3,spatial=2)
                    a=opened_bridge(kind,policy);b=copy.deepcopy(a)
                    inputs=[torch.randn(6,8,requires_grad=True),torch.randn(10,8,requires_grad=True)]
                    other=[x.detach().clone().requires_grad_() for x in inputs]
                    with execution_policy():expected=a(*inputs,meta)
                    with execution_policy(grouped_attention=True):actual=b(*other,meta)
                    for x,y in zip(expected,actual):torch.testing.assert_close(x,y,rtol=2e-5,atol=2e-6)
                    sum(x.square().sum() for x in expected).backward()
                    sum(x.square().sum() for x in actual).backward()
                    for x,y in zip([*a.parameters(),*inputs],[*b.parameters(),*other]):
                        torch.testing.assert_close(x.grad,y.grad,rtol=5e-5,atol=5e-6)

    def test_arbitrary_order_sparse_groups_and_unknown_tokens(self):
        for shape in (False,True):
            meta=metadata(groups=3,spatial=2,shape=shape)
            gp=torch.randperm(len(meta.gen_frame));mp_=torch.randperm(len(meta.motion_frame))
            for key in ('gen_frame','gen_clean','gen_source_start','gen_source_end'):
                setattr(meta,key,getattr(meta,key)[gp])
            for key in ('motion_frame','motion_clean'):setattr(meta,key,getattr(meta,key)[mp_])
            meta.gen_frame[0]=-1
            n=len(gp)+len(mp_)
            q,k,v=[torch.randn(2,n,8) for _ in range(3)]
            for policy in ('directional','bidirectional'):
                torch.testing.assert_close(grouped_joint_attention(q,k,v,meta,policy),
                    reference(q,k,v,meta,policy),rtol=2e-5,atol=2e-6)

    def test_grouped_path_never_materializes_dense_attention_mask(self):
        bridge=opened_bridge();meta=metadata(groups=3,spatial=2)
        before=set(bridge.state_dict())
        with patch.object(LocalModalityBridge,'_attention_mask',side_effect=AssertionError('dense mask')):
            with patch('scripts.phase3_performance.attention_bias',side_effect=AssertionError('dense cache')):
                with execution_policy(grouped_attention=True):
                    # Existing trainers enter a cached policy of their own.
                    with execution_policy():
                        bridge(torch.randn(6,8),torch.randn(10,8),meta)
        self.assertEqual(before,set(bridge.state_dict()))

    def test_context_restores_dense_path_and_plan_cache_is_content_keyed(self):
        original=LocalModalityBridge._joint_attention
        meta=metadata('joint');other=metadata('joint')
        a=grouped_attention_plan(meta,'cpu','directional')
        self.assertIs(a,grouped_attention_plan(other,'cpu','directional'))
        self.assertIs(a,grouped_attention_plan(meta,'cpu','directional'))
        self.assertIsNot(a,grouped_attention_plan(metadata('video2motion'),'cpu','directional'))
        with execution_policy(grouped_attention=True):self.assertIsNot(original,LocalModalityBridge._joint_attention)
        self.assertIs(original,LocalModalityBridge._joint_attention)

    def test_attention_is_fp32_even_inside_autocast(self):
        meta=metadata(groups=3,spatial=2)
        q,k,v=[torch.randn(2,16,8) for _ in range(3)]
        expected=reference(q,k,v,meta,'bidirectional')
        with torch.autocast('cpu',dtype=torch.bfloat16):
            actual=grouped_joint_attention(q,k,v,meta,'bidirectional')
        self.assertEqual(actual.dtype,torch.float32)
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)

    def test_custom_bias_is_rejected_instead_of_ignored(self):
        bridge=opened_bridge();meta=metadata(groups=3,spatial=2)
        q,k,v=[torch.randn(2,16,4) for _ in range(3)]
        with execution_policy(grouped_attention=True):
            with self.assertRaisesRegex(ValueError,'custom attention bias'):
                bridge._joint_attention(q,k,v,meta,attn_bias=torch.zeros(16,16))

    def test_grouped_ddp_and_checkpoint_recomputation(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(grouped_ddp_worker,args=(str(Path(directory)/'gloo'),),nprocs=2,join=True)


if __name__=='__main__':unittest.main()
