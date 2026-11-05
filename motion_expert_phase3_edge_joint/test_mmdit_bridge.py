"""Private modality weights, joint attention, cache parity and complete CPU/DDP recovery."""
import argparse
import copy
from dataclasses import replace
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

from modality_bridge import BRIDGE_PARAMETERIZATIONS, BridgeMeta, LocalModalityBridge
from scripts.phase3_performance import attention_bias, execution_policy
from . import settings
from .adapters import (
    Phase3LoRA, add_training_options, configure_training, configure_checkpoint,
    load_trainable_state, trainable_parameters, trainable_state, new_optimizer,
    new_ema, restore_ema,
)
from .test_adapters import TinyNative, saved_payload


def metadata(mode='joint'):
    return BridgeMeta(mode, torch.tensor([0, 1, 1]),
        torch.ones(3, dtype=torch.bool) if mode == 'video2motion' else torch.tensor([True, False, False]),
        torch.tensor([-1, 0, 1, 4, 5]),
        torch.ones(5, dtype=torch.bool) if mode == 'motimg2video' else torch.tensor([True, False, False, False, False]),
        torch.tensor([0, 1, 1]), torch.tensor([0, 4, 4]))


def configure(mode, *, lora=False, ema=False):
    parser = add_training_options(settings.add_bridge_options(argparse.ArgumentParser()))
    flags = ['--bridge-parameterization', mode, '--bridge-lr', '.001']
    if lora:
        flags += ['--phase3-lora-rank', '2', '--phase3-lora-lr', '.0002']
    if ema:
        flags += ['--ema-decay', '.9']
    configure_training(parser.parse_args(flags))


class BridgeOwner(nn.Module):
    def __init__(self, mode, attention='bidirectional'):
        super().__init__()
        object.__setattr__(self, 'native', TinyNative())
        self.bridges = nn.ModuleDict({'0': LocalModalityBridge(8, 2, 4, attention, mode)})
        self.phase3_lora = Phase3LoRA(self.native, settings.RECIPE.get('phase3_lora'), expected_layers=2)

    def forward(self, gen, motion, mode='joint', checkpointed=False):
        gen = self.native(gen)
        def bridge_step(g, m):
            return self.bridges['0'](g, m, metadata(mode))
        result = checkpoint(bridge_step, gen, motion, use_reentrant=False) if checkpointed else bridge_step(gen, motion)
        anchor = sum(p.flatten()[0] * 0 for p in trainable_parameters(self))
        return result[0] + anchor, result[1] + anchor


def update(model, optimizer, ema, gen, motion, *, mode='joint', checkpointed=False):
    optimizer.zero_grad(set_to_none=True)
    g, m = model(gen, motion, mode, checkpointed)
    loss = (g - .4).square().mean() if mode == 'motimg2video' else (m + .2).square().mean()
    if mode == 'joint':
        loss = loss + (g - .4).square().mean()
    loss.backward()
    owner = model.module if isinstance(model, DDP) else model
    torch.nn.utils.clip_grad_norm_(trainable_parameters(owner), 1.)
    optimizer.step()
    if ema:
        ema.update(owner)


def mmdit_ddp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    try:
        configure('mmdit', ema=True)
        torch.manual_seed(100 + rank)
        model = BridgeOwner('mmdit')
        for p in model.native.parameters():
            dist.broadcast(p, 0)
        ddp = DDP(model, broadcast_buffers=False, find_unused_parameters=False)
        optimizer, ema = new_optimizer(model), new_ema(model)
        gen, motion = torch.randn(3, 8), torch.randn(5, 8)
        # Different target directions exercise the otherwise unused output
        # projection/gate on each rank, as real homogeneous task batches do.
        with execution_policy():
            for _ in range(3):
                update(ddp, optimizer, ema, gen, motion,
                       mode='video2motion' if rank == 0 else 'motimg2video', checkpointed=True)
        for p in trainable_parameters(model):
            other = p.detach().clone(); dist.broadcast(other, 0)
            assert torch.equal(p, other)
            assert p.grad is not None
        for state in ema.state.values():
            for value in state.values():
                other = value.clone(); dist.broadcast(other, 0)
                assert torch.equal(value, other)
        for role in ('g', 'm'):
            for name in ('norm', 'q', 'k', 'v', 'o'):
                p = getattr(model.bridges['0'], f'{name}_{role}').weight
                assert p.grad.abs().max() > 0, f'{name}_{role} received no gradient'
        assert all(p.grad is None for p in model.native.parameters())
    finally:
        dist.destroy_process_group()


class MMDiTBridgeTests(unittest.TestCase):
    def setUp(self):
        self.recipe = copy.deepcopy(settings.RECIPE)
        torch.set_num_threads(1)
        torch.manual_seed(71)

    def tearDown(self):
        settings.RECIPE.clear(); settings.RECIPE.update(self.recipe)

    def test_shared_parameters_forward_backward_match_with_same_new_mask(self):
        # Preserve legacy math on frame-only inputs. Shape bypass is tested
        # separately, since it deliberately changes nonzero-gate behavior.
        repo = Path(__file__).resolve().parents[1]
        source = subprocess.check_output(['git', '-C', str(repo), 'show',
            'HEAD:motion_expert_joint_attention/modality_bridge.py'], text=True)
        legacy = types.ModuleType('_bridge_before_mmdit'); sys.modules[legacy.__name__] = legacy
        exec(compile(source, 'bridge_before_mmdit.py', 'exec'), legacy.__dict__)
        for policy in ('directional', 'bidirectional'):
            for mode in ('video2motion', 'motimg2video', 'joint'):
                torch.manual_seed(3); old = legacy.LocalModalityBridge(8, 2, 4, policy)
                torch.manual_seed(3); new = LocalModalityBridge(8, 2, 4, policy)
                old._attention_mask = new._attention_mask
                self.assertEqual(set(old.state_dict()), set(new.state_dict()))
                for p, q in zip(old.parameters(), new.parameters()):
                    self.assertTrue(torch.equal(p, q))
                with torch.no_grad():
                    old.gate_g.fill_(.2); old.gate_m.fill_(.3)
                    new.load_state_dict(old.state_dict())
                g, m = torch.randn(3, 8), torch.randn(4, 8)
                meta=metadata(mode)
                meta=replace(meta,motion_frame=meta.motion_frame[1:],motion_clean=meta.motion_clean[1:])
                a, b = old(g, m, meta), new(g, m, meta)
                for x, y in zip(a, b):
                    self.assertTrue(torch.equal(x, y))
                sum(x.square().sum() for x in a).backward()
                sum(x.square().sum() for x in b).backward()
                for p, q in zip(old.parameters(), new.parameters()):
                    self.assertTrue(torch.equal(p.grad, q.grad))

    def test_three_stages_have_only_the_requested_sharing(self):
        self.assertEqual(BRIDGE_PARAMETERIZATIONS, settings.BRIDGE_PARAMETERIZATIONS)
        for mode in BRIDGE_PARAMETERIZATIONS[1:]:
            bridge = LocalModalityBridge(8, 2, 4, parameterization=mode)
            self.assertFalse(hasattr(bridge, 'norm'))
            self.assertNotEqual(bridge.norm_g.weight.data_ptr(), bridge.norm_m.weight.data_ptr())
            self.assertNotEqual(bridge.norm_g.bias.data_ptr(), bridge.norm_m.bias.data_ptr())
            for name in ('q', 'k', 'v'):
                self.assertEqual(hasattr(bridge, name), mode != 'mmdit')
                if mode == 'mmdit':
                    self.assertNotEqual(getattr(bridge,name+'_g').weight.data_ptr(),
                                        getattr(bridge,name+'_m').weight.data_ptr())
            self.assertEqual(hasattr(bridge, 'o'), mode == 'separate_norm')
            if mode != 'separate_norm':
                self.assertNotEqual(bridge.o_g.weight.data_ptr(), bridge.o_m.weight.data_ptr())
        with self.assertRaises(ValueError):
            LocalModalityBridge(8, 2, 4, parameterization='bad')

    def test_zero_gates_are_exact_identity_for_all_tasks_and_bfloat16_inputs(self):
        for mode in BRIDGE_PARAMETERIZATIONS:
            bridge = LocalModalityBridge(8, 2, 4, 'bidirectional', mode)
            for dtype in (torch.float32, torch.bfloat16):
                g, m = torch.randn(3, 8).to(dtype), torch.randn(5, 8).to(dtype)
                for task in ('video2motion', 'motimg2video', 'joint'):
                    a, b = bridge(g, m, metadata(task))
                    self.assertTrue(torch.equal(a, g)); self.assertTrue(torch.equal(b, m))
                    self.assertEqual(a.dtype, g.dtype); self.assertEqual(b.dtype, m.dtype)

    def test_mmdit_matches_independent_joint_attention_equations(self):
        # Head projection width differs from hidden width to catch output slicing errors.
        bridge = LocalModalityBridge(8, 2, 3, 'bidirectional', 'mmdit')
        with torch.no_grad():
            bridge.gate_g.fill_(.2); bridge.gate_m.fill_(-.3)
        g, m = torch.randn(3, 8), torch.randn(5, 8)
        hg, hm = bridge.norm_g(g), bridge.norm_m(m[1:])
        def project(name):
            a = getattr(bridge, name+'_g')(hg).reshape(3, 2, 3).permute(1,0,2)
            b = getattr(bridge, name+'_m')(hm).reshape(4, 2, 3).permute(1,0,2)
            return torch.cat((a,b), dim=1)
        q, k, v = [project(name) for name in ('q', 'k', 'v')]
        scores = torch.einsum('hid,hjd->hij', q, k) / (3 ** .5)
        meta=metadata()
        meta=replace(meta,motion_frame=meta.motion_frame[1:],motion_clean=meta.motion_clean[1:])
        scores.masked_fill_(~bridge._attention_mask(meta), torch.finfo(torch.float32).min)
        y = torch.einsum('hij,hjd->hid', scores.softmax(-1), v).permute(1,0,2).reshape(7,6)
        expected = g + .2*bridge.o_g(y[:3]), torch.cat((m[:1],m[1:] - .3*bridge.o_m(y[3:])))
        for a, b in zip(bridge(g,m,metadata()), expected):
            torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)

    def test_mmdit_parameters_route_to_their_own_modality(self):
        bridge = LocalModalityBridge(8, 2, 4, 'bidirectional', 'mmdit')
        with torch.no_grad():
            bridge.gate_g.fill_(.2); bridge.gate_m.fill_(.3)
        calls = []
        handles = []
        for name in ('norm_g','norm_m','q_g','q_m','k_g','k_m','v_g','v_m','o_g','o_m'):
            handles.append(getattr(bridge,name).register_forward_pre_hook(
                lambda _module,args,label=name: calls.append((label,args[0].shape[0]))))
        bridge(torch.randn(3,8),torch.randn(5,8),metadata())
        for handle in handles:
            handle.remove()
        self.assertEqual(dict(calls), {name:3 if name.endswith('_g') else 4 for name,_ in calls})
        self.assertEqual(len(calls),10)

    def test_cached_policy_matches_outputs_gradients_and_attention_masks(self):
        for parameterization in BRIDGE_PARAMETERIZATIONS:
            for policy in ('directional', 'bidirectional'):
                for task in ('video2motion', 'motimg2video', 'joint'):
                    a=LocalModalityBridge(8,2,4,policy,parameterization)
                    with torch.no_grad():
                        a.gate_g.fill_(.2); a.gate_m.fill_(.3)
                    b=copy.deepcopy(a); g,m=torch.randn(3,8),torch.randn(5,8); meta=metadata(task)
                    expected=a(g,m,meta)
                    with execution_policy():
                        actual=b(g,m,meta)
                    for x,y in zip(expected,actual):
                        self.assertTrue(torch.equal(x,y))
                    sum(x.square().sum() for x in expected).backward()
                    sum(x.square().sum() for x in actual).backward()
                    for x,y in zip(a.parameters(),b.parameters()):
                        self.assertTrue(torch.equal(x.grad,y.grad))
                    self.assertTrue(torch.equal(attention_bias(meta,torch.device('cpu'),policy)==0,a._attention_mask(meta)))

    def test_each_new_mode_has_real_bidirectional_cross_modal_influence(self):
        g,m=torch.randn(3,8),torch.randn(5,8)
        for mode in BRIDGE_PARAMETERIZATIONS[1:]:
            bridge=LocalModalityBridge(8,2,4,'bidirectional',mode)
            with torch.no_grad():
                bridge.gate_g.fill_(.2);bridge.gate_m.fill_(.3)
            a,b=bridge(g,m,metadata())
            gm,mg=g.clone(),m.clone();gm[:,0]+=2;mg[:,0]+=2
            self.assertGreater((bridge(g,mg,metadata())[0]-a).abs().max().item(),1e-7)
            self.assertGreater((bridge(gm,m,metadata())[1]-b).abs().max().item(),1e-7)

    def test_all_private_parameters_receive_gradients_after_gates_open(self):
        bridge=LocalModalityBridge(8,2,4,'bidirectional','mmdit')
        g,m=torch.randn(3,8),torch.randn(5,8)
        def loss():
            a,b=bridge(g,m,metadata());return a.square().sum()+b.square().sum()
        loss().backward()
        self.assertTrue(all(p.grad is not None and p.grad.count_nonzero()==0
                            for name,p in bridge.named_parameters() if not name.startswith('gate_')))
        bridge.zero_grad()
        with torch.no_grad():
            bridge.gate_g.fill_(.2);bridge.gate_m.fill_(.3)
        loss().backward()
        self.assertTrue(all(p.grad is not None and p.grad.abs().max()>0 for p in bridge.parameters()))

    def test_mode_checkpoint_roundtrip_and_legacy_default(self):
        for mode in BRIDGE_PARAMETERIZATIONS:
            configure(mode)
            payload={'scheduler':{'recipe':copy.deepcopy(settings.RECIPE)}}
            configure_checkpoint(payload)
            self.assertEqual(settings.RECIPE.get('bridge_parameterization','shared'),mode)
            with self.assertRaisesRegex(ValueError,'conflicts'):
                configure_checkpoint(payload,bridge_parameterization='mmdit' if mode=='shared' else 'shared')
            model=BridgeOwner(mode)
            payload['bridges']=copy.deepcopy(model.bridges.state_dict())
            load_trainable_state(model,payload)
            other=BridgeOwner('shared' if mode!='shared' else 'mmdit')
            with self.assertRaisesRegex(ValueError,'parameterization'):
                load_trainable_state(other,payload)
        configure_checkpoint({})
        self.assertNotIn('bridge_parameterization',settings.RECIPE)
        self.assertNotIn('phase3_lora',settings.RECIPE)

    def test_mmdit_raw_and_ema_resume_continue_with_optional_lora(self):
        for lora in (False,True):
            configure('mmdit',lora=lora,ema=True)
            model=BridgeOwner('mmdit');optimizer,ema=new_optimizer(model),new_ema(model)
            g,m=torch.randn(3,8),torch.randn(5,8)
            for _ in range(2):update(model,optimizer,ema,g,m,checkpointed=True)
            payload=saved_payload(model,optimizer,ema)
            configure_checkpoint(payload)
            restored=BridgeOwner('mmdit');restored.native.load_state_dict(model.native.state_dict())
            load_trainable_state(restored,payload)
            ropt,rema=new_optimizer(restored),new_ema(restored)
            ropt.load_state_dict(payload['optimizer']);restore_ema(rema,restored,payload)
            update(model,optimizer,ema,g,m,checkpointed=True)
            update(restored,ropt,rema,g,m,checkpointed=True)
            for a,b in zip(trainable_parameters(model),trainable_parameters(restored)):
                self.assertTrue(torch.equal(a,b))
            for name,state in ema.state.items():
                for key,value in state.items():
                    self.assertTrue(torch.equal(value,rema.state[name][key]))
            payload=saved_payload(model,optimizer,ema)
            load_trainable_state(restored,payload,weights='ema')
            for name,state in trainable_state(restored).items():
                for key,value in state.items():
                    self.assertTrue(torch.equal(value,ema.state[name][key]))
            self.assertTrue(all(p.grad is None for p in model.native.parameters()))

    def test_real_two_process_mmdit_ddp(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(mmdit_ddp_worker,args=(str(Path(directory)/'gloo'),),nprocs=2,join=True)


if __name__ == '__main__':
    unittest.main()
