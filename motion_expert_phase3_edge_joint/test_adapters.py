"""CPU acceptance of additive deltas, complete recovery and real DDP synchronization."""
import argparse
import copy
import io
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

from .adapters import (
    TARGETS, Phase3LoRA, TrainableEMA, add_training_options, configure_training,
    configure_checkpoint, trainable_parameters, trainable_state, load_trainable_state,
    new_optimizer, new_ema, restore_ema, set_optimizer_lr,
)
from .settings import RECIPE


class Phase1Linear(nn.Linear):
    """Same additive forward/key geometry as NVIDIA LoraInjectedLinear."""
    def __init__(self):
        super().__init__(8, 8)
        self.lora_A = nn.Linear(8, 2, bias=False)
        self.lora_B = nn.Linear(2, 8, bias=False)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias) + 2 * self.lora_B(self.lora_A(x))


class TinyNative(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleDict({name: Phase1Linear() for name in TARGETS})
                                     for _ in range(2)])
        self.requires_grad_(False).eval()

    def forward(self, x):
        for layer in self.layers:
            x = sum(module(x) for module in layer.values()).tanh()
        return x


class TinyPhase3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        object.__setattr__(self, 'native', TinyNative())
        self.bridges = nn.Linear(8, 8)
        self.phase3_lora = Phase3LoRA(self.native, cfg, expected_layers=2)

    def forward(self, x, native_enabled=True, checkpointed=False):
        hidden = (checkpoint(self.native, x, use_reentrant=False) if checkpointed else self.native(x)) if native_enabled else x
        return self.bridges(hidden) + sum(p.flatten()[0] * 0 for p in trainable_parameters(self))


def setup_recipe(*, enabled=True, ema=.9):
    args = ['--bridge-lr', '.001', '--ema-decay', str(ema)]
    if enabled:
        args += ['--phase3-lora-rank', '2', '--phase3-lora-alpha', '4', '--phase3-lora-lr', '.0002']
    configure_training(add_training_options(argparse.ArgumentParser()).parse_args(args))
    return RECIPE.get('phase3_lora')


def saved_payload(model, optimizer, ema):
    buffer = io.BytesIO()
    torch.save(dict(**trainable_state(model), optimizer=optimizer.state_dict(),
                    scheduler={'recipe': copy.deepcopy(RECIPE)},
                    ema=None if ema is None else ema.state_dict()), buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=False)


def update(model, optimizer, ema, x, **kwargs):
    optimizer.zero_grad(set_to_none=True)
    (model(x, **kwargs) - .3).square().mean().backward()
    torch.nn.utils.clip_grad_norm_(trainable_parameters(model.module if isinstance(model, DDP) else model), 1.)
    optimizer.step()
    if ema is not None:
        ema.update(model.module if isinstance(model, DDP) else model)


def ddp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    try:
        cfg = setup_recipe()
        torch.manual_seed(11 + rank)
        model = TinyPhase3(cfg)
        # Native is outside DDP and is identical in real runs loaded from one checkpoint.
        for p in model.native.parameters():
            dist.broadcast(p, 0)
        ddp = DDP(model, broadcast_buffers=False, find_unused_parameters=False)
        ema, optimizer = new_ema(model), new_optimizer(model)
        frozen = {k: v.clone() for k, v in model.native.state_dict().items()}
        for step in range(3):
            # Rank zero has no native loss edge at all; zero anchors must still
            # let DDP synchronize all adapter parameters without unused errors.
            update(ddp, optimizer, ema, torch.full((3, 8), float(rank + 1)),
                   native_enabled=rank == 1, checkpointed=True)
        for p in trainable_parameters(model):
            other = p.detach().clone()
            dist.broadcast(other, 0)
            assert torch.equal(p, other), 'DDP failed to synchronize a trainable parameter'
        assert any(p.grad.abs().max() > 0 for p in model.phase3_lora.parameters())
        for state in ema.state.values():
            for value in state.values():
                other = value.clone(); dist.broadcast(other, 0)
                assert torch.equal(value, other), 'EMA pair diverged across ranks'
        for key, value in model.native.state_dict().items():
            assert torch.equal(value, frozen[key]), 'Phase1 tensor changed'
        assert all(p.grad is None for p in model.native.parameters())
    finally:
        dist.destroy_process_group()


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.recipe = copy.deepcopy(RECIPE)
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def tearDown(self):
        RECIPE.clear(); RECIPE.update(self.recipe)

    def test_zero_delta_bit_exact_and_native_has_no_new_parameters(self):
        cfg = setup_recipe()
        native = TinyNative()
        x = torch.randn(4, 8)
        baseline = native(x)
        old_keys = set(native.state_dict())
        adapters = Phase3LoRA(native, cfg, expected_layers=2)
        self.assertTrue(torch.equal(baseline, native(x)))
        self.assertEqual(set(native.state_dict()), old_keys)
        self.assertEqual(len(adapters), 8)
        self.assertTrue(all(p.dtype == torch.float32 and p.requires_grad for p in adapters.parameters()))
        self.assertTrue(all(not p.requires_grad for p in native.parameters()))
        with self.assertRaisesRegex(RuntimeError, 'already installed'):
            Phase3LoRA(native, cfg, expected_layers=2)

    def test_only_phase3_deltas_and_bridges_receive_gradients(self):
        cfg = setup_recipe()
        model = TinyPhase3(cfg)
        frozen = copy.deepcopy(model.native.state_dict())
        optimizer = new_optimizer(model)
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        self.assertEqual(expected, {id(p) for group in optimizer.param_groups for p in group['params']})
        x = torch.randn(3, 8)
        update(model, optimizer, None, x, checkpointed=True)
        self.assertTrue(any(delta.B.weight.grad.abs().max() > 0 for delta in model.phase3_lora.values()))
        self.assertTrue(all(delta.A.weight.grad.count_nonzero() == 0 for delta in model.phase3_lora.values()))
        update(model, optimizer, None, x, checkpointed=True)
        self.assertTrue(any(delta.A.weight.grad.abs().max() > 0 for delta in model.phase3_lora.values()))
        for key, value in model.native.state_dict().items():
            self.assertTrue(torch.equal(value, frozen[key]))
        self.assertTrue(all(p.grad is None for p in model.native.parameters()))

    def test_raw_and_ema_checkpoint_resume_continue_exactly(self):
        cfg = setup_recipe()
        model = TinyPhase3(cfg)
        optimizer, ema = new_optimizer(model), new_ema(model)
        x = torch.randn(3, 8)
        for _ in range(2):
            update(model, optimizer, ema, x)
        payload = saved_payload(model, optimizer, ema)
        restored = TinyPhase3(cfg)
        restored.native.load_state_dict(model.native.state_dict())
        load_trainable_state(restored, payload)
        ropt, rema = new_optimizer(restored), new_ema(restored)
        ropt.load_state_dict(payload['optimizer']); restore_ema(rema, restored, payload)
        for _ in range(2):
            for opt in (optimizer, ropt):
                set_optimizer_lr(opt, .0007)
                self.assertAlmostEqual(opt.param_groups[1]['lr'], .00014)
            update(model, optimizer, ema, x)
            update(restored, ropt, rema, x)
        for p, q in zip(trainable_parameters(model), trainable_parameters(restored)):
            self.assertTrue(torch.equal(p, q))
        self.assertEqual(ema.num_updates, rema.num_updates)
        for name, state in ema.state.items():
            for key, value in state.items():
                self.assertTrue(torch.equal(value, rema.state[name][key]))
        final = saved_payload(model, optimizer, ema)
        load_trainable_state(restored, final, weights='ema')
        for name, state in trainable_state(restored).items():
            for key, value in state.items():
                self.assertTrue(torch.equal(value, ema.state[name][key]))
        self.assertEqual(set(final), {'bridges','phase3_lora','optimizer','scheduler','ema'})

    def test_ema_tracks_known_bridge_and_lora_values(self):
        model = TinyPhase3(setup_recipe())
        ema = TrainableEMA(model, .5)
        before = copy.deepcopy(ema.state)
        with torch.no_grad():
            for p in trainable_parameters(model):
                p.add_(2.)
        ema.update(model)
        for name, state in ema.state.items():
            for key, value in state.items():
                torch.testing.assert_close(value, before[name][key] + 1.)

    def test_incompatible_or_missing_lora_and_ema_fail(self):
        cfg = setup_recipe()
        model = TinyPhase3(cfg)
        payload = saved_payload(model, new_optimizer(model), None)
        for altered in [dict(payload, phase3_lora={}),
                        {key: value for key, value in payload.items() if key != 'phase3_lora'}]:
            with self.assertRaises(ValueError):
                load_trainable_state(model, altered)
        other = copy.deepcopy(payload)
        other['scheduler']['recipe']['phase3_lora']['alpha'] = 99
        with self.assertRaisesRegex(ValueError, 'configuration'):
            load_trainable_state(model, other)
        with self.assertRaisesRegex(ValueError, 'no EMA'):
            load_trainable_state(model, payload, weights='ema')
        with self.assertRaisesRegex(ValueError, 'EMA configuration'):
            restore_ema(new_ema(model), model, payload)
        configure_checkpoint({})
        with self.assertRaises(ValueError):
            load_trainable_state(TinyPhase3(None), payload)

    def test_disabled_preserves_legacy_recipe_optimizer_and_checkpoint(self):
        RECIPE.pop('phase3_lora', None); RECIPE.pop('trainable_ema', None)
        before = copy.deepcopy(RECIPE)
        configure_training(add_training_options(argparse.ArgumentParser()).parse_args([]))
        self.assertEqual(RECIPE, before)
        model = TinyPhase3(None)
        self.assertEqual(len(new_optimizer(model).param_groups), 1)
        self.assertIsNone(new_ema(model))
        legacy = dict(bridges=copy.deepcopy(model.bridges.state_dict()))
        load_trainable_state(model, legacy)
        configure_checkpoint(legacy)
        self.assertNotIn('phase3_lora', RECIPE)

    def test_bad_configuration_fails_before_training(self):
        parser = add_training_options(argparse.ArgumentParser())
        for args in [['--phase3-lora-rank','-1'], ['--ema-decay','1'], ['--ema-decay','nan'],
                     ['--bridge-lr','0'], ['--bridge-weight-decay','-1'],
                     ['--phase3-lora-rank','2','--phase3-lora-alpha','0'],
                     ['--phase3-lora-rank','2','--phase3-lora-lr','nan']]:
            with self.assertRaises(ValueError):
                configure_training(parser.parse_args(args))

    def test_historical_scratch_complete_state_still_roundtrips(self):
        from motion_expert_phase3_edge_image_joint_camera_scratch.state import checkpoint_state, load_checkpoint_state

        class Motion(nn.Linear):
            motion_state_dict = nn.Linear.state_dict
            load_motion_state_dict = nn.Linear.load_state_dict

        model = TinyPhase3(None)
        model.motion = Motion(8, 8)
        expected = copy.deepcopy(checkpoint_state(model))
        with torch.no_grad():
            model.bridges.weight.add_(1.)
            model.motion.weight.add_(2.)
        load_checkpoint_state(model, expected)
        for name, state in checkpoint_state(model).items():
            for key, value in state.items():
                self.assertTrue(torch.equal(value, expected[name][key]))
        with self.assertRaises(ValueError):
            load_checkpoint_state(model, expected, weights='ema')

    def test_autocast_preserves_fp32_master_and_zero_output(self):
        native = TinyNative().to(torch.bfloat16)
        x = torch.randn(3, 8, dtype=torch.bfloat16)
        expected = native(x)
        adapters = Phase3LoRA(native, setup_recipe(), expected_layers=2)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = native(x)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(all(p.dtype == torch.float32 for p in adapters.parameters()))

    def test_real_two_process_ddp_with_checkpointing_and_unused_task_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(ddp_worker, args=(str(Path(directory)/'gloo'),), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
