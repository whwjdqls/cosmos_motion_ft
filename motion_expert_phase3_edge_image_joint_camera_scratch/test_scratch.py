"""Training scope, complete recovery and absence of learned Phase2 tensors."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from motion_expert_phase3_edge_image_joint_camera.policy import configure as camera_configure
from motion_expert_phase3_edge_joint.settings import RECIPE
from .model import require_metadata_only, state_digest
from .policy import INITIALIZATION, configure
from .state import trainable_parameters, trainable_state, load_trainable_state, new_optimizer, frozen_versions, assert_frozen


class TinyMotion(nn.Linear):
    motion_state_dict = nn.Linear.state_dict
    load_motion_state_dict = nn.Linear.load_state_dict


class ScratchTests(unittest.TestCase):
    def setUp(self):
        self.recipe = copy.deepcopy(RECIPE)
        RECIPE.pop('motion_initialization', None)

    def tearDown(self):
        RECIPE.clear(); RECIPE.update(self.recipe)

    def test_only_initialization_and_training_scope_change_recipe(self):
        expected = copy.deepcopy(camera_configure())
        actual = copy.deepcopy(configure())
        actual.pop('motion_initialization')
        actual['experiment'] = expected['experiment']
        actual['camera_prediction']['trainable_parameters'] = expected['camera_prediction']['trainable_parameters']
        self.assertEqual(actual, expected)
        self.assertEqual(actual['camera_prediction']['loss_weight'], 0)

    def test_architecture_input_rejects_learned_state(self):
        configure()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'metadata.pt'
            value = dict(contract={'hidden_size':2048},scratch_initialization=INITIALIZATION,
                         reference_contract_sha256='reference')
            torch.save(value,path)
            self.assertEqual(require_metadata_only(path), value)
            torch.save({**value,'motion_state':{'weight':torch.ones(1)}},path)
            with self.assertRaises(ValueError):require_metadata_only(path)
            torch.save({**value,'contract':{'hidden_size':torch.tensor(2048)}},path)
            with self.assertRaises(ValueError):require_metadata_only(path)

    def test_optimizer_and_recovery_cover_motion_and_bridges(self):
        configure()
        model = SimpleNamespace(bridges=nn.Linear(3,3),motion=TinyMotion(3,3),native=nn.Linear(3,3).requires_grad_(False))
        versions = frozen_versions(model)
        before = state_digest(trainable_state(model))
        optimizer = new_optimizer(model)
        expected_ids = {id(p) for p in model.bridges.parameters()}|{id(p) for p in model.motion.parameters()}
        self.assertEqual({id(p) for g in optimizer.param_groups for p in g['params']},expected_ids)
        loss = model.bridges(torch.ones(2,3)).square().mean()+model.motion(torch.ones(2,3)).square().mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(trainable_parameters(model),1.);optimizer.step()
        assert_frozen(model,versions)
        saved = {k:v.detach().clone() for k,v in trainable_state(model).items()}
        self.assertNotEqual(before,state_digest(saved))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.pt'
            torch.save(dict(state=saved,optimizer=optimizer.state_dict()),path)
            restored=torch.load(path,weights_only=False)
            with torch.no_grad():
                for p in trainable_parameters(model):p.zero_()
            load_trainable_state(model,restored['state']);optimizer.load_state_dict(restored['optimizer'])
            self.assertEqual(state_digest(trainable_state(model)),state_digest(saved))
            with self.assertRaises(ValueError):load_trainable_state(model,{k:v for k,v in saved.items() if k.startswith('bridges.')})
        with torch.no_grad():next(model.native.parameters()).add_(1)
        with self.assertRaises(RuntimeError):assert_frozen(model,versions)

    def test_zero_initialized_decoder_must_train_to_transmit_motion_gradient(self):
        hidden=nn.Linear(3,4)
        decoder=nn.Linear(4,2)
        nn.init.zeros_(decoder.weight);nn.init.zeros_(decoder.bias)
        optimizer=torch.optim.AdamW([*hidden.parameters(),*decoder.parameters()],lr=1e-3)
        def loss():return (decoder(hidden(torch.ones(2,3)))-1).square().mean()
        loss().backward()
        self.assertEqual(hidden.weight.grad.abs().max().item(),0)
        self.assertGreater(decoder.weight.grad.abs().max().item(),0)
        optimizer.step();optimizer.zero_grad();loss().backward()
        self.assertGreater(hidden.weight.grad.abs().max().item(),0)

    def test_bfloat16_and_scalar_state_digests_are_supported(self):
        values={'motion':torch.randn(3,4,dtype=torch.bfloat16),'gate':torch.zeros(())}
        self.assertEqual(state_digest(values),state_digest({k:v.clone() for k,v in values.items()}))


if __name__=='__main__':unittest.main()
