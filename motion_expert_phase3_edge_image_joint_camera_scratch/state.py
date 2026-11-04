"""Optimizer and complete trainable state, excluding the frozen Cosmos model."""
import torch
from motion_expert_phase3_edge_joint.settings import RECIPE


def trainable_parameters(model):
    yield from model.bridges.parameters()
    yield from model.motion.parameters()


def trainable_state(model):
    return {**{'bridges.'+k: v for k,v in model.bridges.state_dict().items()},
            **{'motion.'+k: v for k,v in model.motion.motion_state_dict().items()}}


def load_trainable_state(model, state):
    expected = set(trainable_state(model))
    if set(state) != expected:
        raise ValueError('Incomplete bridge+motion state')
    model.bridges.load_state_dict({k[8:]:v for k,v in state.items() if k.startswith('bridges.')}, strict=True)
    model.motion.load_motion_state_dict({k[7:]:v for k,v in state.items() if k.startswith('motion.')}, strict=True)


def frozen_versions(model):
    return {id(p):p._version for p in model.native.parameters()}


def assert_frozen(model, versions):
    for p in model.native.parameters():
        if p.requires_grad or p.grad is not None or p._version != versions[id(p)]:
            raise RuntimeError('Frozen Phase1 Cosmos parameter changed or received gradients')


def new_optimizer(model):
    return torch.optim.AdamW(trainable_parameters(model), lr=RECIPE['lr'],
        betas=tuple(RECIPE['betas']), weight_decay=RECIPE['weight_decay'])
