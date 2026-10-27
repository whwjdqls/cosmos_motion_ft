"""Explicitly requested representation and pose-module GPU acceptance checks."""
from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from egocontrol_edge.representation import BODY23, body23_to_pose, from_normalized_motion, matrix_to_euler
from egocontrol_edge.layers import PoseEmbedding, PoseBlock
from motion_expert_phase3_edge.settings import require_compute_node, sha256, atomic_json
from nymeria_baselines.visualize_official_baselines import decode


def official_functions(root):
    namespace = {'np': np, 'math': math}
    for name in ('action_dataset_utils.py', 'pose_dataset_utils.py'):
        path = root / name
        tree = ast.parse(path.read_text())
        wanted = {'alpha2rotm', 'beta2rotm', 'gamma2rotm', 'euler2rotm', 'rotm2euler', 'isRotm',
                  'get_velocity', 'process_body_poses', 'pose_single_joint'}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted
                 or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'FULL_BODY_JOINTS' for t in n.targets)]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    require_compute_node()
    torch.set_num_threads(1)
    cache = Path(os.environ['EGOCONTROL_CACHE'])
    index = json.loads((cache / 'index.json').read_text())
    mean, std = np.load(os.environ['EGOCONTROL_MOTION_MEAN']), np.load(os.environ['EGOCONTROL_MOTION_STD'])
    assert sha256(os.environ['EGOCONTROL_MOTION_MEAN']) == index['source']['mean']
    assert sha256(os.environ['EGOCONTROL_MOTION_STD']) == index['source']['std']
    official = official_functions(args.upstream)
    differences, examples = [], []
    for row in [index['rows'][i] for i in np.linspace(0, len(index['rows']) - 1, 12, dtype=int)]:
        motion = np.load(cache / (row['array'] + '.motion.npy'), mmap_mode='r')[row['row']]
        position, rotation = decode(motion * std + mean, 30)
        position, rotation = position[:, BODY23], rotation[:, BODY23]
        angles = matrix_to_euler(rotation)
        source = {'global_translations': position[:, :, None], 'global_rotations_euler': angles}
        expected = official['process_body_poses'](source, 96, relative_joints=official['FULL_BODY_JOINTS']).reshape(96, 138)
        actual = body23_to_pose(position, rotation)
        error = float(np.max(np.abs(actual - expected)))
        assert error < 2e-4, (row, error)
        differences.append(error)
        examples.append(actual)
    # Small dimensions exercise the same modulation/attention implementation,
    # without competing with the still-running production model's VRAM.
    torch.manual_seed(42)
    embedding = PoseEmbedding(hidden=64, pose_channels=32).cuda()
    block = PoseBlock(hidden=64, pose_channels=32, heads=4).cuda()
    pose = torch.tensor(np.stack(examples[:2]), device='cuda')
    x = torch.randn(2 * 1600, 64, device='cuda')
    context = embedding(pose)
    initial = block.cross(x, context)
    assert torch.equal(initial, x), 'Initial extra cross-attention must be a no-op'
    parameters = list(embedding.parameters()) + list(block.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=8e-5, betas=(.9, .99), weight_decay=.05)
    target = torch.randn_like(x)
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        context = embedding(pose)
        result = embedding.final(block.cross(x, context), context)
        loss = (result - target).square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)
        optimizer.step()
        losses.append(float(loss.detach()))
    assert block.k_proj.weight.grad is not None and block.k_proj.weight.grad.abs().sum() > 0
    with torch.no_grad():
        correct = block.cross(x, embedding(pose))
        swapped = block.cross(x, embedding(pose.flip(0)))
        difference = float((correct - swapped).abs().max())
        assert difference > 0, 'Pose conditioning is inactive'
        e2, b2 = PoseEmbedding(64, 32).cuda(), PoseBlock(64, 32, 4).cuda()
        e2.load_state_dict(copy.deepcopy(embedding.state_dict()), strict=True)
        b2.load_state_dict(copy.deepcopy(block.state_dict()), strict=True)
        assert torch.equal(correct, b2.cross(x, e2(pose)))
    atomic_json(args.output, {'status': 'passed', 'phase3_windows': len(index['rows']),
        'cache_fingerprint': index['fingerprint'], 'official_representation_max_error': max(differences),
        'initial_noop': True, 'gradient_finite': True, 'pose_swap_max_change': difference,
        'module_reload_bitwise': True, 'losses': losses,
        'scope': 'Representation + small-dimension actual modules. Full native DDP train/save/resume is still required.'})
    print('ACCEPTANCE_MODULES_PASSED', args.output, flush=True)


if __name__ == '__main__':
    main()
