"""Isolated AdamW clipping experiment; never attaches to or changes a live trainer."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


def parameter_groups(name):
    layer, part = name.split('.', 1)
    gate = part.startswith('gate_')
    kind = 'gates' if gate else 'norm' if part.startswith('norm_') else 'out' if part.startswith('o_') else 'qkv'
    modality = 'video' if '_g' in part else 'motion'
    return ['all', 'gates' if gate else 'core', kind, f'{kind}/{modality}',
            f'layer/{layer}/{kind}', f'projection/{part.split("_")[0]}']


def checkpoint_states(payload, names):
    identifiers = [p for group in payload['optimizer']['param_groups'] for p in group['params']]
    assert len(names) == len(identifiers) == len(payload['bridges'])
    assert names == list(payload['bridges'])
    return {name: payload['optimizer']['state'][pid] for name, pid in zip(names, identifiers)}


class Arm:
    def __init__(self, payload, *, name, device, cold=False, scales=None, scale_epsilon=False):
        self.name = name
        self.parameters = OrderedDict((key, torch.nn.Parameter(value.to(device=device, copy=True)))
                                      for key, value in payload['bridges'].items())
        assert len(payload['optimizer']['param_groups']) == 1
        options = {k: copy.deepcopy(v) for k, v in payload['optimizer']['param_groups'][0].items() if k != 'params'}
        groups = []
        for gate in [True, False]:
            key = 'gates' if gate else 'core'
            settings = dict(options)
            settings['params'] = [p for n, p in self.parameters.items() if ('gate_' in n) == gate]
            if scale_epsilon:
                settings['eps'] *= scales[key]
            groups.append(settings)
        self.optimizer = torch.optim.AdamW(groups)
        states = checkpoint_states(payload, list(self.parameters))
        if not cold:
            for key, p in self.parameters.items():
                state = {}
                scale = (scales or {}).get('gates' if 'gate_' in key else 'core', 1.)
                for field, value in states[key].items():
                    if torch.is_tensor(value):
                        # CPU step tensors must also be cloned; optimizer loading
                        # may otherwise alias them across experimental arms.
                        value = value.detach().clone() if field == 'step' else value.to(device=device, copy=True)
                        if field == 'exp_avg':
                            value.mul_(scale)
                        elif field == 'exp_avg_sq':
                            value.mul_(scale * scale)
                    else:
                        value = copy.deepcopy(value)
                    state[field] = value
                self.optimizer.state[p] = state

    @torch.no_grad()
    def install(self, model):
        for key, p in model.bridges.named_parameters():
            p.copy_(self.parameters[key])


def clip(parameters, raw, policy):
    for (name, p), grad in zip(parameters.items(), raw):
        p.grad = grad.detach().clone()
    gates = [p for n, p in parameters.items() if 'gate_' in n]
    core = [p for n, p in parameters.items() if 'gate_' not in n]
    gate_norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in gates]))
    core_norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in core]))
    combined = torch.sqrt(gate_norm.square() + core_norm.square())
    coefficient = lambda value: min(1., 1. / (float(value) + 1e-6))
    if policy == 'global':
        actual = torch.nn.utils.clip_grad_norm_(list(parameters.values()), 1., error_if_nonfinite=True)
        coefficients = {'gates': coefficient(actual), 'core': coefficient(actual)}
    elif policy == 'separate':
        g = torch.nn.utils.clip_grad_norm_(gates, 1., error_if_nonfinite=True)
        c = torch.nn.utils.clip_grad_norm_(core, 1., error_if_nonfinite=True)
        coefficients = {'gates': coefficient(g), 'core': coefficient(c)}
    else:
        raise ValueError(policy)
    return {'policy': policy, 'raw_gate_norm': float(gate_norm), 'raw_core_norm': float(core_norm),
            'raw_global_norm': float(combined), 'coefficients': coefficients,
            'post_clip_global_norm': math.sqrt((float(gate_norm) * coefficients['gates']) ** 2
                                              + (float(core_norm) * coefficients['core']) ** 2)}


@torch.no_grad()
def epsilon_stats(arm):
    result = {}
    for name, p in arm.parameters.items():
        state = arm.optimizer.state.get(p)
        if not state:
            continue
        group = arm.optimizer.param_groups[0 if 'gate_' in name else 1]
        beta2 = group['betas'][1]
        count = max(1., float(state['step']))
        v = state['exp_avg_sq'].flatten()
        sampled = v[::max(1, math.ceil(v.numel() / 2048))].float()
        denom = (sampled / (1 - beta2 ** count)).sqrt()
        influence = group['eps'] / (denom + group['eps'])
        result[name] = {'sampled_coordinates': len(sampled),
                        'fraction_sqrt_vhat_below_eps': float((denom < group['eps']).float().mean()),
                        'mean_epsilon_share_of_denominator': float(influence.mean())}
    return result


@torch.no_grad()
def optimizer_step(arm, raw, policy, lr, *, detailed=True):
    for group in arm.optimizer.param_groups:
        group['lr'] = lr
    clip_stats = clip(arm.parameters, raw, policy)
    before = [p.detach().clone() for p in arm.parameters.values()]
    arm.optimizer.step()
    totals, per_tensor = {}, {}
    for (name, p), old, grad in zip(arm.parameters.items(), before, raw):
        delta = p.detach() - old
        values = torch.stack(((old * old).sum(dtype=torch.float64),
                              (delta * delta).sum(dtype=torch.float64),
                              (delta * grad).sum(dtype=torch.float64),
                              (grad * grad).sum(dtype=torch.float64),
                              delta.count_nonzero().to(torch.float64))).cpu().tolist()
        if not all(math.isfinite(x) for x in values):
            raise FloatingPointError(f'Nonfinite parameter/update in {arm.name}/{name}')
        w2, d2, dot, g2, changed = values
        if detailed:
            per_tensor[name] = {'weight_norm': math.sqrt(w2), 'delta_norm': math.sqrt(d2),
                                'update_to_weight': math.sqrt(d2 / w2) if w2 else None,
                                'fraction_coordinates_changed': changed / p.numel(),
                                'descent_cosine': -dot / math.sqrt(d2 * g2) if d2 * g2 else None}
        for category in set(parameter_groups(name)):
            entry = totals.setdefault(category, [0., 0., 0., 0., 0., 0.])
            for i, value in enumerate(values + [p.numel()]):
                entry[i] += value
    groups = {}
    for key, (w2, d2, dot, g2, changed, elements) in totals.items():
        groups[key] = {'weight_norm': math.sqrt(w2), 'delta_norm': math.sqrt(d2),
                       'update_to_weight': math.sqrt(d2 / w2) if w2 else None,
                       'fraction_coordinates_changed': changed / elements,
                       'descent_cosine': -dot / math.sqrt(d2 * g2) if d2 * g2 else None}
    arm.optimizer.zero_grad(set_to_none=True)
    return dict(arm=arm.name, clipping=clip_stats, groups=groups, parameters=per_tensor)


def restore_model_rng(state):
    torch.random.set_rng_state(state[0])
    torch.cuda.set_rng_state(state[1])


def native_gradient(model, batches, loss_fn, mean, std):
    model.train()
    model.zero_grad(set_to_none=True)
    scores = {'total': 0., 'motion': 0., 'video': 0., 'motion_count': 0, 'video_count': 0}
    versions = [{key: value._version for key, value in inputs.items() if torch.is_tensor(value)}
                for inputs, _ in batches]
    for (inputs, target), version in zip(batches, versions):
        output = model(**inputs)
        loss, metrics = loss_fn(output, target, inputs, mean, std)
        assert torch.isfinite(loss)
        (loss / len(batches)).backward()
        scores['total'] += float(loss.detach()) / len(batches)
        for name in ['motion', 'video']:
            count = int(metrics[name + '_count'])
            scores[name] += float(metrics[name]) * count
            scores[name + '_count'] += count
        assert all(inputs[key]._version == value for key, value in version.items())
        del output, loss
    for name in ['motion', 'video']:
        scores[name] /= max(1, scores[name + '_count'])
    raw = [p.grad.detach() for p in model.bridges.parameters()]
    assert all(torch.isfinite(g).all() for g in raw)
    return raw, scores


@torch.no_grad()
def evaluate(model, arm, batches, loss_fn, mean, std):
    arm.install(model)
    model.eval()
    total = motion = video = 0.
    for inputs, target in batches:
        out = model(**inputs)
        loss, metrics = loss_fn(out, target, inputs, mean, std)
        total += float(loss) / len(batches)
        motion += float(metrics['motion']) / len(batches)
        video += float(metrics['video']) / len(batches)
    return dict(total=total, motion=motion, video=video)


def controls(payload, raw, device, lr):
    scratch = Arm(payload, name='control_global', device=device)
    clipped_global = clip(scratch.parameters, raw, 'global')
    a = clipped_global['coefficients']
    clipped_separate = clip(scratch.parameters, raw, 'separate')
    b = clipped_separate['coefficients']
    scales = {key: b[key] / a[key] for key in a}
    answer = {'fixed_group_scales': scales, 'warm_scale_matched': {}, 'cold_constant_gradient': {}}
    base = optimizer_step(scratch, raw, 'global', lr)
    baseline_parameters = {n: p.detach().clone() for n, p in scratch.parameters.items()}
    for scaled_eps in [False, True]:
        arm = Arm(payload, name='matched_eps_scaled' if scaled_eps else 'matched_actual_eps',
                  device=device, scales=scales, scale_epsilon=scaled_eps)
        receipt = optimizer_step(arm, raw, 'separate', lr)
        error = math.sqrt(sum(float((p.detach()-baseline_parameters[n]).square().sum(dtype=torch.float64))
                              for n,p in arm.parameters.items()))
        errors = {}
        for name,p in arm.parameters.items():
            value = float((p.detach()-baseline_parameters[name]).square().sum(dtype=torch.float64))
            for category in set(parameter_groups(name)):
                errors[category] = errors.get(category,0.) + value
        relative = {key: math.sqrt(value)/base['groups'][key]['delta_norm']
                    if base['groups'][key]['delta_norm'] else None for key,value in errors.items()}
        answer['warm_scale_matched'][arm.name] = {'step': receipt, 'post_step_parameter_error_l2_vs_global': error,
                                                'difference_relative_to_global_update_by_group': relative}
        del arm
    answer['warm_scale_matched']['global'] = base
    del scratch, baseline_parameters
    cold = [Arm(payload, name='cold_global', device=device, cold=True),
            Arm(payload, name='cold_separate', device=device, cold=True)]
    for i in range(32):
        for arm, policy in zip(cold, ['global', 'separate']):
            receipt = optimizer_step(arm, raw, policy, lr, detailed=i in [0,31])
            if i in [0,31]:
                answer['cold_constant_gradient'][arm.name + f'_step{i+1}'] = receipt
    return answer


def main():
    parser = argparse.ArgumentParser()
    for key in ['snapshot','run-dir','checkpoint','phase1','phase2','cache','cases','out']:
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--microbatch', type=int, default=4)
    args = parser.parse_args()
    assert args.out.resolve().is_relative_to('/mnt/local') and 32 % args.microbatch == 0
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for suffix in reversed(('motion_expert_t2m_edge','motion_expert_joint_attention','nymeria_world','')):
        sys.path.insert(0,str(args.snapshot/suffix))
    from motion_expert_phase3_edge_joint.settings import RECIPE, digest_json, atomic_json, sha256
    from motion_expert_phase3_edge_image_joint.train import configure_local_paths
    from motion_expert_phase3_edge_image_joint.model import EdgePhase3
    from motion_expert_phase3_edge_joint.adapters import load_trainable_state
    from motion_expert_phase3_edge_joint.data import Phase3Dataset, GlobalDrawSampler, collate
    from motion_expert_phase3_edge_joint.objectives import noise_batch, specialist_loss
    from motion_expert_phase3_edge_joint.runtime import execution_context
    from motion_expert_phase3_edge.quantitative import check_bundle
    from motion_expert_phase3_edge_joint.train import lr_for_step, frozen_versions, assert_frozen
    import config
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    contract = json.loads((args.run_dir/'contract.json').read_text())
    assert payload['fingerprint'] == digest_json(contract)
    assert payload['scheduler']['recipe'] == contract['recipe']
    RECIPE.clear(); RECIPE.update(contract['recipe'])
    assert RECIPE['weight_decay'] == 0 and RECIPE['caption_policy'] == {'video2motion':'empty','motimg2video':'empty'}
    assert sha256(args.snapshot/'motion_expert_joint_attention/modality_bridge.py') == contract['shared_bridge']
    assert sha256(args.snapshot/'scripts/phase3_performance.py') == contract['execution_policy']
    for filename, expected in contract['implementation'].items():
        assert sha256(args.snapshot/'motion_expert_phase3_edge_joint'/filename) == expected, filename
    configure_local_paths()
    torch.manual_seed(190017)
    model = EdgePhase3(args.phase1,args.phase2,device='cuda',activation_checkpointing=True)
    assert payload['sources']['phase2_contract'] == model.provenance['phase2_contract']
    assert payload['sources']['phase1_metadata_sha256'] == model.provenance['phase1_metadata_sha256']
    load_trainable_state(model,payload)
    dataset = Phase3Dataset(args.cache)
    assert dataset.index['fingerprint'] == contract['dataset']
    assert list(dict(model.bridges.named_parameters())) == list(payload['bridges'])
    mean = torch.from_numpy(np.load(config.MOTION_STATS_MEAN)).cuda().float()
    std = torch.from_numpy(np.load(config.MOTION_STATS_STD)).cuda().float()
    arms = [Arm(payload,name='global',device='cuda'), Arm(payload,name='separate_replay',device='cuda'),
            Arm(payload,name='separate_native',device='cuda')]
    for name in payload['bridges']:
        ptrs = [arm.optimizer.state[arm.parameters[name]]['step'].data_ptr() for arm in arms]
        assert len(set(ptrs)) == 3
    frozen = frozen_versions(model)
    records = json.loads(args.cases.read_text())
    test_rows=[]
    for row in records[::4][:16]:
        motion,neutral,video=check_bundle(row,model.provenance['phase2_contract'])
        for mode in ['video2motion','motimg2video']:
            test_rows.append(dict(motion=torch.from_numpy(motion), neutral=torch.from_numpy(neutral),
                                  video=torch.from_numpy(video), mode=mode, caption='',case=row['id'],draw=-1))
    torch.manual_seed(91237)
    test_batches=[noise_batch(collate(test_rows[i:i+args.microbatch]),'cuda') for i in range(0,len(test_rows),args.microbatch)]
    sampler=iter(GlobalDrawSampler(len(dataset),payload['step'],payload['step']+args.steps,args.microbatch,0,1))
    torch.manual_seed(713101)
    header=dict(checkpoint=str(args.checkpoint),fingerprint=payload['fingerprint'],start_step=payload['step'],
                steps=args.steps,global_batch=32,microbatch=args.microbatch,world_size=1,live_training_modified=False,
                note='Single-GPU gradient accumulation; deterministic matched inputs between arms, not bit-exact live two-rank noise continuation.',
                policies={'global':'one joint norm cap1','separate':'gate cap1 and core cap1; combined norm may exceed1'},
                heldout_case_ids=sorted(set(r['case'] for r in test_rows)),test_windows=len(test_rows)//2,
                torch_version=torch.__version__,script_sha256=sha256(__file__),recipe=RECIPE)
    atomic_json(args.out/'plan.json',header)
    history=(args.out/'metrics.jsonl').open('a',buffering=1)
    with execution_context(RECIPE['bridge_execution']):
        baseline=evaluate(model,arms[0],test_batches,specialist_loss,mean,std)
        atomic_json(args.out/'heldout_baseline.json',baseline)
        for offset in range(args.steps):
            start=time.monotonic();step=payload['step']+offset+1
            batches=[];case_ids=[];counts={}
            for _ in range(32//args.microbatch):
                raw=collate([dataset[key] for key in next(sampler)])
                assert all(not c for c in raw['captions'])
                for mode in raw['modes']:counts[mode]=counts.get(mode,0)+1
                case_ids.extend(raw['cases']);batches.append(noise_batch(raw,'cuda'))
            assert all(not c for inputs,_ in batches for c in inputs['captions'])
            rng=(torch.random.get_rng_state(),torch.cuda.get_rng_state())
            arms[0].install(model)
            raw_global,loss_global=native_gradient(model,batches,specialist_loss,mean,std)
            after=(torch.random.get_rng_state(),torch.cuda.get_rng_state())
            if offset==0:
                measured=controls(payload,raw_global,'cuda',lr_for_step(step))
                atomic_json(args.out/'adam_scale_controls.json',measured)
            row=dict(step=step,offset=offset,case_ids=case_ids,task_counts=counts,arms={})
            for arm,policy in zip(arms[:2],['global','separate']):
                if offset%16==0:row.setdefault('epsilon',{})[arm.name]=epsilon_stats(arm)
                row['arms'][arm.name]=optimizer_step(arm,raw_global,policy,lr_for_step(step))
            del raw_global
            restore_model_rng(rng)
            arms[2].install(model)
            raw_native,loss_native=native_gradient(model,batches,specialist_loss,mean,std)
            if offset==0:
                # Both model branches start from exactly the same weights/input/RNG.
                assert abs(loss_global['total']-loss_native['total'])<1e-6
            row['arms'][arms[2].name]=optimizer_step(arms[2],raw_native,'separate',lr_for_step(step))
            row['losses']={'global':loss_global,'separate_native':loss_native}
            restore_model_rng(after)
            assert_frozen(model,frozen)
            if (offset+1)%32==0 or offset+1==args.steps:
                row['heldout']={arm.name:evaluate(model,arm,test_batches,specialist_loss,mean,std) for arm in arms}
            row['seconds']=time.monotonic()-start
            history.write(json.dumps(row)+'\n')
            atomic_json(args.out/'status.json',dict(step=step,offset=offset+1,seconds=row['seconds'],stage='probing'))
            if offset%8==0:
                print('[clip-probe]',offset+1,'seconds',round(row['seconds'],2),'global norm',row['arms']['global']['clipping']['raw_global_norm'],
                      'QKV deltas',{k:v['groups']['qkv']['update_to_weight'] for k,v in row['arms'].items()},flush=True)
            del batches,raw_native
        assert all(int(arm.optimizer.state[p]['step'])==payload['step']+args.steps for arm in arms for p in arm.parameters.values())
        assert all(int(state['step'])==payload['step'] for state in payload['optimizer']['state'].values())
        for arm in [arms[0],arms[2]]:
            # Isolated forks only, kept locally for reproducibility/review.
            torch.save(dict(bridges={n:p.detach().cpu() for n,p in arm.parameters.items()},optimizer=arm.optimizer.state_dict(),
                            start_checkpoint=str(args.checkpoint),step=payload['step']+args.steps,policy=arm.name),args.out/(arm.name+'_isolated_final.pt'))
    history.close()
    atomic_json(args.out/'COMPLETE.json',dict(steps=args.steps,live_training_modified=False,frozen_specialists_unchanged=True,
                                            completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    print('[clip-probe] COMPLETE',args.out,flush=True)


if __name__=='__main__':
    main()
