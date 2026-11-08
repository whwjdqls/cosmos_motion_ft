"""Read-only probes of an immutable Phase3 snapshot; never updates training weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import types

import numpy as np
import torch


def summary(values):
    x = torch.as_tensor(values).detach().float().flatten().cpu()
    good = x[torch.isfinite(x)]
    if not good.numel():
        return {"count": x.numel(), "nonfinite": x.numel()}
    return {"count": x.numel(), "nonfinite": x.numel() - good.numel(),
            "mean": good.mean().item(), "median": good.median().item(),
            "p95": torch.quantile(good, .95).item(), "max": good.max().item(),
            "min": good.min().item()}


def attention_rows(probabilities, allowed):
    """Summarize actual unpadded rows; exclude forced singleton rows from collapse rates."""
    n = allowed.sum(-1)
    p = probabilities
    entropy = -(p * p.clamp_min(1e-30).log()).sum(-1)
    eligible = n > 1
    top = p.max(-1).values
    return {"rows": len(p), "non_singleton_rows": int(eligible.sum()),
            "normalized_entropy": summary(entropy[eligible] / n[eligible].float().log()),
            "effective_keys": summary(entropy.exp()), "top1": summary(top),
            "fraction_top1_gt_099": float((top[eligible] > .99).float().mean()) if eligible.any() else None,
            "fraction_top1_gt_090": float((top[eligible] > .9).float().mean()) if eligible.any() else None}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bootstrap(snapshot):
    for suffix in reversed(("motion_expert_t2m_edge", "motion_expert_joint_attention", "nymeria_world", "")):
        sys.path.insert(0, str(snapshot / suffix))


class AttentionProbe:
    """Retain the execution policy's grouped-method identity during observation."""
    def __init__(self, original, invoke):
        self.__func__ = original.__func__
        self.invoke = invoke

    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)


class Observer:
    def __init__(self, model):
        self.records, self.residuals, self.maps = [], [], {}
        self.context = None
        self.originals, self.hooks = [], []
        from modality_bridge import grouped_attention_plan
        for layer, bridge in model.bridges.items():
            original = bridge._joint_attention
            self.originals.append((bridge, original))

            def inspected(block, q, k, v, meta, *, attn_bias=None, _original=original, _layer=layer):
                result = _original(q, k, v, meta, attn_bias=attn_bias)
                if self.context is None:
                    return result
                with torch.no_grad(), torch.autocast(device_type=q.device.type, enabled=False):
                    plan = grouped_attention_plan(meta, q.device, block.conditioning_attention)
                    h, _, d = q.shape
                    b, nq = plan.query_indexes.shape
                    nk = plan.key_indexes.shape[1]
                    qg = q.to(torch.bfloat16).index_select(1, plan.query_indexes.flatten()).reshape(h, b, nq, d).transpose(0, 1)
                    kg = k.to(torch.bfloat16).index_select(1, plan.key_indexes.flatten()).reshape(h, b, nk, d).transpose(0, 1)
                    vg = v.to(torch.bfloat16).index_select(1, plan.key_indexes.flatten()).reshape(h, b, nk, d).transpose(0, 1)
                    scores = (qg @ kg.transpose(-2, -1)).float() * d ** -.5
                    probs = (scores + plan.bias).softmax(-1)
                    reference_probs = ((qg.float() @ kg.float().transpose(-2, -1)) * d ** -.5 + plan.bias).softmax(-1)
                    reference_output = reference_probs @ vg.float()
                    reproduced = (probs.to(torch.bfloat16) @ vg).float().transpose(0, 1).reshape(h, b * nq, d).index_select(1, plan.restore_indexes)
                    torch.testing.assert_close(result, reproduced, rtol=0, atol=0)
                    valid = torch.zeros(b * nq, device=q.device, dtype=torch.bool)
                    valid[plan.restore_indexes] = True
                    valid = valid.reshape(b, nq)
                    allowed = (plan.bias[:, 0] > -1e20)[:, None].expand(b, h, nq, nk)
                    ng = len(meta.gen_frame)
                    query_video = plan.query_indexes < ng
                    key_video = plan.key_indexes < ng
                    clean = torch.cat((meta.gen_clean, meta.motion_clean)).index_select(0, plan.query_indexes.flatten()).reshape(b, nq)
                    record = {**self.context, "layer": int(_layer),
                              "probabilities_finite": bool(torch.isfinite(probs).all()),
                              "masked_probability_max": probs.masked_select(~allowed).abs().max().item() if (~allowed).any() else 0.,
                              "exact_native_grouped_output_reproduced": True,
                              "q_norm": summary(q.float().norm(dim=-1)), "k_norm": summary(k.float().norm(dim=-1)),
                              "roles": {}}
                    for video_role, role in ((True, "video"), (False, "motion")):
                        for clean_role in (True, False):
                            select = valid & (query_video == video_role) & (clean == clean_role)
                            rows = select[:, None].expand(b, h, nq)
                            if not rows.any():
                                continue
                            pp, aa = probs[rows], allowed[rows]
                            cross_keys = (key_video != video_role)[:, None, None, :].expand_as(probs)[rows]
                            stats = attention_rows(pp, aa)
                            stats["cross_modal_mass"] = summary((pp * cross_keys).sum(-1))
                            stats["uniform_cross_modal_mass"] = summary((aa & cross_keys).sum(-1) / aa.sum(-1))
                            per_row_max = scores[rows].masked_fill(~aa, -torch.inf).max(-1).values
                            per_row_min = scores[rows].masked_fill(~aa, torch.inf).min(-1).values
                            stats["allowed_logit_span"] = summary(per_row_max - per_row_min)
                            ref = reference_probs[rows]
                            actual_head_output = (probs.to(torch.bfloat16) @ vg).float()[rows]
                            ref_head_output = reference_output[rows]
                            stats['bf16_vs_fp32_fixed_projected_qkv'] = {
                                'attention_l1_per_row': summary((pp-ref).abs().sum(-1)),
                                'argmax_disagreement_fraction': float((pp.argmax(-1) != ref.argmax(-1)).float().mean()),
                                'relative_output_rms': float((actual_head_output-ref_head_output).square().mean().sqrt()
                                                             / ref_head_output.square().mean().sqrt().clamp_min(1e-30))}
                            # Sharp rows alone do not imply all queries collapse to one key.
                            query_count = select.sum(-1)
                            eligible_groups = query_count > 1
                            winner_counts = (torch.nn.functional.one_hot(probs.argmax(-1), nk).float()
                                             * select[:, None, :, None]).sum(-2)
                            winner_share = winner_counts.max(-1).values / query_count.clamp_min(1)[:, None]
                            stats["group_head_dominant_key_query_fraction"] = summary(winner_share[eligible_groups])
                            stats["fraction_group_heads_all_queries_same_key"] = (float((winner_share[eligible_groups] >= 1 - 1e-6).float().mean())
                                                                                  if eligible_groups.any() else None)
                            head_output = (probs.to(torch.bfloat16) @ vg).float()
                            selected_output = head_output * select[:, None, :, None]
                            output_mean = selected_output.sum(-2) / query_count.clamp_min(1)[:, None, None]
                            variance = ((head_output - output_mean[:, :, None]).square()
                                        * select[:, None, :, None]).sum((-2, -1))
                            energy = selected_output.square().sum((-2, -1)).clamp_min(1e-30)
                            stats["within_group_head_output_variance_fraction"] = summary((variance / energy)[eligible_groups])
                            stats["head_mean_normalized_entropy"] = []
                            for head in range(h):
                                head_stats = attention_rows(probs[:, head][select], allowed[:, head][select])
                                stats["head_mean_normalized_entropy"].append(head_stats["normalized_entropy"].get("mean"))
                            record["roles"][role + ("_clean" if clean_role else "_noisy")] = stats
                    self.records.append(record)
                    key = f"{self.context['id']}_{meta.mode}_{self.context['sigma']}_layer{_layer}"
                    self.maps[key] = probs.mean(1).cpu().numpy().astype(np.float16)
                    self.maps[key + '_heads_g2'] = probs[min(2, b-1), [0, 7, 15]].cpu().numpy().astype(np.float16)
                return result

            bridge._joint_attention = AttentionProbe(original, types.MethodType(inspected, bridge))

            def residual(block, inputs, outputs, _layer=layer):
                if self.context is None:
                    return
                record = {**self.context, "layer": int(_layer)}
                for name, before, after in zip(("video", "motion"), inputs[:2], outputs):
                    if name == "motion":
                        before, after = before[1:], after[1:]
                    base = before.float().square().mean().sqrt()
                    delta = (after.float() - before.float()).square().mean().sqrt()
                    record[name] = {"input_rms": base.item(), "gated_delta_rms": delta.item(),
                                    "residual_ratio": (delta / base.clamp_min(1e-20)).item()}
                self.residuals.append(record)

            self.hooks.append(bridge.register_forward_hook(residual))

    def close(self):
        for bridge, original in self.originals:
            bridge._joint_attention = original
        for hook in self.hooks:
            hook.remove()


def gradient_stats(model, gradients):
    totals = {}
    for (name, p), grad in zip(model.bridges.named_parameters(), gradients):
        if grad is None:
            continue
        layer, parameter = name.split(".", 1)
        role = "video" if "_g" in parameter else "motion"
        kind = parameter.split("_")[0]
        value = float(grad.detach().float().square().sum())
        for key in ("all", f"layer/{layer}", f"modality/{role}", f"kind/{kind}", f"parameter/{name}"):
            totals[key] = totals.get(key, 0.) + value
    return {key: math.sqrt(value) for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--phase2", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--skip-gradients", action="store_true")
    parser.add_argument("--caption-branch", choices=('native', 'empty'), default='native')
    args = parser.parse_args()
    assert args.out.resolve().is_relative_to('/mnt/local')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.manual_seed(7301)
    torch.backends.cuda.matmul.allow_tf32 = False
    bootstrap(args.snapshot)
    from motion_expert_phase3_edge_joint.settings import RECIPE, digest_json, atomic_json
    from motion_expert_phase3_edge_image_joint.train import configure_local_paths
    from motion_expert_phase3_edge_image_joint.model import EdgePhase3
    from motion_expert_phase3_edge_joint.adapters import load_trainable_state
    from motion_expert_phase3_edge_joint.runtime import execution_context
    from motion_expert_phase3_edge_joint.data import Phase3Dataset, collate
    from motion_expert_phase3_edge_joint.objectives import noise_batch, specialist_loss
    import config
    contract = json.loads((args.run_dir / 'contract.json').read_text())
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert payload['fingerprint'] == digest_json(contract)
    recipe = payload['scheduler']['recipe']
    assert recipe == contract['recipe']
    RECIPE.clear()
    RECIPE.update(recipe)
    configure_local_paths()
    # Match source identities, not just checkpoint tensor shapes.
    assert sha256(args.snapshot / 'motion_expert_joint_attention/modality_bridge.py') == contract['shared_bridge']
    for name, expected in contract['implementation'].items():
        assert sha256(args.snapshot / 'motion_expert_phase3_edge_joint' / name) == expected, name
    assert sha256(args.snapshot / 'scripts/phase3_performance.py') == contract['execution_policy']
    model = EdgePhase3(args.phase1, args.phase2, device='cuda', activation_checkpointing=True)
    assert payload['sources']['phase2_contract'] == model.provenance['phase2_contract']
    assert payload['sources']['phase1_metadata_sha256'] == model.provenance['phase1_metadata_sha256']
    load_trainable_state(model, payload)
    version = {name: p._version for name, p in model.bridges.named_parameters()}
    gates = {name: p.item() for name, p in model.bridges.named_parameters() if 'gate_' in name}
    dataset = Phase3Dataset(args.cache)
    assert dataset.index['fingerprint'] == contract['dataset']
    selected = []
    for i in range(args.cases):
        index = (173 + i * 104729) % len(dataset)
        selected.append((index, dataset[(i * 37 + 11, index)]))
    def task_row(index, row, mode):
        # The cache dataset chooses a task before captions. Reassigning the task
        # alone would silently probe the dropout branch of old M2V checkpoints.
        uses_caption = str(recipe['caption_policy'][mode]).startswith('caption')
        caption = dataset.rows[index]['caption'] if uses_caption and args.caption_branch == 'native' else ''
        return {**row, 'mode':mode, 'caption':caption}

    specialist_conditioning = []
    original_motion_prepare, original_video_pack = model.motion.prepare_conditions, model.edge.pack_video
    def inspect_motion(captions):
        specialist_conditioning.append({'specialist':'motion','caption_nonempty':[bool(v) for v in captions]})
        return original_motion_prepare(captions)
    def inspect_video(captions, modes, latents, sigma):
        specialist_conditioning.append({'specialist':'video','modes':modes,'caption_nonempty':[bool(v) for v in captions]})
        return original_video_pack(captions, modes, latents, sigma)
    model.motion.prepare_conditions, model.edge.pack_video = inspect_motion, inspect_video
    mean = torch.from_numpy(np.load(config.MOTION_STATS_MEAN)).to('cuda').float()
    std = torch.from_numpy(np.load(config.MOTION_STATS_STD)).to('cuda').float()
    observations = []
    with execution_context(recipe['bridge_execution']):
        observer = Observer(model)
        model.eval()
        with torch.no_grad():
            for index, row in selected:
                for mode in ('video2motion', 'motimg2video'):
                    raw = collate([task_row(index, row, mode)])
                    for sigma in (.1, .5, .9):
                        torch.manual_seed(100 + index)
                        inputs, target = noise_batch(raw, 'cuda')
                        m = raw['motion'].to('cuda').float()
                        v = raw['video'].to('cuda').float()
                        ms = sigma if mode == 'video2motion' else 0.
                        vs = sigma if mode == 'motimg2video' else 0.
                        mn, vn = torch.randn_like(m), torch.randn_like(v)
                        inputs.update(motion=torch.lerp(m, mn, ms),
                                      video=torch.lerp(v, vn, vs),
                                      motion_sigma=torch.tensor([ms], device='cuda'),
                                      video_sigma=torch.tensor([vs], device='cuda'))
                        inputs['video'][:, :, 0] = v[:, :, 0]
                        target['velocity'] = vn - v
                        observer.context = dict(id=index, case=row['case'], mode=mode, sigma=sigma)
                        output = model(**inputs)
                        observation = {**observer.context, 'caption_nonempty': bool(inputs['captions'][0]),
                                       'finite_outputs': {key: bool(torch.isfinite(output[key]).all()) for key in ('motion','video')}}
                        if sigma == .5:
                            baseline = model(**inputs, bridges_enabled=False)
                            active_loss, _ = specialist_loss(output, target, inputs, mean, std)
                            baseline_loss, _ = specialist_loss(baseline, target, inputs, mean, std)
                            key = 'motion' if mode == 'video2motion' else 'video'
                            delta = (output[key].float() - baseline[key].float()).square().mean().sqrt()
                            observation['bridge_ablation'] = dict(active_native_loss=active_loss.item(),
                                disabled_native_loss=baseline_loss.item(), prediction_delta_rms=delta.item())
                            del baseline, active_loss, baseline_loss
                        observations.append(observation)
                        del output, inputs, target
                        print(f"[diagnostics] attention case={index} mode={mode} sigma={sigma}", flush=True)
        observer.context = None
        model.train()
        gradients = []
        for offset in range(0, 0 if args.skip_gradients else len(selected), 2):
            pair = selected[offset:offset + 2]
            if len(pair) != 2:
                break
            raw = collate([task_row(index, row, 'video2motion' if j == 0 else 'motimg2video') for j, (index, row) in enumerate(pair)])
            torch.manual_seed(9000 + offset)
            inputs, target = noise_batch(raw, 'cuda')
            output = model(**inputs)
            loss, metrics = specialist_loss(output, target, inputs, mean, std)
            params = tuple(model.bridges.parameters())
            # Recover the exact two native task losses using one-element views.
            branches = {}
            for i, name in enumerate(('motion', 'video_weight10')):
                so = {k: output[k][i:i+1] for k in ('motion','video')}
                so['anchor'] = output['anchor']
                so['packed'] = types.SimpleNamespace(vision=types.SimpleNamespace(condition_mask=[output['packed'].vision.condition_mask[i]]))
                si = {k: (v[i:i+1] if isinstance(v, torch.Tensor) else v[i:i+1]) for k,v in inputs.items()}
                st = {k: v[i:i+1] for k,v in target.items()}
                branch, _ = specialist_loss(so, st, si, mean, std)
                branches[name] = branch * .5
            torch.testing.assert_close(loss, sum(branches.values()), rtol=1e-5, atol=1e-6)
            gm = torch.autograd.grad(branches['motion'], params, retain_graph=True, allow_unused=True)
            gv = torch.autograd.grad(branches['video_weight10'], params, allow_unused=True)
            total = tuple((a if b is None else b if a is None else a + b) for a,b in zip(gm,gv))
            stats = {'batch_cases':raw['cases'], 'motion_sigma':inputs['motion_sigma'].tolist(),
                     'video_sigma':inputs['video_sigma'].tolist(), 'loss':float(loss.detach()),
                     'motion':gradient_stats(model,gm), 'video_weight10':gradient_stats(model,gv),
                     'total':gradient_stats(model,total), 'batch_size':2,
                     'note':'Offline batch2 native loss gradients; not directly comparable to live global32 norms.'}
            gradients.append(stats)
            print('[diagnostics] gradient norms', {k:stats[k]['all'] for k in ('motion','video_weight10','total')}, flush=True)
            del output, loss, gm, gv, total, branches, params
    observer.close()
    model.motion.prepare_conditions, model.edge.pack_video = original_motion_prepare, original_video_pack
    assert all(p._version == version[name] for name,p in model.bridges.named_parameters())
    assert all(p.grad is None for p in model.parameters())
    np.savez_compressed(args.out/'attention_maps.npz', **observer.maps)
    receipt = dict(checkpoint=str(args.checkpoint), step=payload['step'], run_dir=str(args.run_dir),
                   snapshot=str(args.snapshot), source_verified=True, recipe=recipe, gates=gates,
                   cases=[{'index':i,'case':r['case']} for i,r in selected],
                   observations=observations, attention=observer.records, residuals=observer.residuals,
                   caption_branch=args.caption_branch, specialist_conditioning=specialist_conditioning,
                   gradients=gradients, weights_unchanged=True, optimizer_steps=0,
                   audit_script_sha256=sha256(__file__), completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    atomic_json(args.out/'diagnostics.json', receipt)
    atomic_json(args.out/'COMPLETE.json', {k:receipt[k] for k in ('step','checkpoint','weights_unchanged','completed_utc')})
    print('[diagnostics] COMPLETE', args.out, flush=True)


if __name__ == '__main__':
    main()
