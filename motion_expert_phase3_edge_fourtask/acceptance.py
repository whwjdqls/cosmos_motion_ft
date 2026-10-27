"""Compute-node acceptance of all conditioning modes before fresh production."""
import hashlib
import json

import torch
import torch.distributed as dist

from checkpoint import capture_rng_state, restore_rng_state
from modality_bridge import LocalModalityBridge
from motion_expert_phase3_edge.model import EdgePhase3 as BaseEdgePhase3
from .data import collate
from .model import bridge_meta
from .objectives import noise_batch, specialist_loss
from .settings import RECIPE, atomic_json, has_camera, has_video, has_text, effective_caption


def run_preflight(args, dataset, model, ddp, mean, std, rank, world, device, fingerprint, *, helpers):
    checks, token_counts = [], {}
    for mode in RECIPE["tasks"]:
        meta = bridge_meta(mode, device)
        ng = meta.gen_frame.numel()
        expected = (1600 if has_video(mode) else 0) + (96 if has_camera(mode) else 0)
        assert ng == expected
        bridge = LocalModalityBridge(8, 2, 4).to(device)
        mask = bridge._attention_mask(meta)
        local = bridge._source_interval_pair_mask(meta)
        assert local[:, 1:].any(dim=0).all(), "Every source motion frame needs a local conditioning edge"
        if has_camera(mode):
            actions = local[-96:]
            expected_actions = torch.zeros_like(actions)
            i = torch.arange(96, device=device)
            expected_actions[i, i + 1] = True
            expected_actions[i, i + 2] = True
            assert torch.equal(actions, expected_actions)
        if mode == "motimg2video":
            assert not mask[ng:, :ng].any()
        else:
            assert not mask[:ng, ng:].any()
        with torch.no_grad():
            bridge.gate_g.fill_(0.1)
            bridge.gate_m.fill_(0.1)
            g, m = torch.randn(ng, 8, device=device), torch.randn(98, 8, device=device)
            ga, ma = bridge(g, m, meta)
            gb, mb = bridge(g + 1, m, meta) if mode == "motimg2video" else bridge(g, m + 1, meta)
            torch.testing.assert_close(ma if mode == "motimg2video" else ga,
                                       mb if mode == "motimg2video" else gb, rtol=0, atol=0)
        del bridge, mask, local
    checks.append("causal_Wan_and_camera_transition_intervals_with_clean_condition_isolation")
    rows = [dataset.get_item((i, (rank * args.microbatch + i) % len(dataset)),
                             force_mode="motimg2video", dropout=False) for i in range(args.microbatch)]
    batch = collate(rows)
    assert torch.isfinite(batch["camera_action"]).all()
    assert torch.count_nonzero(batch["camera_action"][:, :, 9:]) == 0
    versions = helpers.frozen_versions(model)
    model.eval()
    with torch.no_grad():
        for mode in RECIPE["tasks"]:
            batch["modes"] = [mode] * len(rows)
            batch["captions"] = [effective_caption(mode, "The camera wearer walks forward.")] * len(rows)
            inputs, target = noise_batch(batch, device)
            assert torch.equal(inputs["camera_action"].cpu(), batch["camera_action"])
            if mode != "motimg2video":
                assert torch.count_nonzero(inputs["video_sigma"]) == 0
                torch.testing.assert_close(inputs["video"], batch["video"].to(device).float(), rtol=0, atol=0)
            else:
                assert torch.count_nonzero(inputs["motion_sigma"]) == 0
                torch.testing.assert_close(inputs["motion"], batch["motion"].to(device), rtol=0, atol=0)
            actual = model(**inputs)
            assert actual["vision_sample_indices"] == (list(range(len(rows))) if has_video(mode) else [])
            assert actual["camera_sample_indices"] == (list(range(len(rows))) if has_camera(mode) else [])
            packed = actual["packed"]
            nv = packed.vision.sequence_indexes.numel() if has_video(mode) else 0
            nc = packed.action.sequence_indexes.numel() if has_camera(mode) else 0
            assert nv == len(rows) * (1600 if has_video(mode) else 0)
            assert nc == len(rows) * (96 if has_camera(mode) else 0)
            if not has_video(mode):
                vision = getattr(packed, "vision", None)
                indices = getattr(vision, "sequence_indexes", None)
                assert indices is None or indices.numel() == 0
            if has_camera(mode):
                assert all(item.numel() == 0 for item in packed.action.noisy_frame_indexes)
                assert all(int(item) == 2 for item in packed.action.domain_id)
            token_counts[mode] = {"vision_per_case": nv // len(rows), "camera_per_case": nc // len(rows)}
            if mode != "motimg2video":
                baseline = model.motion(
                    reasoner_inputs=model.motion.prepare_conditions(inputs["captions"]),
                    x_sigma=inputs["motion"], sigma=inputs["motion_sigma"], neutral_joints=inputs["neutral"],
                    motion_pad_mask=torch.zeros(len(rows), 97, dtype=torch.bool, device=device))
                torch.testing.assert_close(actual["motion"], baseline, rtol=0, atol=0)
            else:
                reference = model.native(packed)["preds_vision"]
                baseline = torch.stack([v.reshape(48, 25, 16, 16) for v in reference])
                torch.testing.assert_close(actual["video"], baseline, rtol=0, atol=0)
                _, terms = specialist_loss(actual, target, inputs, mean, std)
                loss_mask = torch.ones_like(baseline)
                loss_mask[:, :, 0] = 0
                expected = ((baseline.float() - target["velocity"]).square() * loss_mask).mean()
                torch.testing.assert_close(terms["video"], expected, rtol=1e-6, atol=1e-7)
            if mode in {"video2motion", "motimg2video"}:
                old = BaseEdgePhase3.forward(model, **{k: v for k, v in inputs.items() if k != "camera_action"})
                torch.testing.assert_close(actual["motion"], old["motion"], rtol=0, atol=0)
                torch.testing.assert_close(actual["video"], old["video"], rtol=0, atol=0)
                del old
            bypass = model(**inputs, bridges_enabled=False)
            torch.testing.assert_close(actual["motion"], bypass["motion"], rtol=0, atol=0)
            torch.testing.assert_close(actual["video"], bypass["video"], rtol=0, atol=0)
            print(f"[camcond/preflight] rank={rank} mode={mode} zero_gate_parity=PASS tokens={token_counts[mode]}", flush=True)
            del actual, baseline, bypass
    checks.extend(["native_clean_raw_camera_domain2_and_no_camera_loss",
                   "zero_bridge_bit_exact_original_V2M_M2V_and_frozen_specialists",
                   "no_vision_tokens_in_camera_only_modes", "native_video_weight10_full_denominator"])
    names = list(RECIPE["tasks"])
    batch["modes"] = [names[(rank * len(rows) + i) % len(names)] for i in range(len(rows))]
    batch["captions"] = [effective_caption(m, "The camera wearer walks forward.") for m in batch["modes"]]
    inputs, target = noise_batch(batch, device)
    optimizer = helpers.new_optimizer(model)
    model.train()
    losses = []
    for step in range(12):
        optimizer.zero_grad(set_to_none=True)
        output = ddp(**inputs)
        loss, _ = specialist_loss(output, target, inputs, mean, std)
        helpers.collective_check(torch.isfinite(loss).item(), "Non-finite camera acceptance loss", device)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.bridges.parameters(), 1.0)
        helpers.collective_check(torch.isfinite(norm).item(), "Non-finite camera acceptance gradients", device)
        gates = [p for name, p in model.bridges.named_parameters() if "gate_" in name]
        core = [p for name, p in model.bridges.named_parameters() if "gate_" not in name]
        if step == 0:
            assert any(p.grad is not None and p.grad.abs().max().item() > 0 for p in gates)
            assert all(p.grad is not None and p.grad.abs().max().item() == 0 for p in core)
        if step == 11:
            assert any(p.grad is not None and p.grad.abs().max().item() > 0 for p in core)
        optimizer.step()
        helpers.assert_frozen(model, versions)
        value = loss.detach().clone()
        if world > 1:
            dist.all_reduce(value)
            value /= world
        losses.append(value.item())
        if rank == 0:
            print(f"[camcond/preflight] mixed_four_task_overfit={step+1}/12 loss={value.item():.7f}", flush=True)
        del output, loss
    if losses[-1] >= losses[0]:
        raise RuntimeError(f"Fixed-batch overfit did not improve: {losses[0]} -> {losses[-1]}")
    checks.extend(["four_task_DDP_finite_backward_and_overfit", "gate_then_core_gradients", "specialists_frozen"])
    model.eval()
    conditioning = {}
    with torch.no_grad():
        single = collate(rows[:1])
        for mode in ("camera2motion", "cameravideo2motion"):
            single["modes"], single["captions"] = [mode], ["The camera wearer walks forward."]
            inp, _ = noise_batch(single, device)
            reference = model(**inp)["motion"]
            perturbed = inp["camera_action"].clone()
            perturbed[:, :, :3] += 1.0
            changed = model(**{**inp, "camera_action": perturbed})["motion"]
            difference = (changed - reference).float().abs().max().item()
            assert difference > 0, f"Camera input has no effect after overfit: {mode}"
            if not has_video(mode):
                omitted = model(**{**inp, "video": inp["video"] + 10.0})["motion"]
                torch.testing.assert_close(reference, omitted, rtol=0, atol=0)
            if not has_text(mode):
                omitted = model(**{**inp, "captions": ["A completely different caption."]})["motion"]
                torch.testing.assert_close(reference, omitted, rtol=0, atol=0)
            conditioning[mode] = {"camera_change_max_abs": difference,
                                  "omitted_video_invariant": not has_video(mode),
                                  "omitted_caption_invariant": not has_text(mode)}
        # Original task paths remain unchanged even after bridge weights learn.
        for mode in ("video2motion", "motimg2video"):
            single["modes"] = [mode]
            inp, _ = noise_batch(single, device)
            actual = model(**inp)
            original = BaseEdgePhase3.forward(model, **{k: v for k, v in inp.items() if k != "camera_action"})
            for key in ("motion", "video"):
                torch.testing.assert_close(actual[key], original[key], rtol=0, atol=0)
    checks.extend(["learned_camera_signal_in_both_trained_variants",
                   "nonzero_bridge_no_video_no_caption_leakage", "learned_original_task_bit_exact_parity"])
    model.train()
    rng = helpers.all_objects(capture_rng_state(), world)
    path = args.out / "preflight_roundtrip.pt"
    expected_state = {k: v.detach().cpu().clone() for k, v in model.bridges.state_dict().items()}
    if rank == 0:
        helpers.atomic_checkpoint(path, {"bridges": expected_state, "optimizer": optimizer.state_dict(), "rng": rng})
    helpers.barrier()
    restored = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    model.bridges.load_state_dict(restored["bridges"], strict=True)
    optimizer.load_state_dict(restored["optimizer"])
    restore_rng_state(restored["rng"][rank])
    digest = hashlib.sha256()
    for key, value in model.bridges.state_dict().items():
        assert torch.equal(value.detach().cpu(), expected_state[key])
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    assert len(set(helpers.all_objects(digest.hexdigest(), world))) == 1
    del restored, expected_state
    optimizer.zero_grad(set_to_none=True)
    output = ddp(**inputs)
    loss, _ = specialist_loss(output, target, inputs, mean, std)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.bridges.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    helpers.assert_frozen(model, versions)
    checks.append("DDP_bit_identical_checkpoint_optimizer_RNG_reload_and_continue")
    helpers.barrier()
    if rank == 0:
        path.unlink()
        atomic_json(args.out / "acceptance.json", {
            "passed": True, "fingerprint": fingerprint, "checks": checks,
            "world_size": world, "microbatch": args.microbatch,
            "overfit_losses": losses, "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            "token_counts": token_counts, "conditioning": conditioning,
            "recipe": RECIPE, "production_weights": "fresh_zero_gated_bridges_not_preflight_weights",
        })
        print("[camcond/preflight] PASSED; start fresh production, not these disposable weights", flush=True)
