"""Install the additive experiment into the unchanged, guarded DDP trainer.

The Foundary operation adapter supplies audited local paths and bounded stop
support. Native specialist formulas, production/DDP gates and resume checks
remain in the existing trainer. All extension and adapter sources are hashed.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from motion_expert_phase3_edge_joint.settings import RECIPE, atomic_json, sha256
from .data import Phase3Dataset
from .model import EdgePhase3
from .objectives import noise_batch, specialist_loss
from .policy import configure
from motion_expert_phase3_edge_image_joint.train import configure_local_paths


def install(train):
    configure()
    train.Phase3Dataset = Phase3Dataset
    train.EdgePhase3 = EdgePhase3
    train.noise_batch = noise_batch
    train.specialist_loss = specialist_loss
    original_identity = train.source_identity

    def source_identity(args, dataset, model):
        if RECIPE["conditioning_attention"] != "bidirectional" or RECIPE["clean_motion_timestep"] != "zero":
            raise ValueError("This experiment requires bidirectional conditioning and clean-motion E_M(0)")
        contract = original_identity(args, dataset, model)
        contract["image_joint_implementation"] = {
            p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        contract["execution_policy"] = sha256(Path(__file__).parents[1] / "scripts/phase3_performance.py")
        if adapter := os.environ.get("PHASE3_IMAGE_JOINT_ADAPTER"):
            contract["site_adapter_sha256"] = sha256(adapter)
        return contract

    train.source_identity = source_identity
    original_preflight = train.preflight

    def preflight(args, dataset, model, ddp, mean, std, rank, world, device, fingerprint):
        original_preflight(args, dataset, model, ddp, mean, std, rank, world, device, fingerprint)
        train.barrier()
        # Revoke the provisional base acceptance until these joint-specific
        # checks also pass. The launcher cannot proceed after a failed check.
        import json
        acceptance_path = args.out / "acceptance.json"
        acceptance = json.loads(acceptance_path.read_text())
        train.barrier()
        if rank == 0:
            acceptance_path.unlink()
        train.barrier()
        rows = [dataset[(rank, rank % len(dataset))]]
        raw = train.collate(rows)
        raw["modes"] = ["joint"]
        raw["captions"] = ["PRIVATE_GROUND_TRUTH_CAPTION_MUST_NOT_ENTER_EITHER_SPECIALIST"]
        inputs, target = noise_batch(raw, device)
        assert inputs["captions"] == [""]
        assert 0 < inputs["motion_sigma"].item() < 1
        assert 0 < inputs["video_sigma"].item() < 1
        assert inputs["motion_sigma"].item() != inputs["video_sigma"].item()
        torch.testing.assert_close(inputs["video"][:, :, 0], raw["video"].to(device).float()[:, :, 0], rtol=0, atol=0)
        seen = {"motion": [], "video": []}
        prepare_conditions, pack_video = model.motion.prepare_conditions, model.edge.pack_video

        def motion_conditions(captions):
            assert captions == [""]
            seen["motion"].append(captions)
            return prepare_conditions(captions)

        def video_pack(captions, modes, latents, sigma):
            assert captions == [""] and modes == ["joint"]
            seen["video"].append(captions)
            return pack_video(captions, modes, latents, sigma)

        model.motion.prepare_conditions, model.edge.pack_video = motion_conditions, video_pack
        model.eval()
        try:
            with torch.no_grad():
                blank = model(**inputs)
                poisoned = model(**{**inputs, "captions": raw["captions"]})
                for key in ("motion", "video"):
                    torch.testing.assert_close(blank[key], poisoned[key], rtol=0, atol=0)
                mask = blank["packed"].vision.condition_mask[0].reshape(25, -1)
                assert bool(mask[0].all()) and not bool(mask[1:].any())
                del blank, poisoned
        finally:
            model.motion.prepare_conditions, model.edge.pack_video = prepare_conditions, pack_video
        assert all(len(calls) == 2 for calls in seen.values())
        model.train()
        # Both native joint losses must contribute gradients to real bridges
        # after the disposable gate-opening updates, without changing weights.
        versions = train.frozen_versions(model)
        output = model(**inputs)
        video_loss = (output["video"][:, :, 1:].float() - target["velocity"][:, :, 1:]).square().sum()
        motion_loss = (output["motion"].float() - target["motion"]).square().mean()
        params = tuple(model.bridges.parameters())
        branch_norms = {}
        for key, loss in (("video", video_loss), ("motion", motion_loss)):
            grads = torch.autograd.grad(loss, params, retain_graph=key == "video", allow_unused=True)
            norm = sum(g.float().square().sum() for g in grads if g is not None).sqrt()
            assert torch.isfinite(norm) and norm.item() > 0, key
            branch_norms[key] = norm.item()
        train.assert_frozen(model, versions)
        # Verify the actual native loss still includes both branches.
        with torch.no_grad():
            _, metrics = specialist_loss(output, target, inputs, mean, std)
            assert metrics["motion_count"].item() == metrics["video_count"].item() == metrics["joint_count"].item() == 1
        del output, video_loss, motion_loss, grads
        receipt = {"passed": True, "motion_sigma": inputs["motion_sigma"].item(),
                   "video_sigma": inputs["video_sigma"].item(), "caption_invariant_outputs_bit_exact": True,
                   "both_specialist_caption_inputs_empty": True, "initial_image_clean_and_loss_excluded": True,
                   "both_joint_branches_reach_bridges": branch_norms, "frozen_specialists": True}
        receipts = train.all_objects(receipt, world)
        if rank == 0:
            acceptance["checks"].extend(["joint_caption_invariance_both_specialists_actual_nonzero_bridges",
                                         "independent_joint_times_initial_image_clean_and_loss_excluded",
                                         "both_joint_targets_have_nonzero_bridge_gradients"])
            acceptance["joint_audits_by_rank"] = receipts
            acceptance["recipe"] = RECIPE.copy()
            atomic_json(acceptance_path, acceptance)
            print("[phase3/image-joint] JOINT_ACCEPTANCE_PASSED", flush=True)
        train.barrier()

    train.preflight = preflight
    return train


def prepare_trainer(probe_updates=0):
    """Use the existing trainer without changing its contracted source files."""
    import inspect
    from motion_expert_phase3_edge_joint import train
    configure()
    mapping = configure_local_paths()
    # The inherited preflight makes a stand-alone Phase2 reference. It must
    # use the actual sanitized captions, just as the new model does.
    source = inspect.getsource(train.preflight)
    old = 'reasoner_inputs=model.motion.prepare_conditions(batch["captions"])'
    new = 'reasoner_inputs=model.motion.prepare_conditions(inputs["captions"])'
    if source.count(old) != 1:
        raise RuntimeError("Upstream preflight changed; review the caption reference adapter")
    exec(compile(source.replace(old, new), str(__file__) + ":preflight_reference", "exec"), train.__dict__)
    # Bounded disposable save/reload probes preserve the full 200k schedule.
    # Production uses zero and retains the inherited signal-driven shutdown.
    source = inspect.getsource(train.main)
    old = "int(STOP)"
    new = "int(STOP or (IMAGE_JOINT_PROBE_UPDATES > 0 and step >= start_step + IMAGE_JOINT_PROBE_UPDATES))"
    if source.count(old) != 1:
        raise RuntimeError("Upstream trainer changed; review the bounded probe adapter")
    train.IMAGE_JOINT_PROBE_UPDATES = probe_updates
    exec(compile(source.replace(old, new), str(__file__) + ":bounded_probe", "exec"), train.__dict__)
    train = install(train)
    original_identity = train.source_identity

    def source_identity(args, dataset, model):
        return {**original_identity(args, dataset, model), "local_cache_path_translation": mapping,
                "runtime_overlays": {"preflight_reference": "actual_sanitized_captions",
                                     "bounded_probe": "optional_stop_after_updates_full_schedule_preserved"}}

    train.source_identity = source_identity
    return train
