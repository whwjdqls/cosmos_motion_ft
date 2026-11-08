"""Fresh directional50/50 MMDiT training through the common acceptance gate."""
import argparse
import inspect
from pathlib import Path
import sys

import torch

from motion_expert_phase3_edge_image_joint.model import EdgePhase3
from motion_expert_phase3_edge_image_joint.train import configure_local_paths
from motion_expert_phase3_edge_joint.settings import RECIPE, sha256
from .data import Phase3Dataset
from .policy import configure, TASKS, attention_policy


def prepare_trainer(probe_updates=0):
    from motion_expert_phase3_edge_joint import train
    configure()
    mapping = configure_local_paths()
    source = inspect.getsource(train.main)
    if source.count("int(STOP)") != 1:
        raise RuntimeError("Review bounded probe overlay after upstream trainer changes")
    train.MMDIT_PROBE_UPDATES = probe_updates
    source = source.replace("int(STOP)", "int(STOP or (MMDIT_PROBE_UPDATES > 0 and step >= start_step + MMDIT_PROBE_UPDATES))")
    exec(compile(source, __file__ + ":bounded_probe", "exec"), train.__dict__)
    train.Phase3Dataset, train.EdgePhase3 = Phase3Dataset, EdgePhase3
    original_identity = train.source_identity

    def source_identity(args, dataset, model):
        if RECIPE["tasks"] != TASKS or RECIPE["conditioning_attention"] != attention_policy():
            raise ValueError("This run requires its selected conditioning policy and two equally weighted tasks")
        contract = original_identity(args, dataset, model)
        contract["mmdit_two_task_implementation"] = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        contract["image_joint_model_sha256"] = sha256(Path(inspect.getfile(EdgePhase3)))
        contract["execution_policy"] = sha256(Path(__file__).parents[1] / "scripts/phase3_performance.py")
        contract["local_cache_path_translation"] = mapping
        return contract

    train.source_identity = source_identity
    original_preflight = train.preflight

    def preflight(args, dataset, model, ddp, mean, std, rank, world, device, fingerprint):
        import json
        from .caption_audit import audit_no_captions
        original_preflight(args, dataset, model, ddp, mean, std, rank, world, device, fingerprint)
        train.barrier()
        path = args.out / "acceptance.json"
        receipt = json.loads(path.read_text())
        train.barrier()
        if rank == 0:
            path.unlink()
        train.barrier()
        assert any(p.detach().abs().max().item() > 0
                   for name, p in model.bridges.named_parameters() if "gate_" in name)
        raw = train.collate([dataset[(rank, rank % len(dataset))]])
        versions = train.frozen_versions(model)
        audit = audit_no_captions(model, raw, train.noise_batch, device, list(TASKS))
        train.assert_frozen(model, versions)
        audits = train.all_objects(audit, world)
        if rank == 0:
            receipt["checks"].append("all_enabled_tasks_no_caption_at_both_specialists_nonzero_bridges")
            receipt["no_text_audits_by_rank"] = audits
            from motion_expert_phase3_edge_joint.settings import atomic_json
            atomic_json(path, receipt)
            print("[phase3/mmdit] NO_CAPTION_ACCEPTANCE_PASSED", flush=True)
        train.barrier()

    train.preflight = preflight
    return train


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--probe-updates", type=int, default=0)
    args, rest = parser.parse_known_args()
    if args.probe_updates < 0:
        raise ValueError("Probe updates must be nonnegative")
    for name, value in (("conditioning-attention", attention_policy()), ("clean-motion-timestep", "zero")):
        if "--" + name not in rest:
            rest += ["--" + name, value]
    train = prepare_trainer(args.probe_updates)
    sys.argv = [sys.argv[0], *rest]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train.main()


if __name__ == "__main__":
    main()
