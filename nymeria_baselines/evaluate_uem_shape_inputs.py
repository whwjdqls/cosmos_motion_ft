"""Inference-only RGB/trajectory ablation with the same supplied-shape UEM."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from nymeria_baselines.body_shape import BODY_JOINTS, MOTION_INDICES, PARENTS
from nymeria_baselines.evaluate_uem_phase3 import read, save, sha
from nymeria_baselines.uem_visual_cache import frame_ids, require_compute

CONDITIONS = ("rgb_only", "traj_only", "rgb_traj")


def progress(root, **values):
    save(root / "progress.json", dict(time=time.time(), job=os.environ.get("SLURM_JOB_ID"), **values))
    print(json.dumps(values), flush=True)


def prepare(a):
    """Pin a stable checkpoint before workers start; never touch training files."""
    destination = a.output / "checkpoint.ckpt"
    if destination.exists() or (a.output / "configuration.json").exists():
        raise FileExistsError("Use a fresh output directory for checkpoint preparation")
    temporary = destination.with_suffix(".tmp")
    before = a.checkpoint.stat()
    shutil.copyfile(a.checkpoint, temporary)
    after = a.checkpoint.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        temporary.unlink()
        raise RuntimeError("Training replaced the checkpoint during copying; retry preparation")
    payload = torch.load(temporary, map_location="cpu", weights_only=False)
    cfg = payload["hyper_parameters"]
    data = cfg["DATA"]
    assert data["REPRE_TYPE"] == "soma_body220_shape"
    assert data["COND_IMG_FEAT"] and data["COND_TRAJ"] and not data["COND_BETAS"]
    assert data["WINDOW"] == 97 and not cfg["MODEL"]["LEARN_TRAJ"]
    assert cfg["MODEL"]["FINETUNE_TYPE"] == "recon"
    experiment = read(a.checkpoint.parent / "experiment.json")
    assert experiment["shape_prediction"] is False and experiment["shape_loss"] is False
    assert experiment["state_dim"] == 220 and experiment["shape_dim"] == 90
    assert experiment["task_weights"] == {"recon": 1.0}
    assert Path(experiment["cache"]).resolve() == Path(data["DATA_DIR"]).resolve()
    upstream = Path(experiment["upstream"])
    manifest = read(upstream / "source_manifest.json")
    assert sha(upstream / "source_manifest.json") == experiment["source_manifest_sha256"]
    for relative, expected in manifest["output_files"].items():
        assert sha(upstream / relative) == expected, relative
    cache = Path(data["DATA_DIR"])
    assert sha(cache / "train_stats.npz") == experiment["statistics_sha256"]
    assert sha(cache / "index.json") == experiment["index_sha256"]
    reference = read(a.reference / "configuration.json")
    assert len(reference["cases"]) == len({r["uuid"] for r in reference["cases"]}) == 71
    assert sha(cache / "train_stats.npz") == reference["uem_stats_sha256"]
    features = Path(data["VISUAL_CACHE"])
    ready = read(features / "READY.json")
    assert ready["fingerprint"] == data["VISUAL_CACHE_FINGERPRINT"]
    assert ready["cache_index_sha256"] == experiment["index_sha256"]
    temporary.replace(destination)
    policy = dict(
        checkpoint=str(destination), checkpoint_sha256=sha(destination),
        original_checkpoint=str(a.checkpoint), epoch=int(payload["epoch"]),
        global_step=int(payload["global_step"]), hyper_parameters=cfg,
        upstream=str(upstream), upstream_manifest_sha256=sha(upstream / "source_manifest.json"),
        training_contract=experiment, reference=str(a.reference),
        reference_sha256=sha(a.reference / "configuration.json"), cases=reference["cases"],
        phase3_contract=reference["phase3_contract"], conditions=list(CONDITIONS),
        weights="state_dict, same policy as the previous matched71 UEM comparison; no EMA substitution",
        sampler="Unmodified UEM_Module.sample / diffusion.p_sample_loop",
        diffusion_steps=cfg["MODEL"]["DIFFUSION_STEPS"], cond_scale=cfg["TRAIN"]["COND_SCALE"],
        shape="Identical Phase3 SOMA30 neutral skeleton, centroid-centered meters; clean input in every mode",
        ablation="Single frozen RGB+trajectory recon checkpoint. Omit absent modality keys to use native mask tokens. No captions.",
        caveat="rgb_only and traj_only are missing-input inference ablations, not trained single-modality specialists",
        coordinates="Native frame-0 head-yaw canonical motion; no predicted grounding, reset, alignment or shape fitting",
        frames=97, fps=20, implementation_sha256=sha(Path(__file__)),
    )
    save(a.output / "configuration.json", policy)
    progress(a.output, stage="prepared", step=policy["global_step"], epoch=policy["epoch"])


def inputs(policy, row):
    from nymeria_baselines.prepare import prepare_window
    cfg = policy["hyper_parameters"]["DATA"]
    cache, features = Path(cfg["DATA_DIR"]), Path(cfg["VISUAL_CACHE"])
    index = read(cache / "index.json")
    arrays = {r["uuid"]: r["array"] for r in index["rows"] if r["split"] == "test"}
    assert row["uuid"] not in {r["uuid"] for r in index["rows"] if r["split"] != "test"}
    key, start = arrays[row["uuid"]], int(row["start"])
    def load(suffix):
        return np.load(cache / "records" / f"{key}.{suffix}.npy", mmap_mode="r", allow_pickle=False)
    raw, trajectory = prepare_window(load("motion")[start:start + 97],
                                     load("pose_yup")[start:start + 97], row["floor_offset"])
    neutral = np.array(load("neutral"), dtype=np.float32, copy=True)
    neutral -= neutral.mean(axis=0, keepdims=True)
    contract = policy["phase3_contract"]
    with np.load(row["bundle"], allow_pickle=False) as bundle:
        assert str(bundle["motion_mean_sha256"].item()) == contract["motion_mean"]
        assert str(bundle["motion_std_sha256"].item()) == contract["motion_std"]
        gt = bundle["motion"] * np.load(contract["motion_stats_std"]) + np.load(contract["motion_stats_mean"])
        expected = np.array(bundle["neutral"], dtype=np.float32, copy=True)
        expected -= expected.mean(axis=0, keepdims=True)
    shape_error = float(np.max(np.abs(neutral - expected)))
    gt_error = float(np.max(np.abs(raw - gt)))
    assert neutral.shape == (30, 3) and shape_error <= 1e-6, shape_error
    assert gt_error < 2e-4, gt_error
    with np.load(cache / "train_stats.npz", allow_pickle=False) as stats:
        trajectory = ((trajectory - stats["camera_mean"]) / (stats["camera_std"] + 1e-6)).astype(np.float32)
        mean = stats["motion_mean"][MOTION_INDICES].copy()
        std = stats["motion_std"][MOTION_INDICES].copy()
    ids = np.load(features / "records" / f"{key}.frame_ids.npy", mmap_mode="r")
    wanted = frame_ids(start)
    slots = np.searchsorted(ids, wanted)
    assert np.all(slots < len(ids)) and np.array_equal(ids[slots], wanted)
    image = np.array(np.load(features / "records" / f"{key}.features.npy", mmap_mode="r")[slots], dtype=np.float32)
    assert image.shape == (97, 1024) and trajectory.shape == (97, 18)
    assert all(np.isfinite(x).all() for x in (neutral, raw, image, trajectory))
    return neutral, image, trajectory, mean, std, gt, dict(shape_max_difference=shape_error, gt_max_difference=gt_error)


def conditioning(neutral, image, trajectory, condition):
    y = dict(neutral_joints=torch.from_numpy(neutral[None]).cuda(),
             valid_frames=torch.ones(1, 97, dtype=torch.long, device="cuda"))
    if condition != "traj_only":
        y.update(img_embs=torch.from_numpy(image[None]).cuda(),
                 valid_img_embs=torch.ones(1, 97, dtype=torch.long, device="cuda"),
                 img_mask=torch.zeros(1, 97, dtype=torch.long, device="cuda"))
    if condition != "rgb_only":
        y.update(traj=torch.from_numpy(trajectory[None]).cuda(),
                 traj_mask=torch.zeros(1, 97, dtype=torch.long, device="cuda"))
    return y


def sample(a):
    from yacs.config import CfgNode
    policy = read(a.output / "configuration.json")
    sys.path.insert(0, str(Path(policy["upstream"]) / "UniEgoMotion"))
    from module.uem_module import UEM_Module
    assert sha(Path(__file__)) == policy["implementation_sha256"], "Evaluator changed"
    assert sha(policy["checkpoint"]) == policy["checkpoint_sha256"]
    payload = torch.load(policy["checkpoint"], map_location="cpu", weights_only=False)
    cfg = CfgNode(payload["hyper_parameters"])
    cfg.TRAIN.ONLY_VALIDATE = True
    model = UEM_Module(cfg).cuda().eval()
    model.load_state_dict(payload["state_dict"], strict=True)
    assert model.model.input_feats == 220 and hasattr(model.model, "shape2llm")
    del payload
    root = a.output / a.condition
    (root / "predictions").mkdir(parents=True, exist_ok=True)
    lock = (root / "worker.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cases = policy["cases"][:a.limit] if a.limit else policy["cases"]
    started = time.monotonic()
    audits = []
    for i, row in enumerate(cases):
        neutral, image, trajectory, mean, std, gt, audit = inputs(policy, row)
        y = conditioning(neutral, image, trajectory, a.condition)
        assert ("traj" in y) == (a.condition != "rgb_only")
        assert ("img_embs" in y) == (a.condition != "traj_only")
        destination = root / "predictions" / (row["id"] + ".npz")
        if not destination.exists():
            torch.manual_seed(int(row["seed"]))
            torch.cuda.manual_seed_all(int(row["seed"]))
            with torch.inference_mode():
                prediction = model.sample(y, B=1, cond_scale=cfg.TRAIN.COND_SCALE)[0].cpu().numpy()
            assert prediction.shape == (97, 220) and np.isfinite(prediction).all()
            temporary = destination.with_suffix(".tmp.npz")
            np.savez(temporary, normalized=prediction, raw_body=prediction * (std + 1e-6) + mean,
                     gt_body=gt[:, MOTION_INDICES], neutral_joints=neutral,
                     checkpoint_sha256=policy["checkpoint_sha256"], condition=a.condition, seed=row["seed"])
            temporary.replace(destination)
        else:
            with np.load(destination, allow_pickle=False) as z:
                assert str(z["checkpoint_sha256"].item()) == policy["checkpoint_sha256"]
                assert str(z["condition"].item()) == a.condition and int(z["seed"]) == row["seed"]
                assert np.array_equal(z["neutral_joints"], neutral)
                assert np.array_equal(z["gt_body"], gt[:, MOTION_INDICES])
                assert z["raw_body"].shape == (97, 220) and np.isfinite(z["raw_body"]).all()
        audits.append(dict(id=row["id"], condition_keys=sorted(y), **audit))
        save(root / "input_audit.json", audits)
        progress(root, stage="sampling", complete=i + 1, total=len(cases),
                 elapsed_seconds=time.monotonic() - started, case=row["id"])
    save(root / ("SMOKE_COMPLETE.json" if a.limit else "INFERENCE_COMPLETE.json"),
         dict(cases=len(cases), checkpoint_sha256=policy["checkpoint_sha256"], condition=a.condition))


def report(a):
    from types import SimpleNamespace
    from motion_expert_joint_attention.decode_uniego_torch import decode_transforms
    from kimodo.metrics.foot_skate import FootSkateFromHeight
    from nymeria_baselines.body_shape import CONTACT_JOINTS
    from nymeria_baselines.evaluate_uem_phase3 import measure
    from motion_expert_phase3_edge.quantitative import aggregate
    from motion_expert_joint_attention.render_motion import render_single, write_mp4
    policy = read(a.output / "configuration.json")
    for condition in CONDITIONS:
        finished = read(a.output / condition / "INFERENCE_COMPLETE.json")
        assert finished["cases"] == 71 and finished["checkpoint_sha256"] == policy["checkpoint_sha256"]
    skate = FootSkateFromHeight(skeleton=SimpleNamespace(foot_joint_idx=list(CONTACT_JOINTS)), fps=20)
    rows, visuals = [], []
    for row in policy["cases"]:
        positions, transforms = {}, {}
        common = None
        for condition in CONDITIONS:
            with np.load(a.output / condition / "predictions" / (row["id"] + ".npz"), allow_pickle=False) as z:
                assert str(z["checkpoint_sha256"].item()) == policy["checkpoint_sha256"]
                assert int(z["seed"]) == row["seed"] and str(z["condition"].item()) == condition
                if common is None:
                    common = (z["gt_body"].copy(), z["neutral_joints"].copy())
                    transforms["GT"] = decode_transforms(torch.from_numpy(common[0])[None], n_joints=23)[0].numpy()
                assert np.array_equal(common[0], z["gt_body"]) and np.array_equal(common[1], z["neutral_joints"])
                transforms[condition] = decode_transforms(torch.from_numpy(z["raw_body"].copy())[None], n_joints=23)[0].numpy()
        positions = {name: tf[..., :3, 3] for name, tf in transforms.items()}
        rows.append(dict(id=row["id"], uuid=row["uuid"], start=row["start"], seed=row["seed"],
                         metrics={name: measure(p, positions["GT"], skate) for name, p in positions.items()}))
        save(a.output / "per_case.json", rows)
        if len(visuals) < a.visuals:
            panels = [render_single(positions[name], joint_parents=PARENTS, caption=name + " | supplied shape",
                                    color="tab:blue" if name == "GT" else "tab:red", width=448, height=448,
                                    max_frames=None, skip_joints=(), head_rotation0=transforms[name][0, 6, :3, :3])
                      for name in ("GT", *CONDITIONS)]
            frames = np.concatenate((np.concatenate(panels[:2], axis=2), np.concatenate(panels[2:], axis=2)), axis=1)
            folder = a.output / "qualitative"
            folder.mkdir(exist_ok=True)
            path = folder / (row["id"] + ".mp4")
            write_mp4(frames, str(path), fps=20)
            visuals.append(dict(id=row["id"], path=str(path), sha256=sha(path), frames=len(frames)))
        progress(a.output, stage="metrics", complete=len(rows), total=71)
    summary = {name: aggregate([r["metrics"][name] for r in rows]) for name in ("GT", *CONDITIONS)}
    save(a.output / "summary.json", dict(cases=71, checkpoint_step=policy["global_step"],
         aggregate=summary, caveat=policy["caveat"], qualitative=visuals))
    columns = [("mpjpe_m", 100, "MPJPE cm"), ("pa_mpjpe_m", 100, "PA-MPJPE cm"),
               ("root_err_m", 100, "Root ADE cm"), ("root_fde_m", 100, "Root FDE cm"),
               ("foot_skate_cm_s", 1, "Skate cm/s"), ("foot_depth_mean_cm", 1, "Foot depth cm"),
               ("all_feet_above5cm_frame_fraction", 100, "Airborne %")]
    with (a.output / "comparison.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["Condition", "Checkpoint step", *[c[2] for c in columns]])
        for name, values in summary.items():
            writer.writerow([name, policy["global_step"], *[values[k]["mean"] * scale for k, scale, _ in columns]])
    save(a.output / "COMPLETE.json", dict(cases=71, predictions=213, qualitative=len(visuals)))
    print((a.output / "comparison.csv").read_text(), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "sample", "report"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--reference", type=Path)
    p.add_argument("--condition", choices=CONDITIONS)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--visuals", type=int, default=4)
    a = p.parse_args()
    require_compute()
    torch.set_num_threads(4)
    a.output.mkdir(parents=True, exist_ok=True)
    if a.mode == "prepare" and (a.checkpoint is None or a.reference is None):
        p.error("prepare needs --checkpoint and --reference")
    if a.mode == "sample" and a.condition is None:
        p.error("sample needs --condition")
    try:
        globals()[a.mode](a)
    except Exception as e:
        import traceback
        root = a.output / a.condition if a.mode == "sample" else a.output
        save(root / "FAILED.json", dict(error=str(e), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
