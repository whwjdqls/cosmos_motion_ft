"""Matched held-out71 UniEgoMotion input ablations; never changes training."""
from __future__ import annotations

import argparse
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import runpy
import time
import traceback
import urllib.parse

import numpy as np
import torch

from nymeria_baselines.body_shape import BODY_JOINTS, CONTACT_JOINTS, MOTION_INDICES
from nymeria_baselines.evaluate_uem_phase3 import read, save, sha, measure
from nymeria_baselines.uem_visual_cache import frame_ids, require_compute


CONDITIONS = {
    "RGB_recon": ("video",),
    "recon": ("video_trajectory", "video", "trajectory"),
    "all": ("video_trajectory", "video", "trajectory"),
}
PAGE_ID = "3ffd2a44-f44b-806c-a79e-c5987d6fd1f3"
PAGE_URL = "https://app.notion.com/p/binne/Video2motion-comparison-" + PAGE_ID.replace("-", "")
COLUMNS = (
    ("mpjpe_m", 100, "MPJPE cm"),
    ("pa_mpjpe_m", 100, "PA-MPJPE cm"),
    ("root_err_m", 100, "Root ADE cm"),
    ("root_fde_m", 100, "Root FDE cm"),
    ("root_relative_ade_m", 100, "Relative root ADE cm"),
    ("foot_skate_cm_s", 1, "Skate cm/s"),
    ("foot_depth_mean_cm", 1, "Penetration cm"),
    ("all_feet_above5cm_frame_fraction", 100, "Airborne %"),
    ("accel_err_m_s2", 1, "Acceleration error m/s2"),
)


def progress(root, stage, **values):
    record = dict(stage=stage, time=time.time(), job=os.environ.get("SLURM_JOB_ID"), **values)
    save(root / "progress.json", record)
    print(json.dumps(record), flush=True)


def source(args):
    reference = read(args.reference / "configuration.json")
    cases = reference["cases"]
    if len(cases) != 71 or len({r["uuid"] for r in cases}) != 71:
        raise RuntimeError("Expected the existing 71-recording comparison, one T97 window each")
    return reference, cases


def sample(args):
    from yacs.config import CfgNode
    from module.uem_module import UEM_Module
    from nymeria_baselines.prepare import prepare_window

    reference, cases = source(args)
    root = args.output / args.variant
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "worker.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    checkpoint = args.runs_root / f"uniegomotion-{args.variant}_nymeria_soma_b64_350ep_20261019/last.ckpt"
    progress(root, "loading_checkpoint", checkpoint=str(checkpoint), variant=args.variant)
    before = checkpoint.stat()
    checkpoint_sha = sha(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    after = checkpoint.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Checkpoint changed while loading; refusing mixed provenance")
    cfg = CfgNode(payload["hyper_parameters"])
    cache = Path(reference["cache"])
    if Path(cfg.DATA.DATA_DIR).resolve() != cache.resolve():
        raise RuntimeError("Checkpoint uses a different dataset cache")
    if (cfg.DATA.WINDOW != 97 or cfg.DATA.REPRE_TYPE != "soma_body230"
            or cfg.MODEL.LEARN_TRAJ or cfg.DATA.COND_BETAS):
        raise RuntimeError("Expected body230 motion prediction without supplied GT shape")
    if not cfg.DATA.COND_IMG_FEAT or bool(cfg.DATA.COND_TRAJ) != (args.variant != "RGB_recon"):
        raise RuntimeError("Checkpoint modality flags disagree with the named variant")
    stats_sha = sha(cache / "train_stats.npz")
    if stats_sha != reference["uem_stats_sha256"]:
        raise RuntimeError("Normalization changed relative to the published comparison")
    with np.load(cache / "train_stats.npz", allow_pickle=False) as stats:
        mean, std = stats["motion_mean"][MOTION_INDICES], stats["motion_std"][MOTION_INDICES]
        cmean, cstd = stats["camera_mean"], stats["camera_std"]
    contract = reference["phase3_contract"]
    if sha(contract["motion_stats_mean"]) != contract["motion_mean"] or sha(contract["motion_stats_std"]) != contract["motion_std"]:
        raise RuntimeError("Reference GT motion statistics changed")
    pmean, pstd = np.load(contract["motion_stats_mean"]), np.load(contract["motion_stats_std"])
    index = read(cache / "index.json")
    test = [r for r in index["rows"] if r["split"] == "test"]
    arrays = {r["uuid"]: r["array"] for r in test}
    heldout = {r["uuid"] for r in cases}
    if not heldout <= arrays.keys() or heldout & {r["uuid"] for r in index["rows"] if r["split"] != "test"}:
        raise RuntimeError("Reference cases do not match the held-out recording split")
    features = Path(cfg.DATA.VISUAL_CACHE)
    ready = read(features / "READY.json")
    if (ready["fingerprint"] != cfg.DATA.VISUAL_CACHE_FINGERPRINT
            or ready["cache_index_sha256"] != sha(cache / "index.json")):
        raise RuntimeError("Visual feature cache does not match checkpoint/data provenance")
    policy = dict(
        variant=args.variant, checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
        epoch=int(payload["epoch"]), global_step=int(payload["global_step"]),
        reference=str(args.reference), reference_configuration_sha256=sha(args.reference / "configuration.json"),
        cases=cases, cache=str(cache), statistics_sha256=stats_sha,
        visual_cache=str(features), visual_fingerprint=ready["fingerprint"],
        visual_sampling=ready["sampling"], conditions=list(CONDITIONS[args.variant]),
        frames=97, fps=20, body_joints=list(BODY_JOINTS),
        sampler="Unchanged UEM_Module.sample / diffusion.p_sample_loop",
        diffusion_steps=int(cfg.MODEL.DIFFUSION_STEPS), cond_scale=cfg.TRAIN.COND_SCALE,
        weights="state_dict, matching the existing published evaluation; no EMA substitution",
        seed_policy="Original per-case seed, reset independently for each input condition",
        normalization="Checkpoint cache camera/motion mean and std, including native +1e-6 denominator",
        inputs="Observed modalities only; absent modalities omitted using native mask-token branches. No captions, GT motion, or GT body shape are supplied.",
        missing_input_caveat="For all/recon, video-only and trajectory-only are missing-input inference ablations, not their trained full-observation reconstruction setting.",
        implementation_sha256=sha(Path(__file__)),
    )
    if (root / "configuration.json").exists() and read(root / "configuration.json") != policy:
        raise RuntimeError("Refusing to combine different checkpoint/evaluation configurations")
    save(root / "configuration.json", policy)
    cfg.TRAIN.ONLY_VALIDATE = True
    model = UEM_Module(cfg)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.cuda().eval()
    del payload
    completed, audits, started = 0, [], time.monotonic()
    for row in cases:
        key, start = arrays[row["uuid"]], int(row["start"])
        motion = np.load(cache / "records" / (key + ".motion.npy"), mmap_mode="r")[start:start + 97]
        pose = np.load(cache / "records" / (key + ".pose_yup.npy"), mmap_mode="r")[start:start + 97]
        raw, camera = prepare_window(motion, pose, row["floor_offset"])
        with np.load(row["bundle"], allow_pickle=False) as bundle:
            if str(bundle["motion_mean_sha256"].item()) != contract["motion_mean"] or str(bundle["motion_std_sha256"].item()) != contract["motion_std"]:
                raise RuntimeError("Reference GT bundle normalization mismatch")
            gt = bundle["motion"] * pstd + pmean
        delta = float(np.max(np.abs(raw - gt)))
        if delta >= 2e-4:
            raise RuntimeError(f"GT/floor/canonicalization mismatch: {row['id']} ({delta})")
        trajectory = ((camera - cmean) / (cstd + 1e-6)).astype(np.float32)
        ids = np.load(features / "records" / (key + ".frame_ids.npy"), mmap_mode="r")
        wanted = frame_ids(start)
        slots = np.searchsorted(ids, wanted)
        if np.any(slots >= len(ids)) or not np.array_equal(ids[slots], wanted):
            raise RuntimeError("The exact reference window is missing cached causal RGB features")
        cached = np.load(features / "records" / (key + ".features.npy"), mmap_mode="r")
        images = np.array(cached[slots], dtype=np.float32)
        if images.shape != (97, 1024) or not np.isfinite(images).all() or not np.isfinite(trajectory).all():
            raise RuntimeError("Invalid RGB/trajectory condition")
        audits.append(dict(id=row["id"], gt_feature_max_difference=delta,
                           image_frame_ids=wanted.tolist(), feature_values_sha256=__import__("hashlib").sha256(images.tobytes()).hexdigest()))
        for condition in CONDITIONS[args.variant]:
            destination = root / condition / "predictions" / (row["id"] + ".npz")
            if not destination.exists():
                y = {"valid_frames": torch.ones(1, 97, dtype=torch.long, device="cuda")}
                if condition in ("trajectory", "video_trajectory"):
                    y["traj"] = torch.from_numpy(trajectory[None]).cuda()
                    y["traj_mask"] = torch.zeros(1, 97, dtype=torch.long, device="cuda")
                if condition in ("video", "video_trajectory"):
                    y["img_embs"] = torch.from_numpy(images[None]).cuda()
                    y["valid_img_embs"] = torch.ones(1, 97, dtype=torch.long, device="cuda")
                    y["img_mask"] = torch.zeros(1, 97, dtype=torch.long, device="cuda")
                torch.manual_seed(int(row["seed"]))
                torch.cuda.manual_seed_all(int(row["seed"]))
                with torch.inference_mode():
                    prediction = model.sample(y, B=1, cond_scale=cfg.TRAIN.COND_SCALE)[0].cpu().numpy()
                if prediction.shape != (97, 230) or not np.isfinite(prediction).all():
                    raise RuntimeError("Invalid native UEM prediction")
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".tmp.npz")
                np.savez(temporary, normalized=prediction,
                         raw_body=prediction[:, :220] * (std + 1e-6) + mean,
                         shape_pca10=prediction[:, 220:], gt_body=gt[:, MOTION_INDICES],
                         seed=int(row["seed"]), condition=condition,
                         checkpoint_sha256=checkpoint_sha)
                os.replace(temporary, destination)
            completed += 1
            progress(root, "sampling", variant=args.variant, complete=completed,
                     total=len(cases) * len(CONDITIONS[args.variant]), case=row["id"],
                     condition=condition, elapsed_seconds=time.monotonic() - started)
        save(root / "input_audit.json", audits)
    save(root / "INFERENCE_COMPLETE.json", dict(cases=71, predictions=completed,
                                                checkpoint_sha256=checkpoint_sha))
    progress(root, "inference_complete", variant=args.variant, complete=completed)


def report(args):
    from types import SimpleNamespace
    from decode_uniego_torch import decode_transforms
    from kimodo.metrics.foot_skate import FootSkateFromHeight
    from motion_expert_phase3_edge.quantitative import aggregate

    reference, cases = source(args)
    previous = {r["id"]: r for r in read(args.reference / "per_case.json")}
    if set(previous) != {r["id"] for r in cases}:
        raise RuntimeError("Published per-case comparison changed")
    contract = reference["phase3_contract"]
    if sha(contract["motion_stats_mean"]) != contract["motion_mean"] or sha(contract["motion_stats_std"]) != contract["motion_std"]:
        raise RuntimeError("Reference statistics changed")
    mean, std = np.load(contract["motion_stats_mean"]), np.load(contract["motion_stats_std"])
    models = {
        "GT": dict(step=None, epoch=None, input="reference", status="ground truth"),
        "Phase3 V2M": dict(step=200000, epoch=None, input="video + neutral shape", status="existing reference"),
        "traj_recon / trajectory": dict(step=reference["global_step"], epoch=reference["epoch"],
                                       input="trajectory", status="trained setting; existing predictions",
                                       checkpoint=reference["checkpoint"], checkpoint_sha256=reference["checkpoint_sha256"]),
    }
    policies = {}
    for variant, conditions in CONDITIONS.items():
        root = args.output / variant
        if not (root / "INFERENCE_COMPLETE.json").is_file():
            raise RuntimeError(f"Incomplete inference: {variant}")
        policy = read(root / "configuration.json")
        if policy["reference_configuration_sha256"] != sha(args.reference / "configuration.json"):
            raise RuntimeError("Inference used a different reference case configuration")
        policies[variant] = policy
        for condition in conditions:
            models[variant + " / " + condition] = dict(
                step=policy["global_step"], epoch=policy["epoch"], input=condition,
                status="trained reconstruction setting" if variant == "RGB_recon" or condition == "video_trajectory" else "missing-input inference ablation",
                checkpoint=policy["checkpoint"], checkpoint_sha256=policy["checkpoint_sha256"],
            )
    skate = FootSkateFromHeight(skeleton=SimpleNamespace(foot_joint_idx=list(CONTACT_JOINTS)), fps=20)
    rows = []
    for row in cases:
        old = previous[row["id"]]
        if (old["uuid"], old["start"], old["seed"]) != (row["uuid"], row["start"], row["seed"]):
            raise RuntimeError("Case order/seed changed relative to the existing comparison")
        with np.load(row["bundle"], allow_pickle=False) as bundle:
            gt_raw = bundle["motion"] * std + mean
        target = np.take(decode_transforms(torch.from_numpy(gt_raw)[None])[0].numpy(), BODY_JOINTS, axis=1)[..., :3, 3]
        metrics = {"GT": measure(target, target, skate),
                   "Phase3 V2M": old["metrics"]["Phase3_V2M"],
                   "traj_recon / trajectory": old["metrics"]["UniEgoMotion"]}
        for variant, conditions in CONDITIONS.items():
            for condition in conditions:
                path = args.output / variant / condition / "predictions" / (row["id"] + ".npz")
                with np.load(path, allow_pickle=False) as saved:
                    if (int(saved["seed"]) != row["seed"] or str(saved["checkpoint_sha256"].item()) != policies[variant]["checkpoint_sha256"]
                            or not np.array_equal(saved["gt_body"], gt_raw[:, MOTION_INDICES])):
                        raise RuntimeError("Prediction provenance or GT mismatch")
                    raw = saved["raw_body"].copy()
                positions = decode_transforms(torch.from_numpy(raw)[None], n_joints=23)[0].numpy()[..., :3, 3]
                metrics[variant + " / " + condition] = measure(positions, target, skate)
        rows.append(dict(id=row["id"], uuid=row["uuid"], start=row["start"], seed=row["seed"], metrics=metrics))
        save(args.output / "per_case.json", rows)
        progress(args.output, "metrics", complete=len(rows), total=71)
    summary = dict(
        cases=71, frames=97, fps=20, body_joints=list(BODY_JOINTS), models=models,
        aggregate={name: aggregate([r["metrics"][name] for r in rows]) for name in models},
        reference=str(args.reference), reference_configuration_sha256=sha(args.reference / "configuration.json"),
        reused_per_case_sha256=sha(args.reference / "per_case.json"),
        alignment=reference["alignment"],
        unsupported={"traj_recon": ["video", "video_trajectory"], "RGB_recon": ["trajectory", "video_trajectory"]},
        caveats=["Different training lengths; not a matched-step architecture comparison.",
                 "Single-modality specialists have no trained pathway for the other modality; unsupported combinations are N/A, not fabricated predictions.",
                 "Video-only and trajectory-only inputs for all/recon are missing-input ablations, not their trained full reconstruction setting.",
                 "UEM video features use the original causal 5fps DINO hold at 20fps; camera inputs contain all 97 measured poses.",
                 "GT physics is nonzero. Lower skating alone does not demonstrate better reconstruction.",
                 "Native frame-0 canonical coordinates; no prediction grounding or fitting except explicitly labeled PA-MPJPE.",
                 "Only metrics requested here are generated; existing Notion videos are preserved."],
    )
    save(args.output / "summary.json", summary)
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["Model", "Training step", "Input", "Setting"] + [c[2] for c in COLUMNS])
    lines = ["Model | Step | " + " | ".join(c[2] for c in COLUMNS)]
    for name, meta in models.items():
        values = [f"{summary['aggregate'][name][k]['mean'] * scale:.4f}" for k, scale, _ in COLUMNS]
        writer.writerow([name, meta["step"], meta["input"], meta["status"]] + values)
        lines.append(" | ".join([name, str(meta["step"] or "-")] + values))
    (args.output / "comparison.csv").write_text(stream.getvalue())
    (args.output / "comparison.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    save(args.output / "METRICS_COMPLETE.json", dict(cases=71, rows=len(models), summary=str(args.output / "summary.json")))


def publish(args):
    if not (args.output / "METRICS_COMPLETE.json").exists():
        raise RuntimeError("Do not publish incomplete metrics")
    summary = read(args.output / "summary.json")
    helpers = runpy.run_path(str(args.notion_helper))
    api = helpers["Notion"](helpers["read_token"]())
    title = "UniEgomotion reconstruction input ablations | " + args.output.name
    cursor = None
    while True:
        query = {"page_size": 100}
        if cursor:
            query["start_cursor"] = cursor
        response = api.request("GET", "/blocks/" + PAGE_ID + "/children?" + urllib.parse.urlencode(query))
        for block in response["results"]:
            if block["type"] == "heading_2" and helpers["plain_text"](block["heading_2"].get("rich_text", [])) == title:
                save(args.output / "notion_publication.json", dict(status="existing_section", url=PAGE_URL, heading_id=block["id"]))
                return
        if not response.get("has_more"):
            break
        cursor = response["next_cursor"]
    text = helpers["rich_text"]
    def paragraph(value):
        return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": text(value)}}
    def table(values):
        return {"object": "block", "type": "table", "table": {
            "table_width": len(values[0]), "has_column_header": True, "has_row_header": False,
            "children": [{"object": "block", "type": "table_row", "table_row": {"cells": [text(str(v)) for v in row]}} for row in values]}}
    main = [["Model", "Step", "MPJPE cm", "PA-MPJPE cm", "Root ADE cm", "Skate cm/s", "Penetration cm"]]
    extra = [["Model", "Root FDE cm", "Relative root ADE cm", "Airborne %", "Accel error m/s2"]]
    for name, meta in summary["models"].items():
        values = summary["aggregate"][name]
        main.append([name, meta["step"] or "GT"] + [f"{values[k]['mean'] * scale:.2f}" for k, scale in
                    (("mpjpe_m", 100), ("pa_mpjpe_m", 100), ("root_err_m", 100), ("foot_skate_cm_s", 1), ("foot_depth_mean_cm", 1))])
        extra.append([name] + [f"{values[k]['mean'] * scale:.2f}" for k, scale in
                    (("root_fde_m", 100), ("root_relative_ade_m", 100), ("all_feet_above5cm_frame_fraction", 100), ("accel_err_m_s2", 1))])
    children = [{"object": "block", "type": "heading_2", "heading_2": {"rich_text": text(title)}},
                paragraph("Same 71 fixed held-out T97 windows, per-case seeds, floor calibration, and 23 body joints as the comparison above. Outputs are body motion. Native 1,000-step UniEgoMotion sampling and checkpoint normalization are unchanged; state_dict weights, no EMA substitution or supplied GT shape."),
                table(main), table(extra)]
    children += [paragraph(value) for value in summary["caveats"]]
    children.append(paragraph("Local provenance, individual cases, and CSV: " + str(args.output)))
    response = api.request("PATCH", "/blocks/" + PAGE_ID + "/children", {"children": children})
    save(args.output / "notion_publication.json", dict(status="published", url=PAGE_URL,
                                                      block_ids=[r["id"] for r in response["results"]],
                                                      summary_sha256=sha(args.output / "summary.json")))
    print("Published completed metrics: " + PAGE_URL, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("sample", "report", "publish"))
    parser.add_argument("--variant", choices=tuple(CONDITIONS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--notion-helper", type=Path)
    args = parser.parse_args()
    require_compute()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        if args.mode == "sample" and args.variant is None:
            raise ValueError("Sampling requires --variant")
        {"sample": sample, "report": report, "publish": publish}[args.mode](args)
    except BaseException as error:
        failure_root = args.output / args.variant if args.variant else args.output
        save(failure_root / "FAILED.json", dict(error=str(error), traceback=traceback.format_exc(), time=time.time()))
        raise


if __name__ == "__main__":
    main()
