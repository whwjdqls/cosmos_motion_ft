"""FID/R32 for saved UEM, EgoAllo and Phase3 reconstruction on matched71."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
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

from nymeria_baselines.body23_tmr_adapter import KEEP, MISSING, complete_body, fit_template, neutral_from_pca
from nymeria_baselines.evaluate_uem_phase3 import read, save, sha
from nymeria_baselines.uem_visual_cache import require_compute


PAGE_ID = "3ffd2a44-f44b-806c-a79e-c5987d6fd1f3"
PAGE_URL = "https://app.notion.com/p/binne/Video2motion-comparison-" + PAGE_ID.replace("-", "")


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **values)
    os.replace(temporary, path)


def progress(args, stage, **values):
    record = dict(stage=stage, time=time.time(), host=os.uname().nodename,
                  pid=os.getpid(), job=os.environ.get("SLURM_JOB_ID"),
                  step_id=os.environ.get("SLURM_STEP_ID"), **values)
    save(args.output / "progress.json", record)
    print(json.dumps(record), flush=True)


def fid(motion, target):
    from kimodo.metrics.tmr import calculate_activation_statistics, calculate_frechet_distance
    if np.array_equal(motion, target):
        return 0.0
    ma, ca = calculate_activation_statistics(motion)
    mb, cb = calculate_activation_statistics(target)
    value = float(calculate_frechet_distance(ma, ca, mb, cb))
    if not np.isfinite(value) or value < -1e-6:
        raise RuntimeError("Invalid FID")
    return max(0.0, value)


def retrieval_details(motion, text, ids):
    from retrieval32 import paired_retrieval32
    result = paired_retrieval32(motion, text)
    details = []
    for seed in (0, 1, 2):
        order = np.random.default_rng(seed).permutation(len(ids))
        used = len(ids) // 32 * 32
        groups = order[:used].reshape(-1, 32)
        m, t = motion[groups].astype(np.float64), text[groups].astype(np.float64)
        distances = np.maximum((t*t).sum(-1)[:, :, None] + (m*m).sum(-1)[:, None, :]
                               - 2*np.einsum("gqd,gkd->gqk", t, m), 0)
        ordering = np.argsort(distances, axis=-1, kind="stable")
        ranks = np.argmax(ordering == np.arange(32)[None, :, None], axis=-1) + 1
        details.append(dict(seed=seed, groups=groups.tolist(), group_case_ids=np.asarray(ids)[groups].tolist(),
                            paired_ranks=ranks.tolist(), dropped_case_ids=np.asarray(ids)[order[used:]].tolist()))
    return result, details


def get_specs(args, reference):
    return [
        dict(key="traj_recon", label="UniEgomotion-traj_recon", kind="uem", root=str(args.reference),
             condition=None, trained_setting=True),
        dict(key="egoallo", label="EgoAllo", kind="egoallo", root=str(args.egoallo),
             condition=None, trained_setting=True),
        dict(key="phase3_v2m", label="Phase3 V2M", kind="phase3", root=reference["phase3_root"],
             condition="plain", trained_setting=True),
        *[dict(key=variant + "__" + condition, label="UniEgomotion-" + variant + " / " + condition,
               kind="uem", root=str(args.uem / variant), condition=condition,
               trained_setting=variant == "RGB_recon" or condition == "video_trajectory")
          for variant, conditions in (("RGB_recon", ("video",)),
                                      ("recon", ("video_trajectory", "video", "trajectory")),
                                      ("all", ("video_trajectory", "video", "trajectory")))
          for condition in conditions],
    ]


def publish(args, summary):
    helper = runpy.run_path(str(args.notion_helper))
    api = helper["Notion"](helper["read_token"]())
    title = "FID and R-precision | common body23 completion | " + args.output.name
    cursor = None
    while True:
        query = {"page_size": 100}
        if cursor:
            query["start_cursor"] = cursor
        response = api.request("GET", "/blocks/" + PAGE_ID + "/children?" + urllib.parse.urlencode(query))
        for block in response["results"]:
            if block["type"] == "heading_2" and helper["plain_text"](block["heading_2"].get("rich_text", [])) == title:
                save(args.output / "notion_publication.json", dict(status="existing_section", url=PAGE_URL, id=block["id"]))
                return
        if not response.get("has_more"):
            break
        cursor = response["next_cursor"]
    rt = helper["rich_text"]
    def paragraph(value):
        return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": rt(value)}}
    def table(rows):
        return {"object": "block", "type": "table", "table": {
            "table_width": len(rows[0]), "has_column_header": True, "has_row_header": False,
            "children": [{"object": "block", "type": "table_row", "table_row": {"cells": [rt(str(x)) for x in row]}} for row in rows]}}
    rows = [["Model / input", "Training step", "FID vs completed GT", "R@1 %", "R@2 %", "R@3 %", "Setting"]]
    for item in summary["models"]:
        r = item["retrieval"]
        rows.append([item["label"], item.get("training_step") or "GT", f"{item['FID']:.6f}",
                     *[f"{r[f'R32_R{k:02d}_mean']:.2f} +/- {r[f'R32_R{k:02d}_std']:.2f}" for k in (1, 2, 3)],
                     item["setting"]])
    diagnostic = summary["adapter_diagnostic"]
    drows = [["GT representation", "R@1 %", "R@2 %", "R@3 %"]]
    for name, key in (("Original SOMA30 GT", "original_GT"), ("Common body23-completed GT", "completed_GT")):
        drows.append([name] + [f"{diagnostic[key]['retrieval'][f'R32_R{k:02d}_mean']:.2f}" for k in (1, 2, 3)])
    blocks = [{"object": "block", "type": "heading_2", "heading_2": {"rich_text": rt(title)}},
              paragraph("Same 71 held-out T97 windows and captions. Reuses the validation-selected Nymeria shape-aware TMR evaluator and its original normalization. All headline rows, including GT and Phase3, use the same body23-to-SOMA30 completion. No model was retrained or conditioned on additional GT."),
              table(rows), table(drows),
              paragraph(f"Adapter diagnostic: FID(completed GT, original GT) = {diagnostic['completed_vs_original_FID']:.6f}; mean paired embedding cosine = {diagnostic['paired_cosine_mean']:.6f}. Retained body-joint error is exactly zero."),
              paragraph("R@1/2/3 uses 32 paired candidates and three grouping seeds (0,1,2), reporting mean +/- sample SD. Each seed scores 64 cases and drops seven; FID uses all 71. This is a small-sample reconstruction/semantic-alignment diagnostic, not a stable full-test FID or a test of text conditioning."),
              paragraph("Jaw, eyes and hand endpoints are completed from training-only mean attachment offsets, scaled by each model's predicted neutral body shape and transformed with predicted parent rotations. No held-out omitted joints or GT skeleton are used for UEM/EgoAllo predictions. GT and shape-conditioned Phase3 retain their own known neutral body shape. Clip-mean predicted shape is used for evaluator metadata; original body motion is never changed."),
              paragraph("Missing-input settings are labeled separately. Training steps differ. Scores are not directly comparable with historical Phase2 FID/R@P because the cohort and body-completion protocol differ. MPJPE/skating/penetration results above remain unchanged."),
              paragraph("Configuration, training-only adapter provenance, all embeddings, per-case ranks, GT checks and comparison CSV: " + str(args.output))]
    response = api.request("PATCH", "/blocks/" + PAGE_ID + "/children", {"children": blocks})
    save(args.output / "notion_publication.json", dict(status="published", url=PAGE_URL,
                                                      block_ids=[b["id"] for b in response["results"]],
                                                      summary_sha256=sha(args.output / "summary.json")))


def main(args):
    from kimodo.skeleton import SOMASkeleton30
    from nymeria_eval import NymeriaTMREmbedder
    from decode_uniego_torch import decode_transforms

    require_compute()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "worker.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    request = read(args.evaluator_request)
    reference = read(args.reference / "configuration.json")
    cases = reference["cases"]
    if len(cases) != 71 or len({r["uuid"] for r in cases}) != 71:
        raise RuntimeError("Expected the unchanged fixed71 cohort")
    ids = [r["id"] for r in cases]
    cache = Path(reference["cache"])
    if sha(cache / "index.json") != reference["cache_index_sha256"] or sha(cache / "train_stats.npz") != reference["uem_stats_sha256"]:
        raise RuntimeError("Baseline cache provenance changed")
    contract = reference["phase3_contract"]
    if sha(contract["motion_stats_mean"]) != contract["motion_mean"] or sha(contract["motion_stats_std"]) != contract["motion_std"]:
        raise RuntimeError("Phase3/GT denormalization contract changed")
    mean, std = np.load(contract["motion_stats_mean"]), np.load(contract["motion_stats_std"])
    with np.load(cache / "body_shape16.npz", allow_pickle=False) as archive:
        shape = {key: archive[key].astype(np.float64) for key in ("mean", "scale", "components")}
    skeleton = SOMASkeleton30()
    parents = [int(x) for x in skeleton.joint_parents]
    progress(args, "training_only_adapter")
    template_path = args.output / "adapter.json"
    if template_path.exists():
        template = read(template_path)
    else:
        template = fit_template(cache, parents, {r["uuid"] for r in cases})
        save(template_path, template)
    if template["parents"] != parents or {r["uuid"] for r in template["records"]} & {r["uuid"] for r in cases}:
        raise RuntimeError("Adapter topology or train/test separation failure")
    configuration = dict(
        schema="reconstruction_tmr_body23_completed_v1", cases=cases,
        evaluator_checkpoint=request["evaluator_checkpoint"], evaluator_sha256=sha(request["evaluator_checkpoint"]),
        evaluator_stats=request["evaluator_stats"], text_cache=request["text_cache"],
        text_cache_sha256=sha(request["text_cache"]), evaluator_request_sha256=sha(args.evaluator_request),
        reference=str(args.reference), reference_configuration_sha256=sha(args.reference / "configuration.json"),
        baseline_cache_index_sha256=sha(cache / "index.json"), shape_basis_sha256=sha(cache / "body_shape16.npz"),
        adapter_sha256=sha(template_path), source_fps=20, evaluator_fps=30, frame_count=97,
        features="Raw completed SOMA30 -> native 20-to-30fps resampling -> canonical normalized TMR186, once",
        shape_policy="Clip-mean predicted PCA shape for UEM/EgoAllo; original conditioned body shape for Phase3; own shape for GT. Same train-only completion in all cases.",
        FID="All71, against completed GT in the unchanged Nymeria evaluator embedding space; small-sample diagnostic",
        retrieval=dict(candidates=32, seeds=[0, 1, 2], positive="paired_only", drop_last=True),
        source_code_sha256={p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name("body23_tmr_adapter.py"))},
    )
    destination = args.output / "configuration.json"
    if destination.exists() and read(destination) != configuration:
        raise RuntimeError("Refusing to mix evaluation/adapter identities")
    save(destination, configuration)
    signature = stable_hash(configuration)
    progress(args, "loading_evaluator")
    embedder = NymeriaTMREmbedder(request["evaluator_checkpoint"], request["evaluator_stats"], request["text_cache"], "cuda")
    if embedder.fps != 30:
        raise RuntimeError("Expected the original 30fps Nymeria evaluator")
    test_manifest = Path(request["data_dir"]) / "test.jsonl"
    if sha(test_manifest) != request["expected_manifest_sha256"] or embedder.contract["manifests"]["test"] != request["expected_manifest_sha256"]:
        raise RuntimeError("Evaluator held-out manifest changed")
    evaluator_test = {json.loads(line)["uuid"] for line in test_manifest.read_text().splitlines()}
    if not {r["uuid"] for r in cases} <= evaluator_test:
        raise RuntimeError("A comparison recording is not in the evaluator's held-out split")
    captions = [r["caption"] for r in cases]
    text = embedder.embed_text(captions).cpu().numpy()
    atomic_npz(args.output / "text_embeddings.npz", embedding=text, case_id=np.asarray(ids))
    gt_raw, gt_joints, gt_rotations, gt_neutrals, bundles = [], [], [], [], []
    for row in cases:
        with np.load(row["bundle"], allow_pickle=False) as bundle:
            raw = bundle["motion"] * std + mean
            neutral = bundle["neutral"].copy()
        transforms = decode_transforms(torch.from_numpy(raw)[None])[0].numpy()
        gt_raw.append(raw)
        gt_joints.append(transforms[..., :3, 3])
        gt_rotations.append(transforms[..., :3, :3])
        gt_neutrals.append(neutral)
        bundles.append(dict(id=row["id"], bundle_sha256=sha(row["bundle"])))
    save(args.output / "reference_inputs.json", bundles)
    def embed(joints, neutral):
        values = []
        for start in range(0, len(joints), 16):
            j = torch.from_numpy(np.stack(joints[start:start + 16]).astype(np.float32))
            n = torch.from_numpy(np.stack(neutral[start:start + 16]).astype(np.float32))
            values.append(embedder.embed_motion(j, n, lengths=[97] * len(j), source_fps=20).cpu().numpy())
        result = np.concatenate(values)
        if result.shape != (71, 256) or not np.isfinite(result).all():
            raise RuntimeError("Invalid evaluator embedding")
        return result
    original_gt = embed(gt_joints, gt_neutrals)
    completed, neutrals, errors = [], [], []
    for joints, rotations, neutral in zip(gt_joints, gt_rotations, gt_neutrals, strict=True):
        full, rest = complete_body(joints[:, KEEP], rotations[:, KEEP], neutral[KEEP], template)
        completed.append(full)
        neutrals.append(rest)
        errors.append(float(np.linalg.norm(full[:, MISSING] - joints[:, MISSING], axis=-1).mean()))
    completed_gt = embed(completed, neutrals)
    original_retrieval, original_ranks = retrieval_details(original_gt, text, ids)
    gt_retrieval, gt_ranks = retrieval_details(completed_gt, text, ids)
    diagnostic = dict(
        original_GT=dict(retrieval=original_retrieval, self_FID=0.0),
        completed_GT=dict(retrieval=gt_retrieval, self_FID=0.0),
        completed_vs_original_FID=fid(completed_gt, original_gt),
        paired_cosine_mean=float((completed_gt * original_gt).sum(-1).mean()),
        paired_embedding_l2_mean=float(np.linalg.norm(completed_gt - original_gt, axis=-1).mean()),
        retained_body_max_error_m=0.0, missing_joint_mean_error_cm=float(np.mean(errors) * 100),
        missing_joint_per_case_error_cm=(np.asarray(errors) * 100).tolist(),
        train_only_calibration_recordings=template["train_recordings"],
        uncertainty="71 samples in a 256D embedding space; covariance is rank-deficient and FID is descriptive, not a reliable benchmark.",
    )
    atomic_npz(args.output / "GT_embeddings.npz", original=original_gt, completed=completed_gt, case_id=np.asarray(ids), signature=signature)
    save(args.output / "GT_ranks.json", dict(original=original_ranks, completed=gt_ranks))
    save(args.output / "adapter_diagnostic.json", diagnostic)
    progress(args, "GT_scored", adapter_FID=diagnostic["completed_vs_original_FID"],
             original_GT_R3=original_retrieval["R32_R03_mean"], completed_GT_R3=gt_retrieval["R32_R03_mean"])
    specs = get_specs(args, reference)
    pending = specs.copy()
    results = [dict(key="GT", label="GT (common completion)", FID=0.0, retrieval=gt_retrieval,
                    training_step=None, setting="ground truth")]
    deadline = time.monotonic() + 6 * 3600
    while pending:
        made_progress = False
        for spec in pending.copy():
            root = Path(spec["root"])
            condition = spec["condition"]
            if spec["kind"] == "phase3":
                manifest_path = root / condition / "manifest.json"
                manifest = read(manifest_path)
                manifest_rows = {r["id"]: r for r in manifest["cases"]}
                policy = dict(checkpoint=manifest["checkpoint"], checkpoint_sha256=manifest["checkpoint_sha256"], global_step=200000)
                if manifest["checkpoint_sha256"] != reference["phase3_checkpoint_sha256"]:
                    raise RuntimeError("Phase3 reference changed")
                paths = [Path(manifest_rows[r["id"]]["output"]) for r in cases]
                provenance_hash = sha(manifest_path)
            else:
                if not (root / "INFERENCE_COMPLETE.json").exists():
                    if (root / "FAILED.json").exists():
                        raise RuntimeError("Source inference failed: " + str(root / "FAILED.json"))
                    continue
                policy = read(root / "configuration.json")
                prediction_root = root / condition / "predictions" if condition else root / "predictions"
                paths = [prediction_root / (row["id"] + ".npz") for row in cases]
                provenance_hash = sha(root / "configuration.json")
                if policy["cases"] != cases:
                    raise RuntimeError("Model uses different fixed windows or seeds")
            if not all(p.is_file() for p in paths):
                raise RuntimeError("Completed inference is missing prediction files")
            model_root = args.output / spec["key"]
            model_root.mkdir(exist_ok=True)
            identity = dict(signature=signature, source_configuration_sha256=provenance_hash,
                            predictions=[dict(path=str(p), sha256=sha(p)) for p in paths])
            cached_summary = model_root / "summary.json"
            if cached_summary.exists():
                result = read(cached_summary)
                if result["input_identity"] != identity:
                    raise RuntimeError("A scored prediction source changed")
            else:
                positions, rest_shapes, audits = [], [], []
                for i, (row, path) in enumerate(zip(cases, paths, strict=True)):
                    with np.load(path, allow_pickle=False) as a:
                        if spec["kind"] == "uem":
                            raw = a["raw_body"].copy()
                            transform = decode_transforms(torch.from_numpy(raw)[None], n_joints=23)[0].numpy()
                            joints, rotations = transform[..., :3, 3], transform[..., :3, :3]
                            neutral = neutral_from_pca(a["shape_pca10"], shape)
                            if int(a["seed"]) != row["seed"] or not np.array_equal(a["gt_body"], gt_raw[i][:, __import__("nymeria_baselines.body_shape", fromlist=["MOTION_INDICES"]).MOTION_INDICES]):
                                raise RuntimeError("UEM prediction GT/seed mismatch")
                        elif spec["kind"] == "egoallo":
                            joints, rotations = a["joints"].copy(), a["rotations"].copy()
                            neutral = neutral_from_pca(a["betas"], shape)
                            if int(a["seed"]) != row["seed"]:
                                raise RuntimeError("EgoAllo seed mismatch")
                        else:
                            raw = a["generated"] * std + mean
                            transform = decode_transforms(torch.from_numpy(raw)[None])[0].numpy()
                            joints = transform[..., :3, 3][:, KEEP]
                            rotations = transform[..., :3, :3][:, KEEP]
                            neutral = gt_neutrals[i][KEEP]
                            if not np.array_equal(a["gt_motion"] * std + mean, gt_raw[i]):
                                raise RuntimeError("Phase3 GT mismatch")
                    full, rest = complete_body(joints, rotations, neutral, template)
                    positions.append(full)
                    rest_shapes.append(rest)
                    audits.append(dict(id=row["id"], retained_body_max_error_m=float(np.max(np.abs(full[:, KEEP] - joints))),
                                       neutral_source="conditioned neutral body" if spec["kind"] == "phase3" else "clip-mean predicted shape PCA"))
                embeddings = embed(positions, rest_shapes)
                retrieval, ranks = retrieval_details(embeddings, text, ids)
                result = dict(key=spec["key"], label=spec["label"], input_identity=identity,
                              checkpoint=policy["checkpoint"], checkpoint_sha256=policy["checkpoint_sha256"],
                              training_step=policy.get("global_step", policy.get("step")),
                              setting="trained reconstruction setting" if spec["trained_setting"] else "missing-input inference ablation",
                              FID=fid(embeddings, completed_gt), retrieval=retrieval,
                              paired_GT_embedding_cosine_mean=float((embeddings * completed_gt).sum(-1).mean()))
                atomic_npz(model_root / "embeddings.npz", embedding=embeddings, case_id=np.asarray(ids), signature=signature)
                atomic_npz(model_root / "completed_geometry.npz", joints=np.stack(positions), neutral=np.stack(rest_shapes), case_id=np.asarray(ids))
                save(model_root / "per_case_adapter_audit.json", audits)
                save(model_root / "retrieval_groups.json", ranks)
                save(cached_summary, result)
            results.append(result)
            pending.remove(spec)
            made_progress = True
            save(args.output / "partial_summary.json", dict(adapter_diagnostic=diagnostic, models=results, pending=[s["key"] for s in pending]))
            progress(args, "model_scored", model=spec["key"], FID=result["FID"], R3=result["retrieval"]["R32_R03_mean"], pending=len(pending))
        if pending and not made_progress:
            if time.monotonic() > deadline:
                raise TimeoutError("Source inference did not finish within six hours; saved scores are preserved")
            source_exit = args.uem / "exit_status"
            if source_exit.exists() and source_exit.read_text().strip() != "0":
                raise RuntimeError("Source UEM evaluator terminated unsuccessfully")
            progress(args, "waiting_for_source_predictions", pending=[s["key"] for s in pending])
            time.sleep(30)
    ordered = [results[0]] + [next(r for r in results if r["key"] == s["key"]) for s in specs]
    summary = dict(status="complete", cases=71, configuration_sha256=sha(destination),
                   adapter_diagnostic=diagnostic, models=ordered,
                   interpretation="Same evaluator as Phase2, different fixed71 cohort and common leaf completion. Reconstruction semantics, not text-conditioned generation. FID is small-sample; GT is always reported.")
    save(args.output / "summary.json", summary)
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["Model", "Step", "Setting", "FID", "R1_mean", "R1_SD", "R2_mean", "R2_SD", "R3_mean", "R3_SD"])
    lines = ["Model | Step | FID | R@1 | R@2 | R@3"]
    for item in ordered:
        r = item["retrieval"]
        values = [r[f"R32_R{k:02d}_{stat}"] for k in (1, 2, 3) for stat in ("mean", "std")]
        writer.writerow([item["label"], item.get("training_step"), item["setting"], item["FID"], *values])
        lines.append(" | ".join([item["label"], str(item.get("training_step") or "GT"), f"{item['FID']:.6f}",
                                 *[f"{r[f'R32_R{k:02d}_mean']:.2f}" for k in (1, 2, 3)]]))
    (args.output / "comparison.csv").write_text(stream.getvalue())
    (args.output / "comparison.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    save(args.output / "METRICS_COMPLETE.json", dict(cases=71, rows=len(ordered), summary_sha256=sha(args.output / "summary.json")))
    publish(args, summary)
    save(args.output / "COMPLETE.json", dict(cases=71, models=len(ordered)-1, notion=PAGE_URL))
    progress(args, "complete", cases=71, models=len(ordered)-1, notion=PAGE_URL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "reference", "egoallo", "uem", "evaluator-request", "notion-helper"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        with torch.inference_mode():
            main(args)
    except BaseException as error:
        save(args.output / "FAILED.json", dict(error=str(error), traceback=traceback.format_exc(), time=time.time()))
        raise
