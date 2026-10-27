"""Compare saved Phase3 caption/guidance predictions with matched UEM scores."""
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

from nymeria_baselines.body23_tmr_adapter import KEEP, complete_body
from nymeria_baselines.evaluate_reconstruction_tmr import (
    PAGE_ID, PAGE_URL, atomic_npz, fid, retrieval_details, stable_hash,
)
from nymeria_baselines.evaluate_uem_phase3 import read, save, sha
from nymeria_baselines.uem_visual_cache import require_compute


PHYSICAL = ["MPJPE cm", "PA-MPJPE cm", "Root ADE cm", "Root FDE cm",
            "Relative root ADE cm", "Skate cm/s", "Penetration cm", "Airborne %"]
METRICS = [("body23_mpjpe_m", 100), ("body23_pa_mpjpe_m", 100),
           ("body23_root_err_m", 100), ("root_fde_m", 100),
           ("root_relative_ade_m", 100), ("foot_skate_cm_s", 1),
           ("foot_depth_mean_cm", 1), ("all_feet_above5cm_frame_fraction", 100)]
SCALES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]


def progress(args, stage, **details):
    value = dict(stage=stage, time=time.time(), pid=os.getpid(),
                 host=os.uname().nodename, job=os.environ.get("SLURM_JOB_ID"), **details)
    save(args.output / "progress.json", value)
    print(json.dumps(value), flush=True)


def physical(rows, setting):
    return {label: float(np.mean([r[setting][metric] for r in rows]) * scale)
            for label, (metric, scale) in zip(PHYSICAL, METRICS, strict=True)}


def publish(args, summary):
    helper = runpy.run_path(str(args.notion_helper))
    api = helper["Notion"](helper["read_token"]())
    title = "Phase3 caption / guidance versus UniEgomotion | matched71 | " + args.output.name
    cursor = None
    while True:
        query = {"page_size": 100}
        if cursor:
            query["start_cursor"] = cursor
        page = api.request("GET", "/blocks/" + PAGE_ID + "/children?" + urllib.parse.urlencode(query))
        for block in page["results"]:
            if block["type"] == "heading_2" and helper["plain_text"](block["heading_2"].get("rich_text", [])) == title:
                raise RuntimeError("Comparison section already exists; refusing a duplicate append")
        if not page.get("has_more"):
            break
        cursor = page["next_cursor"]
    rt = helper["rich_text"]
    def paragraph(text):
        return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": rt(text)}}
    def table(rows):
        return {"object": "block", "type": "table", "table": {
            "table_width": len(rows[0]), "has_column_header": True, "has_row_header": False,
            "children": [{"object": "block", "type": "table_row", "table_row": {
                "cells": [rt(str(cell)) for cell in row]}} for row in rows]}}
    lookup = {r["key"]: r for r in summary["rows"]}
    headline = [lookup[key] for key in summary["headline_keys"]]
    geometry = [["Model / setting", "Observed input", "MPJPE cm", "PA-MPJPE cm", "Root ADE cm", "Skate cm/s", "Penetration cm"]]
    semantics = [["Model / setting", "FID", "R@1 %", "R@2 %", "R@3 %"]]
    for row in headline:
        geometry.append([row["label"], row["input"]] + [f"{row['physical'][k]:.2f}" for k in
                         ("MPJPE cm", "PA-MPJPE cm", "Root ADE cm", "Skate cm/s", "Penetration cm")])
        r = row["retrieval"]
        semantics.append([row["label"], f"{row['FID']:.6f}"] +
                         [f"{r[f'R32_R{k:02d}_mean']:.2f} +/- {r[f'R32_R{k:02d}_std']:.2f}" for k in (1, 2, 3)])
    sweep = [["Phase3 input", "Video scale", "MPJPE cm", "FID", "R@1 %", "R@2 %", "R@3 %", "Skate cm/s", "Penetration cm"]]
    for row in summary["rows"]:
        if row["family"] != "phase3":
            continue
        r = row["retrieval"]
        sweep.append(["video + caption" if row["caption"] else "video", row["video_scale"],
                      f"{row['physical']['MPJPE cm']:.2f}", f"{row['FID']:.6f}",
                      *[f"{r[f'R32_R{k:02d}_mean']:.2f}" for k in (1, 2, 3)],
                      f"{row['physical']['Skate cm/s']:.2f}", f"{row['physical']['Penetration cm']:.2f}"])
    blocks = [
        {"object": "block", "type": "heading_2", "heading_2": {"rich_text": rt(title)}},
        paragraph("Matched 71 fixed T97 windows, one per test recording, not all windows in the test recordings. Existing predictions are reused: no training, resampling, grounding, or head inpainting. All rows share Body23 physical metrics and the same training-only body23-to-SOMA30 completion for the Nymeria TMR evaluator."),
        table(geometry), table(semantics),
        paragraph("Scales 1 and 2 are shown together to distinguish the native baseline from the exploratory guidance setting. All seven scales are reported below; scale 2 is not claimed to be independently validation-selected."),
        table(sweep),
        paragraph("Important input differences: Phase3 receives video and supplied neutral body shape, with captions only in the labeled arm. UniEgomotion predicts shape; its trajectory variants receive all 97 measured camera poses. Caption access also changes the semantic-retrieval task. These are system-level comparisons, not input-matched architecture rankings. Training steps differ and are preserved in the CSV."),
        paragraph("The video-guidance reference disables entire learned modality bridges, including their within-motion attention. This is bridge-disabled extrapolation, not a classifier-free null trained with video dropout. Captions remain in both branches of the caption arm. Scale 0 has no video-to-motion bridge path; scale 1 is native inference. Caption-conditioned V2M was not a training task for this baseline checkpoint."),
        paragraph("FID uses all 71 embeddings and is a small-sample diagnostic. R@1/2/3 uses the unchanged 32-candidate, three-seed protocol: 64 cases per seed and seven dropped. GT is reported with the same completion; its original-versus-completed diagnostic remains in the preceding section. Physical metrics retain all body predictions exactly; low skating alone is not evidence of better reconstruction."),
        paragraph("Full comparison including UniEgomotion missing-input ablations, all 14 Phase3 settings, per-case ranks, embeddings, source hashes and audit records: " + str(args.output)),
        paragraph("Existing qualitative galleries: " + str(args.guidance / "gallery/index.html") + " and " + str(args.guidance / "video_and_text/gallery/index.html") + ". The existing comparison videos above are unchanged and still show plain Phase3 V2M, not the new caption/guidance arms."),
    ]
    response = api.request("PATCH", "/blocks/" + PAGE_ID + "/children", {"children": blocks})
    save(args.output / "notion_publication.json", dict(url=PAGE_URL, status="published",
        block_ids=[b["id"] for b in response["results"]], summary_sha256=sha(args.output / "summary.json")))


def main(args):
    from decode_uniego_torch import decode_transforms
    from nymeria_eval import NymeriaTMREmbedder

    require_compute()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "worker.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    base = read(args.base / "configuration.json")
    previous = read(args.base / "summary.json")
    if previous["status"] != "complete" or previous["configuration_sha256"] != sha(args.base / "configuration.json"):
        raise RuntimeError("Prior evaluator comparison is incomplete or changed")
    cases = base["cases"]
    ids = [r["id"] for r in cases]
    if len(ids) != 71 or len(set(ids)) != 71:
        raise RuntimeError("Expected fixed71")
    template = read(args.base / "adapter.json")
    if sha(args.base / "adapter.json") != base["adapter_sha256"]:
        raise RuntimeError("Common adapter changed")
    reference = read(Path(base["reference"]) / "configuration.json")
    contract = reference["phase3_contract"]
    for field in ("mean", "std"):
        if sha(contract["motion_stats_" + field]) != contract["motion_" + field]:
            raise RuntimeError("Generator normalization changed")
    mean, std = np.load(contract["motion_stats_mean"]), np.load(contract["motion_stats_std"])
    with np.load(args.base / "GT_embeddings.npz", allow_pickle=False) as a:
        gt = a["completed"].copy()
        if a["case_id"].tolist() != ids or str(a["signature"]) != stable_hash(base):
            raise RuntimeError("GT embedding identity mismatch")
    with np.load(args.base / "text_embeddings.npz", allow_pickle=False) as a:
        text = a["embedding"].copy()
        if a["case_id"].tolist() != ids:
            raise RuntimeError("Caption embedding identity mismatch")
    gt_motion, neutral_shapes = [], []
    for case in cases:
        with np.load(case["bundle"], allow_pickle=False) as a:
            gt_motion.append(a["motion"].copy())
            neutral_shapes.append(a["neutral"].copy())
    with (args.uem / "comparison.csv").open() as stream:
        native = {r["Model"]: r for r in csv.DictReader(stream)}
    results = []
    for row in previous["models"]:
        key = row["key"]
        if key in ("egoallo", "phase3_v2m"):
            continue
        name = "traj_recon / trajectory" if key == "traj_recon" else key.replace("__", " / ")
        source = native[name]
        results.append(dict(key=key, label=row["label"], family="GT" if key == "GT" else "uem",
            input=source["Input"].replace("video_trajectory", "video + measured trajectory").replace("trajectory", "camera trajectory"),
            training_step=row.get("training_step"), setting=source["Setting"],
            FID=row["FID"], retrieval=row["retrieval"], physical={k:float(source[k]) for k in PHYSICAL}))
    parent_gt = next(r for r in results if r["key"] == "GT")
    arms = []
    provenance = {"prior_configuration":sha(args.base / "configuration.json"),
                  "prior_summary":sha(args.base / "summary.json"), "adapter":sha(args.base / "adapter.json"),
                  "GT_embeddings":sha(args.base / "GT_embeddings.npz"),
                  "text_embeddings":sha(args.base / "text_embeddings.npz"),
                  "uem_physical_csv":sha(args.uem / "comparison.csv")}
    for arm, caption, folder in (("video", False, args.guidance), ("caption", True, args.guidance / "video_and_text")):
        cfg = read(folder / "configuration.json")
        native_rows = read(folder / "per_case.json")
        if cfg["checkpoint_sha256"] != reference["phase3_checkpoint_sha256"]:
            raise RuntimeError("Phase3 checkpoint differs from the baseline comparison")
        if cfg["scales"] != SCALES or cfg["frames"] != 97 or cfg["fps"] != 20 or cfg["steps"] != 35 or cfg["shift"] != 3:
            raise RuntimeError("Unexpected guidance sampling recipe")
        if list(cfg["body23"]) != KEEP.tolist():
            raise RuntimeError("Body joint subset changed")
        for a, b in zip(cfg["cases"], cases, strict=True):
            if any(a[k] != b[k] for k in ("id", "uuid", "start", "seed", "bundle", "caption", "floor_offset")):
                raise RuntimeError("Guidance cases differ from baseline cases")
        by_id = {r["id"]: r for r in native_rows}
        if set(by_id) != set(ids):
            raise RuntimeError("Guidance physical metrics use different cases")
        native_rows = [by_id[key] for key in ids]
        gt_physical = physical(native_rows, "GT")
        if any(abs(gt_physical[k] - parent_gt["physical"][k]) > 0.00011 for k in PHYSICAL):
            raise RuntimeError("GT physical convention differs")
        for name in ("configuration.json", "per_case.json", "summary.json"):
            provenance[arm + "/" + name] = sha(folder / name)
        arms.append((arm, caption, folder, cfg, native_rows))
    configuration = dict(schema="v2m_guidance_uem_matched71_v1", cases=cases, parent_comparison=str(args.base),
        guidance_root=str(args.guidance), provenance=provenance, scales=SCALES,
        evaluator_checkpoint=base["evaluator_checkpoint"], evaluator_sha256=base["evaluator_sha256"],
        evaluator_stats=base["evaluator_stats"], text_cache=base["text_cache"],
        adapter_sha256=base["adapter_sha256"], retrieval=base["retrieval"],
        source_fps=20, evaluator_fps=30, common_body23=True,
        checkpoint_sha256=reference["phase3_checkpoint_sha256"], source_sha256=sha(__file__),
        guidance_caveat="whole-bridge-disabled reference, not trained video-dropout CFG; caption retained in both branches",
        selection_caveat="all seven test-sweep settings reported; no validation-selected winner")
    target = args.output / "configuration.json"
    if target.exists() and read(target) != configuration:
        raise RuntimeError("Refusing to mix comparison identities")
    save(target, configuration)
    if sha(base["evaluator_checkpoint"]) != base["evaluator_sha256"] or sha(base["text_cache"]) != base["text_cache_sha256"]:
        raise RuntimeError("Evaluator or caption cache changed")
    progress(args, "loading_evaluator")
    embedder = NymeriaTMREmbedder(base["evaluator_checkpoint"], base["evaluator_stats"], base["text_cache"], "cuda")
    if embedder.fps != 30:
        raise RuntimeError("Evaluator frame rate changed")
    parity = None
    for arm, caption, folder, cfg, native_rows in arms:
        for scale in SCALES:
            setting = f"video_cfg_{scale:g}".replace(".", "p")
            key = arm + "__" + setting
            positions, neutrals, predictions, audits = [], [], [], []
            for i, row in enumerate(cases):
                path = folder / setting / (row["id"] + ".npz")
                with np.load(path, allow_pickle=False) as a:
                    if int(a["seed"]) != row["seed"] or float(a["video_scale"]) != scale:
                        raise RuntimeError("Prediction seed/scale mismatch")
                    if not np.array_equal(a["gt_motion"], gt_motion[i]) or not np.array_equal(a["neutral"], neutral_shapes[i]):
                        raise RuntimeError("Prediction GT/shape differs from the matched reference")
                    if caption and str(a["caption"]) != row["caption"]:
                        raise RuntimeError("Caption mismatch")
                    raw = a["generated"] * std + mean
                transforms = decode_transforms(torch.from_numpy(raw)[None])[0].numpy()
                joints = transforms[..., :3, 3][:, KEEP]
                rotations = transforms[..., :3, :3][:, KEEP]
                full, neutral = complete_body(joints, rotations, neutral_shapes[i][KEEP], template)
                positions.append(full)
                neutrals.append(neutral)
                predictions.append(dict(id=row["id"], path=str(path), sha256=sha(path)))
                audits.append(dict(id=row["id"], retained_body_max_error_m=float(np.max(np.abs(full[:, KEEP]-joints)))))
            batches = []
            for start in range(0, 71, 16):
                j = torch.from_numpy(np.stack(positions[start:start+16]).astype(np.float32))
                n = torch.from_numpy(np.stack(neutrals[start:start+16]).astype(np.float32))
                batches.append(embedder.embed_motion(j, n, lengths=[97]*len(j), source_fps=20).cpu().numpy())
            embedding = np.concatenate(batches)
            if embedding.shape != (71,256) or not np.isfinite(embedding).all():
                raise RuntimeError("Invalid embeddings")
            retrieval, ranks = retrieval_details(embedding, text, ids)
            if arm == "video" and scale == 1:
                with np.load(args.base / "phase3_v2m/embeddings.npz", allow_pickle=False) as a:
                    parity = float(np.max(np.abs(embedding-a["embedding"])))
                if parity > 0.0001:
                    raise RuntimeError("Native scale1 evaluator parity failed")
            label = "Phase3 " + ("video + caption" if caption else "video") + f", scale {scale:g}"
            result = dict(key=key, label=label, family="phase3", caption=caption, video_scale=scale,
                input=("video + caption + neutral shape" if caption else "video + neutral shape") if scale != 0 else
                      ("caption + neutral shape; bridges off" if caption else "neutral shape; bridges off"),
                training_step=200000, setting="inference intervention" if caption or scale != 1 else "native V2M",
                FID=fid(embedding, gt), retrieval=retrieval, physical=physical(native_rows, setting),
                predictions=predictions)
            destination = args.output / key
            atomic_npz(destination / "embeddings.npz", embedding=embedding, case_id=np.asarray(ids))
            save(destination / "retrieval_groups.json", ranks)
            save(destination / "adapter_audit.json", audits)
            save(destination / "summary.json", result)
            results.append(result)
            save(args.output / "partial_summary.json", dict(rows=results))
            progress(args, "scored", setting=key, FID=result["FID"], R3=retrieval["R32_R03_mean"])
    headline_keys = ["GT", "traj_recon", "RGB_recon__video", "recon__video_trajectory", "all__video_trajectory",
                     "video__video_cfg_1", "video__video_cfg_2", "caption__video_cfg_1", "caption__video_cfg_2"]
    summary = dict(status="complete", cases=71, rows=results, headline_keys=headline_keys,
        configuration_sha256=sha(target), baseline_scale1_max_abs_embedding_difference=parity,
        adapter_diagnostic=previous["adapter_diagnostic"],
        caveats=["71 fixed windows, not the full test set", "different observed modalities and shape conditioning",
                 "small-sample FID", "test-set guidance sweep, not validation-selected", "caption remains in null branch"])
    save(args.output / "summary.json", summary)
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["Model", "Input", "Step", "Setting", *PHYSICAL, "FID", "R1", "R1_SD", "R2", "R2_SD", "R3", "R3_SD"])
    lines = ["Model | Input | MPJPE cm | FID | R@1 | R@2 | R@3"]
    for row in results:
        r = row["retrieval"]
        writer.writerow([row["label"], row["input"], row.get("training_step"), row["setting"],
                         *[row["physical"][k] for k in PHYSICAL], row["FID"],
                         *[r[f"R32_R{k:02d}_{stat}"] for k in (1,2,3) for stat in ("mean","std")]])
        lines.append(" | ".join([row["label"], row["input"], f"{row['physical']['MPJPE cm']:.2f}",
                                  f"{row['FID']:.6f}", *[f"{r[f'R32_R{k:02d}_mean']:.2f}" for k in (1,2,3)]]))
    (args.output / "comparison.csv").write_text(stream.getvalue())
    (args.output / "comparison.txt").write_text("\n".join(lines)+"\n")
    print("\n".join(lines), flush=True)
    save(args.output / "METRICS_COMPLETE.json", dict(rows=len(results), cases=71, summary_sha256=sha(args.output / "summary.json")))
    publish(args, summary)
    save(args.output / "COMPLETE.json", dict(rows=len(results), cases=71, notion=PAGE_URL))
    progress(args, "complete", rows=len(results), cases=71, notion=PAGE_URL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "guidance", "uem", "output", "notion-helper"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    try:
        with torch.inference_mode():
            main(args)
    except BaseException as error:
        save(args.output / "FAILED.json", dict(error=str(error), traceback=traceback.format_exc(), time=time.time()))
        raise
