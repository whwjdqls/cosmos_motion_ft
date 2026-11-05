"""Camera-conditioned Phase1 video generation on the complete T97 test cohort."""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np

from nymeria_baselines.fulltest_v2m import read, sha, guard, BASE

REPO = Path(__file__).resolve().parents[1]
GOAL = BASE / "operations/idle_gpu_followups_20261104"
EXPECTED_METADATA = "b2b3374b8aa1f688dc73fabb78832889c9d3e4264299f20aa8aefbdcde4ecca8"


def save(path, value):
    """Atomic even when independent GPU shards write an identical receipt."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def prepare(a):
    parent = read(a.reference / "configuration.json")
    rows = read(a.reference / "cases.json")
    assert len(rows) == 11911 and len({r["uuid"] for r in rows}) == 71
    cfg = dict(schema="phase1_videoio_fulltest_forward_v1", cases=len(rows), recordings=71,
               physical_windows=len({(r["uuid"], r["start"]) for r in rows}),
               cases_sha256=sha(a.reference / "cases.json"), reference=str(a.reference),
               reference_configuration_sha256=sha(a.reference / "configuration.json"),
               checkpoint_metadata_sha256=EXPECTED_METADATA,
               sampling=dict(model_mode="forward_dynamics", num_frames=97, fps=20., resolution="256",
                             condition_frame_indexes_vision=[0], prompt="", num_steps=30, shift=10., guidance=1.),
               metrics=["PSNR", "SSIM", "LPIPS-Alex", "DreamSim ensemble", "FVD-I3D400"],
               scored_frames=list(range(1, 97)), generator="Official cosmos_framework.scripts.inference, EMA weights",
               scope="All accepted test annotation windows; caption duplicates retained as paired-seed cases; no text is supplied",
               source_sha256=sha(__file__), shards=a.shards)
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / "configuration.json").exists():
        assert read(a.out / "configuration.json") == cfg
        return
    (a.out / "cases.json").write_text(json.dumps(rows, indent=2)+"\n")
    assert sha(a.out / "cases.json") == cfg["cases_sha256"]
    save(a.out / "configuration.json", cfg)


def checkpoint():
    path = Path(read(GOAL / "phase1_checkpoint.json")["path"])
    if not (path / "model/.metadata").is_file():
        raise FileNotFoundError("Phase1 video-projection checkpoint is not staged: " + str(path))
    assert sha(path / "model/.metadata") == EXPECTED_METADATA, "Wrong Phase1 checkpoint"
    from native_phase_training.run_contract import load_contract_for_checkpoint
    contract, _, _, _ = load_contract_for_checkpoint(path)
    assert contract.adaptation_mode == "global_lora_video_io" and contract.model_family == "edge"
    assert contract.num_frames == 97 and contract.training_prefix_lengths == (1,)
    assert set(contract.active_modes) == {"forward_dynamics", "inverse_dynamics"}
    return path


@lru_cache(maxsize=4)
def camera(uuid):
    path = Path("/lustre/whwjdqls99/cosmos/weka/nymeriaplus_kimodo_proportional/camera_rgb") / (uuid + ".npz")
    with np.load(path, allow_pickle=False) as z:
        return z["cam_world_pos_upright"].astype(np.float32), z["cam_world_rot_upright"].astype(np.float32)


def record(a, row):
    from nymeria_camera_dataset import decode_window_pyav
    from nymeria_camera_rgb_dataset import rel_action_from_window
    import imageio.v3 as iio
    directory = a.out / "inputs" / row["id"]
    directory.mkdir(parents=True, exist_ok=True)
    image, action = directory / "first_frame.png", directory / "camera_action.json"
    if not image.exists():
        first = decode_window_pyav(row["rgb"], row["start"], 1, 20.)
        assert first.shape[0] == 1
        iio.imwrite(image, first[0])
    if not action.exists():
        position, rotation = camera(row["uuid"])
        start = row["start"]
        values = rel_action_from_window(position[start:start+97], rotation[start:start+97])
        assert values.shape == (96, 9) and np.isfinite(values).all()
        save(action, values.tolist())
    sampling = read(a.out / "configuration.json")["sampling"]
    return dict(**sampling, name=row["id"]+"_forward_dynamics", vision_path=str(image),
                action_path=str(action), action_chunk_size=96, image_size=256,
                domain_name="camera_pose", view_point="ego_view", aspect_ratio="1,1", seed=row["seed"])


def generated_dir(a, row):
    return a.out / "inference" / (row["id"]+"_forward_dynamics")


def validate_output(a, row):
    directory = generated_dir(a, row)
    if not (directory / "sample_outputs.json").exists():
        return False
    meta = read(directory / "sample_outputs.json")
    if meta.get("status") != "success" or not (directory / "vision.mp4").is_file():
        return False
    expected = record(a, row)
    for key in ("model_mode", "num_frames", "fps", "resolution", "prompt", "num_steps", "shift", "guidance", "seed", "vision_path", "action_path"):
        assert meta["args"].get(key) == expected[key], (row["id"], key)
    return True


def generate(a):
    path = checkpoint()
    cfg = read(a.out / "configuration.json")
    assert cfg["source_sha256"] == sha(__file__)
    rows = read(a.out / "cases.json")[a.shard::cfg["shards"]]
    if a.limit:
        rows = rows[:a.limit]
    records = [record(a, row) for row in rows if not validate_output(a, row)]
    if not records:
        return
    source = a.out / "shards" / f"{a.shard:03d}.jsonl"
    source.parent.mkdir(exist_ok=True)
    source.write_text("".join(json.dumps(r)+"\n" for r in records))
    runtime = a.out / "runtime" / f"{a.shard:03d}"
    subprocess.run([sys.executable, str(REPO/"native_phase_training/run_contract.py"),
                    "--checkpoint-path", str(path), "--output-dir", str(runtime)], check=True)
    # The worker launcher sources this same resolved contract before entry.
    framework = Path(os.environ["COSMOS_FRAMEWORK_EDGE_ROOT"])
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
               "-m", "cosmos_framework.scripts.inference", "--checkpoint-path", str(path),
               "--config-file", "native_phase_training/inference_config.py", "--experiment", "world_camera_nymeria_latent_edge",
               "--sampler", "unipc", "--use-ema-weights", "--parallelism-preset", "latency",
               "--dp-shard-size", "1", "--dp-replicate-size", "1", "--cp-size", "1", "--cfgp-size", "1",
               "--no-use-torch-compile", "--no-use-cuda-graphs", "--no-diffusion-cache", "--no-guardrails",
               "-o", str(a.out/"inference"), "-i", str(source)]
    subprocess.run(command, cwd=framework, check=True)
    assert all(validate_output(a, row) for row in rows)
    if a.limit:
        save(a.out/"generation_preflight.json", dict(cases=len(rows), validated=True,
             checkpoint_metadata_sha256=EXPECTED_METADATA, source_sha256=sha(__file__)))


class Metrics:
    def __init__(self):
        import torch
        from native_phase_training.evaluate_inverse_forward import LPIPSAlex
        from dreamsim import dreamsim
        assets = read(GOAL / "metric_assets.json")
        sys.path.insert(0, assets["cdfvd_root"])
        from cdfvd.third_party.i3d.utils import load_i3d_model, preprocess_i3d
        self.preprocess_i3d = preprocess_i3d
        assert sha(assets["i3d_checkpoint"]) == assets["i3d_sha256"]
        self.i3d = load_i3d_model(torch.device("cuda"), assets["i3d_checkpoint"]).eval()
        self.lpips = LPIPSAlex("cuda", 8)
        self.dream, self.preprocess = dreamsim(pretrained=True, device="cuda",
            cache_dir=assets["dreamsim_cache"], dreamsim_type="ensemble", normalize_embeds=True)
        self.identity = assets

    def acceptance(self, gt):
        values, features = self.evaluate(gt, gt.copy())
        assert values["psnr_db"] == 100. and abs(values["ssim"]-1.) < 1e-7
        assert abs(values["lpips_alex"]) < 1e-6 and abs(values["dreamsim"]) < 1e-5
        assert features[0].shape == features[1].shape == (1,400)
        np.testing.assert_allclose(features[0], features[1], rtol=0., atol=1e-6)
        return dict(identity_metrics=values, I3D_feature_shape=list(features[0].shape),
                    finite=True, identity_features_equal=True, assets=self.identity)

    def evaluate(self, gt, pred):
        import torch
        from native_phase_training.evaluate_inverse_forward import _frame_metrics
        from native_phase_training.evaluate_forward_dreamsim import _distances
        with torch.inference_mode():
            per_frame = _frame_metrics(gt[1:], pred[1:], self.lpips)
            per_frame["dreamsim"] = _distances(self.dream, self.preprocess, gt[1:], pred[1:], torch.device("cuda"), 8)
            features = [self.i3d(self.preprocess_i3d(x[None, 1:]).cuda()).float().cpu().numpy() for x in (gt, pred)]
        assert all(np.isfinite(v).all() for v in (*per_frame.values(), *features))
        return {k:float(v.mean()) for k,v in per_frame.items()}, features


def score(a):
    from native_phase_training.evaluate_inverse_forward import _read_video_rgb, _resize_gt_like_native
    from nymeria_camera_dataset import decode_window_pyav
    from nymeria_baselines.fulltest_v2m import npz
    cfg = read(a.out / "configuration.json")
    rows = read(a.out / "cases.json")[a.shard::cfg["shards"]]
    if a.limit:
        rows = rows[:a.limit]
    directory = a.out / "metrics"
    directory.mkdir(exist_ok=True)
    scorer = None
    for row in rows:
        destination = directory / (row["id"]+".json")
        if destination.exists():
            continue
        assert validate_output(a, row)
        if scorer is None:
            scorer = Metrics()
            save(a.out / "metric_identity.json", scorer.identity)
        gt = _resize_gt_like_native(decode_window_pyav(row["rgb"], row["start"], 97, 20.),256,256)
        generated = _read_video_rgb(generated_dir(a,row)/"vision.mp4")
        assert generated.shape == gt.shape == (97,256,256,3)
        acceptance = a.out / "metric_acceptance.json"
        if not acceptance.exists():
            save(acceptance, scorer.acceptance(gt))
        values, features = scorer.evaluate(gt,generated)
        npz(directory/(row["id"]+"_features.npz"),GT=features[0],model=features[1])
        save(destination,dict(id=row["id"],uuid=row["uuid"],model=values,
             generated_video_sha256=sha(generated_dir(a,row)/"vision.mp4"),
             checkpoint_metadata_sha256=EXPECTED_METADATA))
    if not a.limit:
        save(a.out/"shards"/f"{a.shard:03d}_COMPLETE.json",dict(cases=len(rows),shard=a.shard))


def metric_preflight(a):
    from native_phase_training.evaluate_inverse_forward import _resize_gt_like_native
    from nymeria_camera_dataset import decode_window_pyav
    row=read(a.out/"cases.json")[0]
    gt=_resize_gt_like_native(decode_window_pyav(row["rgb"],row["start"],97,20.),256,256)
    scorer=Metrics()
    save(a.out/"metric_acceptance.json",scorer.acceptance(gt))
    save(a.out/"metric_identity.json",scorer.identity)


def summarize(a):
    cfg = read(a.out / "configuration.json")
    rows = read(a.out / "cases.json")
    results, real, fake = [], [], []
    for row in rows:
        path = a.out/"metrics"/(row["id"]+".json")
        if not path.exists():
            continue
        results.append(read(path))
        with np.load(path.with_name(row["id"]+"_features.npz")) as z:
            real.append(z["GT"]);fake.append(z["model"])
    if not results:
        return
    values = {k:float(np.mean([r["model"][k] for r in results])) for k in results[0]["model"]}
    by_recording = defaultdict(list)
    for row in results:
        by_recording[row["uuid"]].append(row["model"])
    recording_values = {key:{k:float(np.mean([r[k] for r in rows])) for k in values}
                        for key, rows in by_recording.items()}
    fvd = None
    if len(results)>=2:
        from scipy.linalg import sqrtm
        real,fake=np.concatenate(real).astype(np.float64),np.concatenate(fake).astype(np.float64)
        cr,cf=np.cov(real,rowvar=False),np.cov(fake,rowvar=False)
        value=np.square(real.mean(0)-fake.mean(0)).sum()+np.trace(cr+cf-2*sqrtm(cr@cf).real)
        assert np.isfinite(value) and value>=-1e-4
        fvd=max(0.,float(value))
    complete=len(results)==len(rows)
    save(a.out/"summary.json",dict(cases=len(results),expected=len(rows),complete=complete,model=values,
         per_recording=recording_values,
         recording_balanced={k:float(np.mean([r[k] for r in recording_values.values()])) for k in values},
         FVD_I3D400=fvd,GT_identity_reference=dict(psnr_db=100.,ssim=1.,lpips_alex=0.,dreamsim=0.,FVD=0.),
         reference_note="PSNR identity capped100dB; generated frames1..96 only. Native MP4 outputs are decoded for all pixel/video metrics.",
         cohort_configuration_sha256=sha(a.out/"configuration.json")))


def publish(a):
    """Append/update only this experiment's own Notion table, with readback."""
    import fcntl
    import runpy
    if not (a.out / "summary.json").exists():
        return
    with (a.out / "publication.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        h=runpy.run_path(str(BASE/"operations/phase1_notext_fourway_55k_20261019/upload_notion.py"))
        api=h["Notion"](h["read_token"]()); rt=h["rich_text"]; plain=h["plain_text"]
        summary=read(a.out/"summary.json"); v=summary["model"]
        rows=[["Phase1 forward / no text", "Cases", "PSNR dB ↑", "SSIM ↑", "LPIPS ↓", "DreamSim ↓", "FVD I3D400 ↓"],
              ["GT identity",str(summary["cases"]),"100 (cap)","1","0","0","0"],
              ["Two-task LoRA + video I/O EMA200k",f"{summary['cases']}/{summary['expected']}",
               f"{v['psnr_db']:.4f}",f"{v['ssim']:.6f}",f"{v['lpips_alex']:.6f}",
               f"{v['dreamsim']:.6f}","—" if summary["FVD_I3D400"] is None else f"{summary['FVD_I3D400']:.6f}"]]
        statefile=a.out/"notion_publication.json"
        if statefile.exists():
            state=read(statefile)
            for uid,row in zip(state["rows"],rows,strict=True):
                api.request("PATCH","/blocks/"+uid,dict(table_row=dict(cells=[rt(x) for x in row])))
        else:
            title="Phase1 two-task video projections — all T97 windows, camera + first image, no text"
            note=("Official Edge inference, fixed EMA200k LoRA+video-I/O checkpoint; first image + camera only, "
                  "empty text, UniPC30/shift10/guidance1. Metrics use generated frames1–96 and decoded native MP4. "
                  "11911 paired-seed annotation cases,11677 physical windows,71 recordings. FVD uses pinned TATS "
                  "I3D400 logits, not CD-FVD; partial-cohort results remain provisional. Per-case and recording-balanced "
                  "metrics and measured GT identity acceptance are saved. Artifacts: "+str(a.out))
            blocks=[dict(object="block",type="heading_2",heading_2=dict(rich_text=rt(title))),
                    dict(object="block",type="paragraph",paragraph=dict(rich_text=rt(note))),
                    dict(object="block",type="table",table=dict(table_width=7,has_column_header=True,has_row_header=False,
                         children=[dict(object="block",type="table_row",table_row=dict(cells=[rt(x) for x in row])) for row in rows]))]
            response=api.request("PATCH","/blocks/40dd2a44-f44b-8080-aad3-cda6b5fea7c8/children",dict(children=blocks))
            table=response["results"][-1]["id"]
            actual=api.request("GET","/blocks/"+table+"/children?page_size=100")["results"]
            state=dict(table=table,rows=[b["id"] for b in actual],heading=response["results"][0]["id"])
            save(statefile,state)
        actual=api.request("GET","/blocks/"+state["table"]+"/children?page_size=100")["results"]
        assert [[plain(c) for c in b["table_row"]["cells"]] for b in actual]==rows
        state.update(verified=True,complete=summary["complete"],summary_sha256=sha(a.out/"summary.json"))
        save(statefile,state)
        if summary["complete"]:
            save(a.out/"COMPLETE.json",dict(cases=summary["cases"],notion_verified=True,
                 summary_sha256=sha(a.out/"summary.json"),metric_acceptance_sha256=sha(a.out/"metric_acceptance.json")))
        print(json.dumps(state),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("prepare","generate","score","summarize","publish","metric-preflight"))
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--reference",type=Path,default=BASE/"operations/fulltest_v2m_ablations_20261104")
    p.add_argument("--shard",type=int,default=0);p.add_argument("--shards",type=int,default=32)
    p.add_argument("--limit",type=int,default=0)
    a=p.parse_args()
    if a.mode != "publish":
        guard()
    if a.mode == "publish":
        publish(a)
    else:
        {"prepare":prepare,"generate":generate,"score":score,"summarize":summarize,
         "metric-preflight":metric_preflight}[a.mode](a)


if __name__=="__main__":main()
