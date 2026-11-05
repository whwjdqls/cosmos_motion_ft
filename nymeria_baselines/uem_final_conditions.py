"""Final supplied-shape UEM evaluation with trajectory, video, or both.

Uses the existing native sampler/noise-preserving batch adapter. The earlier
frozen V2M experiment and its intermediate shape checkpoint remain unchanged.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np

from nymeria_baselines import fulltest_v2m as frozen
from nymeria_baselines.fulltest_v2m_batched import Runner

MODES = {"uem_final_trajectory": "trajectory", "uem_final_video": "video",
         "uem_final_both": "video_trajectory"}


def prepare(a):
    import torch
    parent = frozen.read(a.reference / "configuration.json")
    completion = frozen.read(a.training / "COMPLETE.json")
    assert completion["global_step"] == 555800 and completion["epochs"] >= 350
    experiment = frozen.read(a.training / "experiment.json")
    assert experiment["conditioning"] == {"rgb_only": 1/3, "traj_only": 1/3, "rgb_traj": 1/3}
    assert experiment["shape_prediction"] is False and experiment["state_dim"] == 220
    assert frozen.sha(experiment["cache"] + "/train_stats.npz") == experiment["statistics_sha256"]
    assert frozen.sha(Path(experiment["upstream"]) / "source_manifest.json") == experiment["source_manifest_sha256"]
    if (a.out / "configuration.json").exists():
        cfg = frozen.read(a.out / "configuration.json")
        assert cfg["wrapper_sha256"] == frozen.sha(__file__)
        assert cfg["reference_configuration_sha256"] == frozen.sha(a.reference / "configuration.json")
        return
    (a.out / "snapshots").mkdir(parents=True, exist_ok=True)
    snapshot = a.out / "snapshots/final350.ckpt"
    checkpoint = frozen.snapshot(a.training / "last.ckpt", snapshot)
    assert checkpoint["global_step"] == 555800 and checkpoint["epoch"] in (349, 350)
    hp = checkpoint["hyper_parameters"]
    assert hp["DATA"]["REPRE_TYPE"] == "soma_body220_shape"
    assert hp["DATA"]["COND_IMG_FEAT"] and hp["DATA"]["COND_TRAJ"] and not hp["DATA"]["COND_BETAS"]
    cfg = copy.deepcopy(parent)
    cfg["schema"] = "final350_shape_uem_three_conditions_v1"
    cfg["reference_configuration_sha256"] = frozen.sha(a.reference / "configuration.json")
    cfg["wrapper_sha256"] = frozen.sha(__file__)
    cfg["batch_adapter_sha256"] = frozen.sha(Path(__file__).with_name("fulltest_v2m_batched.py"))
    cfg["training_contract_sha256"] = frozen.sha(a.training / "experiment.json")
    cfg["training_completion"] = completion
    cfg["models"] = [dict(label=key, family="uem", variant="shape", condition=mode,
        step=555800, epoch=int(checkpoint["epoch"]), checkpoint=str(snapshot),
        checkpoint_sha256=frozen.sha(snapshot), shape_supplied=True,
        generator_inverse="normalized * (train_std + 1e-6) + train_mean") for key, mode in MODES.items()]
    cfg["protocol"] = "All accepted T97/20fps test cases; same seeds in all three conditions; clean SOMA30 shape always supplied; native1000 ancestral steps; no captions or target motion conditioning."
    shutil.copyfile(a.reference / "cases.json", a.out / "cases.json")
    assert frozen.sha(a.out / "cases.json") == parent["cases_sha256"]
    for filename in ("llm2vec_fulltest_extended.pt", "llm2vec_fulltest_extended.provenance.json"):
        link = a.out / filename
        target = a.reference / filename
        if link.is_symlink() or link.exists():
            assert link.resolve() == target.resolve(), (link, target)
        else:
            link.symlink_to(target)
    frozen.save(a.out / "configuration.json", cfg)
    frozen.progress(a.out, "prepared", cases=cfg["cases"], models=3, checkpoint_step=555800)


class ConditionalRunner(Runner):
    def __init__(self, a):
        super().__init__(a)
        self.condition = MODES[a.model]
        assert self.cfg["wrapper_sha256"] == frozen.sha(__file__)
        assert self.cfg["batch_adapter_sha256"] == frozen.sha(Path(__file__).with_name("fulltest_v2m_batched.py"))
        # This is the exact inverse of the native training normalization.
        self.std = self.std + np.float32(1e-6)
        with np.load(frozen.CACHE / "train_stats.npz") as z:
            self.camera_mean, self.camera_std = z["camera_mean"].copy(), z["camera_std"].copy()

    def inputs(self, rows):
        import torch
        if self.condition != "trajectory":
            y = super().inputs(rows)
        else:
            neutral = []
            for row in rows:
                assert frozen.sha(row["bundle"]) == row["bundle_sha256"]
                with np.load(row["bundle"]) as z:
                    neutral.append(z["neutral"].copy())
            y = dict(valid_frames=torch.ones(len(rows), 97, dtype=torch.long, device="cuda"),
                     neutral_joints=torch.from_numpy(np.stack(neutral)).cuda().float())
        if self.condition != "video":
            from nymeria_baselines.prepare import camera_trajectory18, IDENTITY_DELTA
            trajectories = []
            for row in rows:
                path = frozen.CACHE / "records" / (row["array"] + ".pose_yup.npy")
                pose = np.array(np.load(path, mmap_mode="r")[row["start"]:row["start"]+97], dtype=np.float32)
                if row["floor_offset"] is not None:
                    pose[:, 1, 3] -= float(row["floor_offset"])
                trajectory = camera_trajectory18(pose)
                trajectory[0, 9:18] = IDENTITY_DELTA
                trajectories.append((trajectory-self.camera_mean)/(self.camera_std+1e-6))
            y["traj"] = torch.from_numpy(np.stack(trajectories).astype(np.float32)).cuda()
            y["traj_mask"] = torch.zeros(len(rows), 97, dtype=torch.long, device="cuda")
        assert ("img_embs" in y) == (self.condition != "trajectory")
        assert ("traj" in y) == (self.condition != "video")
        assert y["neutral_joints"].shape == (len(rows), 30, 3)
        assert all(torch.isfinite(v).all() for v in y.values())
        return y


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "benchmark", "sample", "score", "publish"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference", type=Path, default=frozen.BASE/"operations/fulltest_v2m_ablations_20261104")
    p.add_argument("--training", type=Path, default=frozen.BASE/"operations/uem_shape_threeway_ddp_20261104/training")
    p.add_argument("--model", choices=(*MODES, "GT"))
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 16, 32, 64])
    p.add_argument("--parity-cases", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    if a.command == "publish":
        publish(a)
        return
    frozen.guard()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.command == "prepare":
        prepare(a)
    elif a.command == "score":
        from nymeria_baselines.fulltest_v2m_score import main as score
        score(a)
    else:
        if a.model not in MODES:
            p.error("sample/benchmark requires a condition model")
        runner = ConditionalRunner(a)
        runner.benchmark() if a.command == "benchmark" else runner.run()


def publish(a):
    """Own section only; preserve all existing comparison-page content."""
    import fcntl
    import runpy
    import urllib.parse
    if not (a.out / "configuration.json").exists():
        return
    with (a.out / "publication.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        h = runpy.run_path(str(frozen.BASE/"operations/phase1_notext_fourway_55k_20261019/upload_notion.py"))
        api = h["Notion"](h["read_token"]())
        rt, plain = h["rich_text"], h["plain_text"]
        page = "40dd2a44-f44b-8080-aad3-cda6b5fea7c8"
        rows = [["Final350 UEM / inputs", "Cases", "MPJPE cm", "FID", "R@3 %", "Skate cm/s", "Foot depth cm", "Status"]]
        complete = True
        for key in ("GT", *MODES):
            path = a.out / "scores" / key / "summary.json"
            done = path.parent / "COMPLETE.json"
            if done.exists():
                d = frozen.read(path)
                assert frozen.read(done)["summary_sha256"] == frozen.sha(path)
                v, r = d["physical"], d["retrieval"]
                row = [key, str(d["cases"]), f"{v['mpjpe_m']*100:.3f}", f"{d['FID']:.6f}",
                       f"{r['R32_R03_mean']:.2f} ± {r['R32_R03_std']:.2f}",
                       f"{v['foot_skate_cm_s']:.3f}", f"{v['foot_depth_mean_cm']:.3f}", "Complete"]
            else:
                complete = False
                progress = a.out / "models" / key / "progress.json"
                if progress.exists():
                    x = frozen.read(progress)
                    status = f"{x['stage']}: {x.get('completed',0)}/{x.get('total',11911)}"
                else:
                    status = "Waiting for an idle GPU"
                row = [key, "0", *["—"]*5, status]
            rows.append(row)
        statefile = a.out / "notion_publication.json"
        if statefile.exists():
            state = frozen.read(statefile)
            for uid, row in zip(state["rows"], rows, strict=True):
                api.request("PATCH", "/blocks/"+uid, dict(table_row=dict(cells=[rt(x) for x in row])))
        else:
            title = "Final350 supplied-shape UniEgoMotion — trajectory / video / both"
            note = ("Same final555800-update,350-epoch checkpoint, all11911 T97 test cases and paired seeds. "
                    "Trajectory-only, video-only and both all receive the same clean centered SOMA30 shape. "
                    "Native1000-step sampling; batching accepted only after nativeB1 parity and motion-difference checks. "
                    "Measured GT and the Nymeria evaluator's separate normalization are retained. Artifacts: "+str(a.out))
            blocks = [dict(object="block", type="heading_2", heading_2=dict(rich_text=rt(title))),
                      dict(object="block", type="paragraph", paragraph=dict(rich_text=rt(note))),
                      dict(object="block", type="table", table=dict(table_width=len(rows[0]), has_column_header=True,
                           has_row_header=False, children=[dict(object="block",type="table_row",
                           table_row=dict(cells=[rt(x) for x in row])) for row in rows]))]
            answer = api.request("PATCH", "/blocks/"+page+"/children", dict(children=blocks))
            table = answer["results"][-1]["id"]
            actual = api.request("GET", "/blocks/"+table+"/children?page_size=100")["results"]
            state = dict(table=table, rows=[b["id"] for b in actual], heading=answer["results"][0]["id"])
            frozen.save(statefile, state)
        actual = api.request("GET", "/blocks/"+state["table"]+"/children?page_size=100")["results"]
        assert [[plain(c) for c in b["table_row"]["cells"]] for b in actual] == rows
        state.update(verified=True, complete=complete)
        frozen.save(statefile, state)
        if complete:
            frozen.save(a.out/"COMPLETE.json",dict(cases=11911,models=3,GT=True,notion_verified=True))
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
