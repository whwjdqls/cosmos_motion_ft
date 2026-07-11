"""Render saved viz_step* joint .npy (T,30,3) -> skeleton mp4 via kimodo's SOMA renderer."""
import sys, os, json, numpy as np
sys.path.insert(0, "/home/jungbin_cho/kimodo_open")
sys.path.insert(0, "/home/jungbin_cho/cosmos_motion_ft/motion_expert")
import matplotlib; matplotlib.use("Agg")
from bs_viz import render_pair, load_skeleton
jp, skip = load_skeleton()
vdir = sys.argv[1]
for it in json.load(open(os.path.join(vdir, "manifest.json"))):
    joints = np.load(it["joints_npy"])
    out = it["joints_npy"].replace(".npy", ".mp4")
    gtp = it.get("gt_joints_npy")
    gt = np.load(gtp) if (gtp and os.path.isfile(gtp)) else None  # GT|gen when do_viz saved GT
    try:
        render_pair(gt, joints, jp, out, caption=it["caption"][:70], skip_joints=skip, camera="follow", fps=20)
        print("rendered", os.path.basename(out), joints.shape)
    except Exception as e:
        print("FAIL", os.path.basename(out), type(e).__name__, str(e)[:160])
