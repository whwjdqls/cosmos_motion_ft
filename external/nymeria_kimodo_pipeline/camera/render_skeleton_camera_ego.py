"""Render skeleton + egocentric camera trajectory (3D) side-by-side with the
egocentric video, for sample atomic-action slices.

Left panel : kimodo skeleton (GT-floor grounded) + the head camera trajectory
             (cyan path + current camera dot + a small device-axis triad), drawn in
             the SAME per-slice frame as the skeleton (Z-up->Y-up rotation, start-root
             xz canonicalization, and GT/estimated `ground_offset_y` vertical grounding).
Right panel: the cached egocentric RGB frame (frame_{idx:06d}.webp), same body frame.

Inputs:
  - skeleton : /weka/jungbin/nymeriaplus_kimodo_proportional/{subj}/{seq}.npz
  - camera   : /weka/jungbin/nymeriaplus_kimodo_proportional/camera/{subj}/{seq}.npz
               (from extract_camera_trajectory.py; cam_world_pos/rot in Aria Z-up world)
  - ego video: /weka/jungbin/nymeriaplus_kimodo_proportional/video/{subj}/{seq}.mp4
               (decoded per-window via ffmpeg; supersedes the old per-frame webp images)
  - floor    : metadata_atomic_action_floor.jsonl (ground_offset_y + text per slice)

Output: /weka/jungbin/nymeriaplus_kimodo_proportional/visualization/{seq}__seg{sf}_{label}.mp4
Env: `kimodo`.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/home/jungbin_cho/kimodo_open")
sys.path.insert(0, "/home/jungbin_cho/nymeria_kimodo_pipeline")
import os, argparse, subprocess
import numpy as np
import torch
import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont
from kimodo.skeleton import SOMASkeleton77
from kimodo.scripts.render_soma import _to_display, _draw_skel, _fixed_viewport
from kimodo.scripts.render_hml3d import (
    _world_extent, _setup_axes, _draw_floor_and_grid, _draw_trajectory,
    DEFAULT_GRID_SPACING,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from viz_with_text import floor_row, SKIP

s77 = SOMASkeleton77()
ROOT = int(s77.root_idx)
PAR = s77.joint_parents.tolist()

MROOT = "/weka/jungbin/nymeriaplus_kimodo_proportional"
CAMROOT = f"{MROOT}/camera"
VIDROOT = f"{MROOT}/video"   # per-seq ego mp4 (Stage A of video/); supersedes old webp images
OUTDIR = f"{MROOT}/visualization"
FPS = 20.0

# kimodo (x,y,z) = (world_x, world_z, -world_y):  R_z2y @ v
def world_to_kimodo(v):                     # v: (...,3) Aria Z-up world
    return np.stack([v[..., 0], v[..., 2], -v[..., 1]], axis=-1)

CAM_C = "#00d0d0"                            # camera cyan
TRIAD = ["#ff3030", "#30ff30", "#3060ff"]   # device x,y,z axes


def fk_seq(d):
    lrm = torch.from_numpy(d["local_rot_mats"].astype(np.float32))
    root = torch.from_numpy(d["root_positions"].astype(np.float32))
    nj = torch.from_numpy(d["neutral_joints"].astype(np.float32))
    B = lrm.shape[0]
    _, posed77, _ = s77.fk(lrm, root, neutral_joints=nj.unsqueeze(0).expand(B, -1, -1))
    return posed77.numpy()                  # (B,77,3) kimodo coords


def load_ego_window(subj, seq, sf, ef, size=700):
    """Decode ego frames [sf, ef) from the per-seq mp4 (video/ Stage A), once.
    Returns (n, size, size, 3) uint8; black frames if the mp4 is absent."""
    n = ef - sf
    mp4 = f"{VIDROOT}/{subj}/{seq}.mp4"
    if not os.path.exists(mp4):
        return np.zeros((n, size, size, 3), np.uint8)
    # select the frame range, then scale to the panel size
    vf = f"select='between(n,{sf},{ef - 1})',scale={size}:{size}"
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", mp4, "-vf", vf, "-vsync", "0",
         "-f", "image2pipe", "-pix_fmt", "rgb24", "-vcodec", "rawvideo", "-"],
        capture_output=True)
    buf = np.frombuffer(out.stdout, np.uint8)
    got = buf.size // (size * size * 3)
    frames = np.zeros((n, size, size, 3), np.uint8)
    if got:
        frames[:got] = buf[: got * size * size * 3].reshape(got, size, size, 3)
    return frames


def caption_bar(W, text, info, h=64):
    img = Image.new("RGB", (W, h), (15, 15, 18)); draw = ImageDraw.Draw(img)
    try:
        f1 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
        f2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        f1 = f2 = ImageFont.load_default()
    draw.text((10, 6), text[:120], fill=(255, 255, 255), font=f1)
    draw.text((10, 36), info, fill=(120, 220, 220), font=f2)
    return np.asarray(img)


def render_slice(subj, seq, sf, ef, label, triad_len=0.25):
    d = np.load(f"{MROOT}/{subj}/{seq}.npz", allow_pickle=True)
    p = fk_seq(d)                                              # (B,77,3) kimodo
    B = p.shape[0]; ef = min(ef, B)
    cam = np.load(f"{CAMROOT}/{subj}/{seq}.npz", allow_pickle=True)
    cam_k = world_to_kimodo(cam["cam_world_pos"].astype(np.float64))     # (B,3) kimodo
    cam_axes_k = world_to_kimodo(np.transpose(cam["cam_world_rot"].astype(np.float64), (0, 2, 1)))  # (B,3axes,3) per-axis dir in kimodo

    fr = floor_row(seq, sf)
    text = fr["text"] if fr else label
    off = float(fr["ground_offset_y"]) if (fr and fr.get("ground_offset_y") is not None) else \
        float(p[sf:ef, :, 1].min())
    src = fr.get("floor_source", "?") if fr else "?"
    amb = bool(fr.get("ambiguous")) if fr else False

    # canonicalize skeleton + camera by the SAME offsets (start-root xz, floor y)
    ox, oz = p[sf, ROOT, 0], p[sf, ROOT, 2]
    seg = p[sf:ef].copy()
    seg[:, :, 0] -= ox; seg[:, :, 2] -= oz; seg[:, :, 1] -= off
    camk = cam_k[sf:ef].copy()
    camk[:, 0] -= ox; camk[:, 2] -= oz; camk[:, 1] -= off
    axk = cam_axes_k[sf:ef]                                    # directions: no translation

    seg_d = _to_display(seg)                                   # negate X
    cam_d = _to_display(camk)
    ax_d = axk.copy(); ax_d[..., 0] *= -1                       # display negates X for dirs too
    trail = seg_d[:, ROOT]

    # extent sized to include the whole camera path so the fixed viewport fits both
    extent = _world_extent([seg_d], extra_pts=[trail, cam_d])
    vp = _fixed_viewport(extent)

    ego_win = load_ego_window(subj, seq, sf, ef, size=700)  # decoded once from the mp4
    fig = plt.figure(figsize=(700 / 120, 700 / 120), dpi=120)
    frames = []
    try:
        for t in range(ef - sf):
            fig.clf()
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            ax.computed_zorder = False
            _setup_axes(ax, vp, f"t={sf + t}")
            _draw_floor_and_grid(ax, extent, spacing=DEFAULT_GRID_SPACING, floor_alpha=0.9)
            # camera path (whole slice) + current dot + device-axis triad
            ax.plot(cam_d[:, 0], cam_d[:, 2], cam_d[:, 1], color=CAM_C, lw=1.4, zorder=3, alpha=0.9)
            c = cam_d[t]
            ax.scatter([c[0]], [c[2]], [c[1]], color=CAM_C, s=55, zorder=7,
                       edgecolors="k", linewidths=0.6, depthshade=False)
            for a in range(3):
                dvec = ax_d[t, a] * triad_len
                ax.plot([c[0], c[0] + dvec[0]], [c[2], c[2] + dvec[2]], [c[1], c[1] + dvec[1]],
                        color=TRIAD[a], lw=2.0, zorder=8)
            _draw_trajectory(ax, trail, t, "tab:red")
            _draw_skel(ax, seg_d[t], PAR, "tab:red", skip_joints=SKIP)
            fig.subplots_adjust(left=0.04, right=0.98, top=0.96, bottom=0.04)
            fig.canvas.draw()
            left = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            H = left.shape[0]
            right = ego_win[t]
            if right.shape[0] != H:
                right = np.asarray(Image.fromarray(right).resize((H, H), Image.BILINEAR))
            sep = np.full((H, 4, 3), 60, np.uint8)
            body = np.concatenate([left, sep, right], axis=1)
            info = (f"frame {sf + t}  |  floor_source={src}  ground_offset_y={off:.3f}"
                    f"{'  AMBIGUOUS' if amb else ''}  |  cyan=head camera path/dot, triad=device axes"
                    f"  |  LEFT skeleton+camera  RIGHT egocentric")
            bar = caption_bar(body.shape[1], text, info)
            frames.append(np.concatenate([bar, body], axis=0))
    finally:
        plt.close(fig)

    os.makedirs(OUTDIR, exist_ok=True)
    outp = f"{OUTDIR}/{seq}__seg{sf}_{label}.mp4"
    iio.imwrite(outp, np.stack(frames), fps=int(FPS), codec="libx264")
    print(f"wrote {outp}  ({len(frames)} frames, '{text[:45]}')")
    return outp


DEMO = [
    ("S02", "20231006_s1_kirk_flowers_act0_hfjvo9", 500, 600, "walk_bedroom"),
    ("S02", "20231006_s1_kirk_flowers_act0_hfjvo9", 900, 1000, "kneel_under_table"),
    ("S02", "20231006_s1_kirk_flowers_act0_hfjvo9", 2040, 2140, "up_stairs_ambiguous"),
    ("S02", "20231006_s1_kirk_flowers_act0_hfjvo9", 2640, 2740, "down_stairs_landing_ambiguous"),
    ("S02", "20231006_s1_kirk_flowers_act0_hfjvo9", 4679, 4779, "walk_pick_hourglass"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--slice", nargs=5, metavar=("SUBJ", "SEQ", "SF", "EF", "LABEL"),
                    help="render a single slice")
    args = ap.parse_args()
    if args.slice:
        s, q, sf, ef, lb = args.slice
        render_slice(s, q, int(sf), int(ef), lb)
    else:
        for subj, seq, sf, ef, lb in DEMO:
            render_slice(subj, seq, sf, ef, lb)


if __name__ == "__main__":
    main()
