"""Render Nymeria proportional motions with:
  - the active atomic_action text burned on top (updates per segment),
  - the foot-skating value (sequence + window mean, cm/s) burned on top,
  - root canonicalized so the trajectory starts at (x,z)=(0,0) in the window,
  - per-window grounding (lowest foot in the window -> y=0).
Real-time 20 fps.
"""
import argparse, json, glob, textwrap, os
import numpy as np, torch, imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont
from kimodo.skeleton import SOMASkeleton30, SOMASkeleton77
from kimodo.scripts.render_soma import render_single
from kimodo.motion_rep.feet import foot_detect_from_pos_and_vel
from kimodo.motion_rep.feature_utils import compute_vel_xyz

META = "/weka/jungbin/nymeriaplus_kimodo_proportional/metadata/metadata_atomic_action.jsonl"
s77 = SOMASkeleton77(); s30 = SOMASkeleton30()
IDX30 = [s77.bone_order_names.index(n) for n in s30.bone_order_names]
F30 = s30.foot_joint_idx
ROOT = int(s77.root_idx)
par = s77.joint_parents.tolist()
SKIP = [s77.bone_index[n] for n in ("LeftHandThumbEnd", "LeftHandMiddleEnd",
        "RightHandThumbEnd", "RightHandMiddleEnd") if n in s77.bone_index]
FPS = 20.0


FLOOR_META = "/weka/jungbin/nymeriaplus_kimodo_proportional/metadata/metadata_atomic_action_floor.jsonl"
_floor_idx = None


def floor_row(stem, start_frame):
    """Per-slice GT-floor row keyed by (filename, start_frame); None if absent."""
    global _floor_idx
    if _floor_idx is None:
        _floor_idx = {}
        try:
            for ln in open(FLOOR_META):
                r = json.loads(ln)
                _floor_idx[(r["filename"], int(r["start_frame"]))] = r
        except FileNotFoundError:
            pass
    return _floor_idx.get((stem, int(start_frame)))


def load_segments(stem):
    segs = []
    with open(META) as f:
        for ln in f:
            r = json.loads(ln)
            if r["filename"] == stem:
                segs.append((int(r["start_frame"]), int(r["end_frame"]), r["text"]))
    return sorted(segs)


def active_text(segs, g):
    for s0, e0, t in segs:
        if s0 <= g < e0:
            return t
    return ""


def skating_pool(posed30, ground=False, ground_offset=None):
    """contact-frame foot speeds (cm/s) for a (T,30,3) posed sequence.

    The contact-height gate (foot Y < 0.10) is relative to y=0, so the floor must
    be at y=0 first:
      - ``ground_offset`` given  -> subtract it (the GT floor): contact = foot near
        the GT floor. Use this when grounding by the per-slice GT floor.
      - ``ground=True``          -> subtract the window's min foot (local floor),
        for upper-floor/raised segments lacking a GT floor.
    Foot SPEED is offset-invariant; only WHICH frames count as contact changes."""
    if ground_offset is not None:
        posed30 = posed30.clone()
        posed30[:, :, 1] -= float(ground_offset)
    elif ground:
        posed30 = posed30.clone()
        posed30[:, :, 1] -= posed30[:, F30, 1].min()
    T = posed30.shape[0]
    vel = compute_vel_xyz(posed30.unsqueeze(0), FPS, lengths=torch.tensor([T]))[0]
    fc = foot_detect_from_pos_and_vel(posed30.unsqueeze(0), vel.unsqueeze(0), s30, 0.15, 0.10)[0].numpy()
    p = posed30.numpy(); out = []
    for k in range(4):
        foot = p[:, F30[k]]
        sp = np.linalg.norm(foot[1:] - foot[:-1], axis=-1) * FPS * 100.0
        c = (fc[1:, k] > 0.5) & (fc[:-1, k] > 0.5)
        out.append(sp[c])
    return np.concatenate(out) if out else np.zeros(0)


def overlay(frame, text, skate_line):
    img = Image.fromarray(frame); draw = ImageDraw.Draw(img); W = img.width
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        fsm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except Exception:
        font = fsm = ImageFont.load_default()
    lines = textwrap.fill(text or "(no annotation)", width=68).split("\n")
    sk_lines = skate_line.split("\n")
    box_h = 6 + 20 * len(lines) + 18 * len(sk_lines) + 4
    draw.rectangle([0, 0, W, box_h], fill=(0, 0, 0))
    for i, ln in enumerate(lines):
        draw.text((8, 4 + 20 * i), ln, fill=(255, 255, 255), font=font)
    for j, sl in enumerate(sk_lines):
        draw.text((8, 6 + 20 * len(lines) + 18 * j), sl, fill=(120, 230, 120), font=fsm)
    return np.asarray(img)


def render_one(npz, out, nframes=300, fps=20):
    stem = npz.split("/")[-1].replace(".npz", "")
    segs = load_segments(stem)
    d = np.load(npz, allow_pickle=True)
    lrm = torch.from_numpy(d["local_rot_mats"].astype(np.float32))
    root = torch.from_numpy(d["root_positions"].astype(np.float32))
    nj = torch.from_numpy(d["neutral_joints"].astype(np.float32))
    B = lrm.shape[0]
    _, posed77, _ = s77.fk(lrm, root, neutral_joints=nj.unsqueeze(0).expand(B, -1, -1))
    lrm30 = s30.from_SOMASkeleton77(lrm); nj30 = nj[IDX30]
    _, posed30, _ = s30.fk(lrm30, root, neutral_joints=nj30.unsqueeze(0).expand(B, -1, -1))

    seq_pool = skating_pool(posed30)
    seq_skate = float(seq_pool.mean()) if seq_pool.size else 0.0

    p = posed77.numpy()
    start = segs[0][0] if segs else 0
    nframes = min(nframes, B - start)
    seg = p[start:start + nframes].copy()
    # window skating (from the 30-joint posed over the same window)
    win_pool = skating_pool(posed30[start:start + nframes])
    win_skate = float(win_pool.mean()) if win_pool.size else 0.0
    # canonicalize: window-start root -> (x,z)=(0,0)
    seg[:, :, 0] -= p[start, ROOT, 0]
    seg[:, :, 2] -= p[start, ROOT, 2]
    # per-window ground: lowest foot in window -> y=0
    lf, rf = s77.bone_order_names.index("LeftFoot"), s77.bone_order_names.index("RightFoot")
    seg[:, :, 1] -= seg[:, [lf, rf], 1].min()

    frames = render_single(seg, par, caption="", max_frames=nframes, frame_stride=1,
                           skip_joints=SKIP, camera="follow")
    win_str = f"{win_skate:.1f} cm/s" if win_pool.size else "n/a (no floor contact)"
    seq_str = f"{seq_skate:.1f} cm/s" if seq_pool.size else "n/a"
    skate_line = f"foot skating: seq {seq_str} | window {win_str}"
    out_frames = [overlay(frames[i], active_text(segs, start + i), skate_line) for i in range(len(frames))]
    iio.imwrite(out, np.stack(out_frames), fps=fps, codec="libx264")
    print(f"wrote {out}  seq_skate={seq_skate:.2f} win_skate={win_skate:.2f}")


def _slug(text, n=40):
    keep = "".join(c if c.isalnum() or c == " " else "" for c in text.lower())
    return "_".join(keep.split())[:n] or "action"


def render_segment_windows(npz, out_dir, max_segs=3, min_len=40, prefer=("walk", "step", "go", "turn"),
                           gt_floor=False):
    """Render one clip PER atomic_action segment ([start_frame,end_frame]).

    gt_floor=True grounds each segment by its GT floor (`ground_offset_y` from
    metadata_atomic_action_floor.jsonl) instead of the per-window min-foot. This
    keeps sitting/lying/bent slices at the true room-floor height (min-foot would
    float the body up to the lowest joint)."""
    stem = npz.split("/")[-1].replace(".npz", "")
    segs = load_segments(stem)
    d = np.load(npz, allow_pickle=True)
    lrm = torch.from_numpy(d["local_rot_mats"].astype(np.float32))
    root = torch.from_numpy(d["root_positions"].astype(np.float32))
    nj = torch.from_numpy(d["neutral_joints"].astype(np.float32))
    B = lrm.shape[0]
    _, posed77, _ = s77.fk(lrm, root, neutral_joints=nj.unsqueeze(0).expand(B, -1, -1))
    lrm30 = s30.from_SOMASkeleton77(lrm); nj30 = nj[IDX30]
    _, posed30, _ = s30.fk(lrm30, root, neutral_joints=nj30.unsqueeze(0).expand(B, -1, -1))
    p = posed77.numpy()
    lf, rf = s77.bone_order_names.index("LeftFoot"), s77.bone_order_names.index("RightFoot")

    # qualifying segments: long enough, in-range; locomotion first
    cand = [(s0, e0, t) for (s0, e0, t) in segs if (e0 - s0) >= min_len and e0 <= B]
    cand.sort(key=lambda x: (0 if any(w in x[2].lower() for w in prefer) else 1, -(x[1] - x[0])))
    n = 0
    for (s0, e0, t) in cand[:max_segs]:
        seg = p[s0:e0].copy()
        seg[:, :, 0] -= p[s0, ROOT, 0]; seg[:, :, 2] -= p[s0, ROOT, 2]
        floor_line = None
        falpha = 0.55
        fr_row = floor_row(stem, s0) if gt_floor else None
        if fr_row and fr_row.get("floor_status") == "ok":
            off = fr_row["ground_offset_y"]
            seg[:, :, 1] -= off                              # GT-floor grounding (Y only)
            wp = skating_pool(posed30[s0:e0], ground_offset=off)   # skating relative to GT floor
            falpha = 0.9                                     # opaque floor: figure clearly above it
            amb = "  AMBIGUOUS(multi-floor)" if fr_row.get("ambiguous") else ""
            floor_line = f"GT floor surface_z={fr_row['floor_surface_z']:.2f}  n_floors={fr_row['n_floors_in_slice']}{amb}"
        else:
            seg[:, :, 1] -= seg[:, [lf, rf], 1].min()        # fallback: per-window min-foot
            wp = skating_pool(posed30[s0:e0], ground=True)
            if gt_floor:
                floor_line = f"GT floor: {fr_row.get('floor_status') if fr_row else 'no_row'} (min-foot fallback)"
        wsk = f"{wp.mean():.1f} cm/s ({wp.size} contact-frames)" if wp.size else "n/a (no floor contact)"
        frames = render_single(seg, par, caption="", max_frames=len(seg), frame_stride=1,
                               skip_joints=SKIP, camera="follow", floor_alpha=falpha)
        line = f"foot skating (segment, GT floor): {wsk}" + (f"\n{floor_line}" if floor_line else "")
        of = [overlay(frames[i], t, line) for i in range(len(frames))]
        tag = "gtfloor" if (fr_row and fr_row.get("floor_status") == "ok") else "minfoot"
        out = f"{out_dir}/{npz.split('/')[-2]}__{stem}__seg{s0:05d}_{tag}_{_slug(t)}.mp4"
        iio.imwrite(out, np.stack(of), fps=20, codec="libx264")
        print(f"wrote {out}  ({e0-s0} fr / {(e0-s0)/20:.1f}s)  '{t[:50]}'")
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="*", default=None)
    ap.add_argument("--n-auto", type=int, default=4)
    ap.add_argument("--out-dir", default="/home/jungbin_cho/_nymeria_text_viz")
    ap.add_argument("--nframes", type=int, default=300)
    ap.add_argument("--mode", choices=["window", "segments"], default="window")
    ap.add_argument("--max-segs", type=int, default=3)
    ap.add_argument("--gt-floor", action="store_true",
                    help="ground segments by the GT floor (ground_offset_y) instead of min-foot")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.npz:
        files = args.npz
    else:
        files = []
        for subj in sorted(os.listdir("/weka/jungbin/nymeriaplus_kimodo_proportional")):
            for f in sorted(glob.glob(f"/weka/jungbin/nymeriaplus_kimodo_proportional/{subj}/*.npz")):
                if load_segments(f.split("/")[-1].replace(".npz", "")):
                    files.append(f); break
            if len(files) >= args.n_auto:
                break
    for f in files:
        if args.mode == "segments":
            render_segment_windows(f, args.out_dir, max_segs=args.max_segs, gt_floor=args.gt_floor)
        else:
            out = f"{args.out_dir}/{f.split('/')[-2]}__{f.split('/')[-1].replace('.npz','.mp4')}"
            render_one(f, out, nframes=args.nframes)


if __name__ == "__main__":
    main()
