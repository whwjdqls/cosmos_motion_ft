"""Stage A — extract the egocentric RGB video for each NymeriaPlus sequence as a
single mp4 that is **frame-aligned 1:1 with the kimodo motion NPZ** (and therefore
with the camera-trajectory sidecar), in the format the Cosmos 3 generator's video
data path expects.

Why a per-SEQUENCE mp4 (not per-slice clips, not per-frame webp):
  - Cosmos' SFT video dataset (`cosmos_framework/data/vfm/local_datasets/sft_dataset.py`)
    consumes **raw mp4 + JSONL metadata with frame windows** (`t2w_windows`), decoding
    frames at load time and running the Wan2.2 VAE on-the-fly. It does NOT use
    pre-encoded latents, per-frame images, or per-clip files. One mp4 per sequence +
    a window list is the native shape, and keeps 732 files instead of 146k clips.
  - The old `/weka/jungbin/nymeriaplus_kimodo/images` (224x224 per-frame webp) is
    superseded by this and can be deleted.

Alignment: we sample the head RGB stream at each motion frame's TIME_CODE timestamp
(`timestamps_us`), exactly like `camera/extract_camera_trajectory.py` and the old
`cache_images.py`. So mp4 frame i == motion frame i == camera sidecar frame i ==
egocentric observation at motion frame i.  Motion is ~20 fps (median dt 0.05 s).

Geometry: native Aria RGB is 1408x1408 uint8 RGB, rotated 90 deg from upright; we
`rotate(-90)` then resize to SIZE x SIZE (square == Cosmos "1,1" aspect bucket).
Default SIZE=640 so Cosmos' resize-to-cover can downsample cleanly to the 256p
(256) and 480p (640) square buckets without upscaling; 720p (960) would need a
re-extract at --size 960.

Out-of-span frames: a few motion frames at the very start/end can fall outside the
VRS time span. To keep a contiguous 1:1 mp4 we repeat the nearest decoded frame for
those and record the valid range (`valid_start`,`valid_end`,`n_invalid`) in the
sidecar so the manifest can clip training windows to valid frames.

Output (per sequence):
  /weka/jungbin/nymeriaplus_kimodo_proportional/video/{Sxx}/{seq}.mp4
  /weka/jungbin/nymeriaplus_kimodo_proportional/video/{Sxx}/{seq}.json   # sidecar meta
  ... plus a _done sentinel and a batch _video_summary.json

Pixels are stored as ordinary RGB; Cosmos normalizes to [-1,1] via /127.5-1 inside
the model, so NO normalization is applied here.

Env: `nymeria_plus` (projectaria_tools + the local `nymeriaplus` package). Encoding
uses the system ffmpeg (/usr/bin/ffmpeg) via a raw-rgb24 pipe (the env has no
imageio/av/cv2).
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/jungbin_cho/nymeria_dataset")

FFMPEG = "/usr/bin/ffmpeg"
MROOT = Path("/weka/jungbin/nymeriaplus_kimodo_proportional")
NROOT = Path("/weka/jungbin/nymeriaplus")
OUTROOT = MROOT / "video"
FPS = 20  # nominal; real per-frame timing is in the motion NPZ timestamps_us


def _open_ffmpeg(out_path: Path, size: int, fps: int, crf: int, preset: str):
    """Raw rgb24 frames in on stdin -> h264/yuv420p mp4 on disk."""
    cmd = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size}x{size}",
        "-framerate", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", str(crf), "-preset", preset,
        "-movflags", "+faststart", str(out_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def encode_one_sequence(args_tuple):
    (seq_dir_str, motion_npz_str, out_dir_str, size, crf, preset) = args_tuple
    seq_dir = Path(seq_dir_str)
    motion_npz = Path(motion_npz_str)
    out_dir = Path(out_dir_str)
    subj = seq_dir.parent.name
    seq = seq_dir.name

    from nymeriaplus.loaders.recording import RecordingLoader
    from projectaria_tools.core.sensor_data import TimeDomain

    out_mp4 = out_dir / f"{seq}.mp4"
    out_meta = out_dir / f"{seq}.json"
    sentinel = out_dir / "_done"
    if sentinel.exists() and out_mp4.exists() and out_meta.exists():
        return {"seq": seq, "status": "skip_done", "n_frames": 0, "elapsed_s": 0.0}

    rec = RecordingLoader(seq_dir / "recording_head")
    if not rec.has_rgb:
        return {"seq": seq, "status": "no_rgb", "n_frames": 0, "elapsed_s": 0.0}
    vrs_t0_ns, vrs_t1_ns = rec.get_global_timespan_ns()

    m = np.load(motion_npz)
    ts_us = m["timestamps_us"].astype(np.int64)
    target_ts_ns = ts_us * 1000
    n = int(len(target_ts_ns))

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    proc = _open_ffmpeg(out_mp4, size, FPS, crf, preset)

    last_frame = None            # last successfully decoded (size,size,3) uint8
    first_valid = last_valid = -1
    n_invalid = 0
    blank = np.zeros((size, size, 3), np.uint8)
    try:
        for fi, t_ns in enumerate(target_ts_ns):
            t = int(t_ns)
            frame = None
            if vrs_t0_ns <= t <= vrs_t1_ns:
                try:
                    img_data, _meta, _td = rec.get_rgb_image(t, TimeDomain.TIME_CODE)
                    arr = img_data.to_numpy_array()
                    pil = Image.fromarray(arr).rotate(-90, expand=True).resize(
                        (size, size), Image.LANCZOS)
                    frame = np.asarray(pil, np.uint8)
                except Exception:
                    frame = None
            if frame is not None:
                if first_valid < 0:
                    first_valid = fi
                last_valid = fi
                last_frame = frame
            else:
                n_invalid += 1
                frame = last_frame if last_frame is not None else blank
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        rc = proc.wait()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        return {"seq": seq, "status": "error", "trace": traceback.format_exc(),
                "elapsed_s": round(time.perf_counter() - t0, 1)}

    if rc != 0:
        return {"seq": seq, "status": "ffmpeg_fail", "rc": rc,
                "elapsed_s": round(time.perf_counter() - t0, 1)}

    dt = time.perf_counter() - t0
    meta = {
        "subject": subj, "filename": seq,
        "vision_path": str(out_mp4),
        "width": size, "height": size, "framerate": FPS,
        "nb_frames": n, "duration": round(n / FPS, 3),
        "valid_start": int(first_valid), "valid_end": int(last_valid),
        "n_invalid": int(n_invalid),
        "rotate_deg": -90, "native_res": 1408, "pix_fmt": "rgb24->yuv420p",
        "source": "nymeriaplus/recording_head camera-rgb @ motion timestamps_us (TIME_CODE)",
        "aligned_to": "kimodo motion frame i == mp4 frame i == camera sidecar frame i",
    }
    out_meta.write_text(json.dumps(meta, indent=2))
    sentinel.write_text(f"{n} frames, {n_invalid} invalid, {dt:.1f}s\n")
    return {"seq": seq, "status": "ok", "n_frames": n, "n_invalid": n_invalid,
            "elapsed_s": round(dt, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion-root", type=Path, default=MROOT)
    ap.add_argument("--nymeria-root", type=Path, default=NROOT)
    ap.add_argument("--out-root", type=Path, default=OUTROOT)
    ap.add_argument("--size", type=int, default=640,
                    help="square output edge (640 supports 256p+480p Cosmos buckets)")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", type=str, default="medium")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seqs", nargs="*", default=None,
                    help="restrict to subj/seq pairs, e.g. S02/20231006_s1_kirk_...")
    args = ap.parse_args()

    if args.seqs:
        motion_npzs = [args.motion_root / s.split("/", 1)[0] / f"{s.split('/',1)[1]}.npz"
                       for s in args.seqs]
    else:
        motion_npzs = sorted(args.motion_root.glob("S*/*.npz"))
    if args.limit:
        motion_npzs = motion_npzs[: args.limit]
    print(f"[scan] {len(motion_npzs)} motion NPZs")

    work = []
    for p in motion_npzs:
        subj, seq = p.parent.name, p.stem
        seq_dir = args.nymeria_root / subj / seq
        if not seq_dir.is_dir():
            continue
        out_dir = args.out_root / subj
        work.append((str(seq_dir), str(p), str(out_dir), args.size, args.crf, args.preset))
    print(f"[run] workers={args.workers} jobs={len(work)} size={args.size} out={args.out_root}")

    t0 = time.perf_counter()
    n_ok = n_skip = n_fail = 0
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(encode_one_sequence, w): w for w in work}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
                results.append(r)
                if r["status"] == "ok": n_ok += 1
                elif r["status"].startswith("skip"): n_skip += 1
                else: n_fail += 1
                if i % 5 == 0 or i == len(work):
                    el = time.perf_counter() - t0
                    rate = i / el
                    print(f"  [{i}/{len(work)}] ok={n_ok} skip={n_skip} fail={n_fail} "
                          f"rate={rate*60:.1f}/min eta={(len(work)-i)/rate/60:.1f}min "
                          f"last={r['seq']}({r.get('status')},{r.get('elapsed_s',0)}s)")
            except Exception:
                n_fail += 1
                traceback.print_exc()

    args.out_root.mkdir(parents=True, exist_ok=True)
    json.dump({"results": results, "totals": {"ok": n_ok, "skip": n_skip, "fail": n_fail},
               "size": args.size},
              open(args.out_root / "_video_summary.json", "w"), indent=2)
    print(f"\n=== done {(time.perf_counter()-t0)/60:.1f} min  ok={n_ok} skip={n_skip} fail={n_fail} ===")


if __name__ == "__main__":
    main()
