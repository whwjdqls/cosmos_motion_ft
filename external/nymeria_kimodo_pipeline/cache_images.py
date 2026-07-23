"""DEPRECATED (2026-06-18) — superseded by `video/extract_ego_video.py`.

This produced 224x224 per-frame webp under /weka/jungbin/nymeriaplus_kimodo/images/
(now deleted) which was too small for any Cosmos resolution bucket (>=256) and the
wrong on-disk shape (Cosmos wants raw mp4 + JSONL windows, not per-frame images).
Use `video/extract_ego_video.py` (per-seq mp4, square 640, frame-aligned to motion)
instead. Kept only for reference.

Production driver for egocentric image cache.

For every kimodo motion NPZ under --motion-root, decode the corresponding
head VRS at its 20-fps timestamps, downsample to --size x --size, save as
per-frame WebP under --out-root/{Sxx}/{seq_name}/frame_{idx:06d}.webp.

Idempotent: a sequence whose output dir contains N expected webp files is
skipped (full-length re-check would be expensive; we use a sentinel _done
file as the fast path).

Parallelism: --workers processes, each handles one sequence end-to-end.
VRS decode is CPU-bound (libvrs C++), workers don't share state.
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Allow Python multi-thread libs to coexist with our N workers
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/jungbin_cho/nymeria_dataset")


def encode_one_sequence(args_tuple):
    """Encode one sequence's egocentric images. Runs in a worker process."""
    (seq_dir_str, motion_npz_str, out_dir_str, size, quality, webp_method) = args_tuple
    seq_dir = Path(seq_dir_str)
    motion_npz = Path(motion_npz_str)
    out_dir = Path(out_dir_str)

    from nymeriaplus.loaders.recording import RecordingLoader
    from projectaria_tools.core.sensor_data import TimeDomain

    sentinel = out_dir / "_done"
    if sentinel.exists():
        return {"seq": seq_dir.name, "status": "skip_done", "n_frames": 0, "elapsed_s": 0.0}

    # Recording with RGB?
    rec = RecordingLoader(seq_dir / "recording_head")
    if not rec.has_rgb:
        return {"seq": seq_dir.name, "status": "no_rgb", "n_frames": 0, "elapsed_s": 0.0}
    vrs_t0_ns, vrs_t1_ns = rec.get_global_timespan_ns()

    m = np.load(motion_npz)
    ts_us = m["timestamps_us"]
    target_ts_ns = (ts_us.astype(np.int64) * 1000)
    n = len(target_ts_ns)

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    n_written = n_skipped = 0
    for frame_idx, t_ns in enumerate(target_ts_ns):
        t_ns_int = int(t_ns)
        if t_ns_int < vrs_t0_ns or t_ns_int > vrs_t1_ns:
            n_skipped += 1
            continue
        out_path = out_dir / f"frame_{frame_idx:06d}.webp"
        if out_path.exists():
            n_written += 1
            continue
        try:
            img_data, img_meta, tdiff = rec.get_rgb_image(t_ns_int, TimeDomain.TIME_CODE)
            arr = img_data.to_numpy_array()
        except Exception as e:
            n_skipped += 1
            continue
        pil = Image.fromarray(arr).rotate(-90, expand=True).resize((size, size), Image.BILINEAR)
        pil.save(out_path, format="WEBP", quality=quality, method=webp_method)
        n_written += 1
    dt = time.perf_counter() - t0
    sentinel.write_text(f"{n_written}/{n} written; {n_skipped} skipped; {dt:.1f}s\n")
    return {
        "seq": seq_dir.name,
        "status": "ok",
        "n_frames": int(n),
        "n_written": int(n_written),
        "n_skipped": int(n_skipped),
        "elapsed_s": round(dt, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion-root", type=Path,
                    default=Path("/weka/jungbin/nymeriaplus_kimodo/motions"))
    ap.add_argument("--nymeria-root", type=Path,
                    default=Path("/weka/jungbin/nymeriaplus"))
    ap.add_argument("--out-root", type=Path,
                    default=Path("/weka/jungbin/nymeriaplus_kimodo/images"))
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--webp-method", type=int, default=2,
                    help="WebP encode effort 0-6 (lower=faster).")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    motion_npzs = sorted(args.motion_root.glob("S*/*.npz"))
    if args.limit:
        motion_npzs = motion_npzs[: args.limit]
    print(f"[scan] {len(motion_npzs)} motion NPZs")

    work = []
    for p in motion_npzs:
        subj = p.parent.name
        seq_name = p.stem
        seq_dir = args.nymeria_root / subj / seq_name
        if not seq_dir.is_dir():
            continue
        out_dir = args.out_root / subj / seq_name
        work.append((str(seq_dir), str(p), str(out_dir),
                     args.size, args.quality, args.webp_method))

    print(f"[run] workers={args.workers}  jobs={len(work)}  out_root={args.out_root}")
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
                if i % 10 == 0 or i == len(work):
                    elapsed = time.perf_counter() - t0
                    rate = i / elapsed
                    eta = (len(work) - i) / rate
                    print(f"  [{i}/{len(work)}] ok={n_ok} skip={n_skip} fail={n_fail}  "
                          f"rate={rate*60:.1f}/min  eta={eta/60:.1f}min  "
                          f"last={r['seq']}({r.get('elapsed_s', 0)}s)")
            except Exception as e:
                n_fail += 1
                traceback.print_exc()

    args.out_root.mkdir(parents=True, exist_ok=True)
    json.dump({"results": results, "totals": {"ok": n_ok, "skip": n_skip, "fail": n_fail}},
              open(args.out_root / "_batch_summary.json", "w"), indent=2)
    print(f"\n=== done in {(time.perf_counter()-t0)/60:.1f} min ===")
    print(f"  ok   : {n_ok}")
    print(f"  skip : {n_skip}")
    print(f"  fail : {n_fail}")


if __name__ == "__main__":
    main()
