"""Download NymeriaPlus object_bounding_box (3D scene/object boxes) artifacts.

Thin wrapper over the official `nymeriaplus.downloader.DownloadManager`. Given a
`nymeria_plus_download_urls_<Sxx>_objects.json` URL file, it downloads,
sha1-verifies, and extracts each sequence's `object_bounding_box` zip into
  <out-root>/<seq_name>/...
so the boxes land next to that sequence's existing `body/`, `recording_head/`,
etc. The 3D object bounding boxes (used here to recover floor height) extract
into a per-sequence subdir.

Idempotent: per-artifact flag files under `<out-root>/.download_logs/<seq>/<key>`
let re-runs skip already-fetched zips. Writes `<out-root>/download_summary.json`.

Usage (out-root is the SUBJECT dir, since the manager creates <out-root>/<seq>/):
  python fetch_objects.py \
      --url-json /weka/jungbin/nymeriaplus/nymeria_plus_download_urls_S04_objects.json \
      --out-root /weka/jungbin/nymeriaplus/S04
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, "/home/jungbin_cho/nymeria_dataset")
from nymeriaplus.downloader import DownloadManager


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url-json", type=Path, required=True,
                    help="nymeria_plus_download_urls_<Sxx>_objects.json")
    ap.add_argument("--out-root", type=Path, required=True,
                    help="Subject dir; sequences extract to <out-root>/<seq>/")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-download even if per-artifact flag exists.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    mgr = DownloadManager(args.url_json, args.out_root)
    plan = mgr.build_plan()
    print(f"[plan] sequences={plan.num_sequences} artifacts={plan.num_artifacts} "
          f"size={plan.total_size_gib:.2f} GiB out_root={args.out_root}")

    summary = mgr.download(ignore_existing=not args.overwrite)
    print("\n=== download summary ===")
    for status, n in summary.items():
        if n:
            print(f"  {status:16s}: {n}")
    print(f"\nwrote {mgr.logfile}")


if __name__ == "__main__":
    main()
