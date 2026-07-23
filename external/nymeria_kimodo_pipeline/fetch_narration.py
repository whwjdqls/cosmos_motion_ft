"""Fetch only the narration zips for any NymeriaPlus sequence whose
on-disk `narration/` subdir is missing.

Reuses the official nymeriaplus.downloader.DownloadLink so SHA1
verification + extraction logic is identical to nymeriaplus-download.
"""
from __future__ import annotations
import glob, json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "/home/jungbin_cho/nymeria_dataset")
from nymeriaplus.downloader import DownloadLink, DownloadStatus

BASE = Path("/weka/jungbin/nymeriaplus")
WORKERS = 8  # narration zips are ~10 KB; concurrent fetches are safe


def collect_missing() -> list[tuple[Path, str, DownloadLink]]:
    """Return list of (sequence_dir, seq_name, DownloadLink) for sequences
    whose URL JSON lists a narration entry but disk has no narration/ subdir.
    """
    items = []
    for url_p in sorted(glob.glob(str(BASE / "S*" / "nymeria_plus_download_urls_*.json"))):
        subj_dir = Path(url_p).parent
        d = json.load(open(url_p))
        for seq_name, info in d["sequences"].items():
            narr = info.get("narration")
            if not narr:
                continue
            seq_dir = subj_dir / seq_name
            if (seq_dir / "narration").is_dir():
                continue  # already present
            link = DownloadLink.from_json("narration", narr)
            items.append((seq_dir, seq_name, link))
    return items


def fetch_one(seq_dir: Path, seq_name: str, link: DownloadLink) -> tuple[str, DownloadStatus, str]:
    """Download narration zip for one sequence. Returns (seq_name, status, msg)."""
    subj_dir = seq_dir.parent
    flag_path = subj_dir / ".download_logs" / seq_name / "narration"
    try:
        status = link.get(
            sequence_dir=seq_dir,
            destination=None,  # zip → extracted in-place
            flag_path=flag_path,
            ignore_existing=True,
        )
        return seq_name, status, "ok"
    except Exception as e:
        return seq_name, DownloadStatus.ERR_NETWORK, repr(e)[:120]


def main():
    print(f"[scan] looking for missing narration across {BASE} ...")
    items = collect_missing()
    print(f"[scan] {len(items)} sequence(s) need narration downloaded")
    if not items:
        print("Nothing to do."); return

    total_bytes = sum(it[2].file_size_bytes for it in items)
    print(f"[scan] total bytes to fetch: {total_bytes/1024:.1f} KB")
    print()

    t0 = time.perf_counter()
    ok = err = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(fetch_one, *it): it for it in items}
        for i, fut in enumerate(as_completed(futs), 1):
            seq_name, status, msg = fut.result()
            tag = "OK " if status == DownloadStatus.SUCCESS else "ERR"
            if status == DownloadStatus.SUCCESS:
                ok += 1
            else:
                err += 1
            print(f"[{i:3d}/{len(items)}] {tag} {seq_name}  status={status.value}  {msg}")

    dt = time.perf_counter() - t0
    print()
    print(f"=== done in {dt:.1f}s ===")
    print(f"  success: {ok}")
    print(f"  errors : {err}")


if __name__ == "__main__":
    main()
