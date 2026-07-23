"""Download NymeriaPlus `object_bounding_box` artifacts for ALL subjects, sharded.

The official `nymeriaplus.downloader.DownloadManager` maps every sequence flatly to
`out_rootdir/<seq>/`, but the on-disk NymeriaPlus layout is per-subject:
    /weka/jungbin/nymeriaplus/<Sxx>/<seq>/
So this driver reuses the official `DownloadLink` (streaming download, sha1 verify,
zip extract, idempotent per-artifact flag) but points each sequence at its correct
subject directory, resolved via `_subject_map.json` ("subject" is a download bucket,
NOT derivable from the seq name).

Each `object_bounding_box` zip extracts to `<Sxx>/<seq>/objects/boxy/`
(`instances.json`, `scene_objects.csv`, `3dbb.csv`, `2dbb_*.csv`) -- the inputs that
`extract_floor.py` reads.

Idempotent: per-artifact flag `<Sxx>/.download_logs/<seq>/<key>` lets re-runs skip what
is already fetched, so re-running only retries failures. sha1 is verified inside
`DownloadLink.get` (raises on mismatch -> recorded as ERR_SHA1SUM, no flag written).

Shardable for parallel runs across nodes: `--shard i/N` processes the deterministic
subset `sorted(seqs)[idx % N == i]`. Run one shard per process; `/weka` is shared so
shards on different nodes never collide (each writes only its own seqs + flags).

USAGE (one shard):
  python fetch_objects_all.py \
      --url-json /weka/jungbin/nymeriaplus/nymeria_plus_download_urls_all_object_bbox.json \
      --shard 0/6
"""
from __future__ import annotations
import argparse, json, socket, sys, time
from pathlib import Path

sys.path.insert(0, "/home/jungbin_cho/nymeria_dataset")
from nymeriaplus.downloader import DownloadLink, DownloadStatus


def parse_shard(s: str) -> tuple[int, int]:
    i, n = s.split("/")
    i, n = int(i), int(n)
    if not (0 <= i < n):
        raise ValueError(f"--shard i/N requires 0 <= i < N, got {s}")
    return i, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url-json", type=Path, required=True,
                    help="all-subjects object_bounding_box signed-URL JSON")
    ap.add_argument("--nymeria-root", type=Path,
                    default=Path("/weka/jungbin/nymeriaplus"))
    ap.add_argument("--subject-map", type=Path,
                    default=Path("/weka/jungbin/nymeriaplus/_subject_map.json"))
    ap.add_argument("--key", default="object_bounding_box",
                    help="artifact key to download")
    ap.add_argument("--shard", default="0/1", help="i/N")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-download even if the per-artifact flag exists")
    args = ap.parse_args()

    shard_i, shard_n = parse_shard(args.shard)
    seqs = json.load(open(args.url_json))["sequences"]
    seq_to_subj = json.load(open(args.subject_map))["seq_to_subj"]

    # deterministic shard over all seqs that carry the artifact key
    have_key = sorted(s for s, v in seqs.items()
                      if isinstance(v, dict) and args.key in v)
    mine = [s for idx, s in enumerate(have_key) if idx % shard_n == shard_i]

    host = socket.gethostname()
    print(f"[shard {shard_i}/{shard_n}] host={host} key={args.key} "
          f"seqs={len(mine)}/{len(have_key)}", flush=True)

    counts: dict[str, int] = {}
    unmapped: list[str] = []
    t0 = time.time()
    for k, seq in enumerate(mine, 1):
        subj = seq_to_subj.get(seq)
        if subj is None:  # fallback: find an on-disk subject dir holding this seq
            hits = list(args.nymeria_root.glob(f"S*/{seq}"))
            subj = hits[0].parent.name if hits else None
        if subj is None:
            unmapped.append(seq)
            counts["UNMAPPED"] = counts.get("UNMAPPED", 0) + 1
            continue

        seq_dir = args.nymeria_root / subj / seq
        flag = args.nymeria_root / subj / ".download_logs" / seq / args.key
        link = DownloadLink.from_json(args.key, seqs[seq][args.key])
        try:
            status = link.get(seq_dir, destination=None, flag_path=flag,
                              ignore_existing=not args.overwrite)
        except Exception as e:  # noqa: BLE001
            status = link.status
            if status == DownloadStatus.UNKNOWN:
                status = DownloadStatus.ERR_NETWORK
            print(f"  ! {subj}/{seq}: {status.name}: {e}", flush=True)
        counts[status.name] = counts.get(status.name, 0) + 1

        if k % 10 == 0 or k == len(mine):
            rate = k / max(time.time() - t0, 1e-6) * 60
            done = counts.get("SUCCESS", 0) + counts.get("IGNORED", 0)
            print(f"  [{k}/{len(mine)}] ok+skip={done} "
                  f"{ {s: c for s, c in counts.items()} } "
                  f"rate={rate:.1f}/min", flush=True)

    out = args.nymeria_root / ".objbb_logs"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"host": host, "shard": args.shard, "key": args.key,
               "n_seqs": len(mine), "counts": counts, "unmapped": unmapped,
               "elapsed_min": round((time.time() - t0) / 60, 1)}
    json.dump(summary, open(out / f"shard_{shard_i}_of_{shard_n}.json", "w"), indent=2)
    print(f"\n=== shard {shard_i}/{shard_n} done in {summary['elapsed_min']} min ===")
    for s, c in sorted(counts.items()):
        print(f"  {s:16s}: {c}")
    if unmapped:
        print(f"  UNMAPPED seqs: {unmapped[:10]}{' ...' if len(unmapped) > 10 else ''}")


if __name__ == "__main__":
    main()
