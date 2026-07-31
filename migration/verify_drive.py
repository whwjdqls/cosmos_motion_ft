#!/usr/bin/env python
"""Verify the exact Google Drive archive declared by DRIVE_ARTIFACT_MANIFEST.tsv."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("DRIVE_ARTIFACT_MANIFEST.tsv")
DRIVE_REMOTE = os.environ.get("DRIVE_REMOTE", "data:").rstrip("/") + "/"


def read_manifest() -> list[dict[str, str]]:
    lines = [
        line
        for line in MANIFEST.read_text().splitlines()
        if line and not line.startswith("#")
    ]
    fields = (
        "id",
        "category",
        "kind",
        "local_path",
        "drive_path",
        "restore_path",
        "expected_count",
        "expected_bytes",
        "sha256",
        "status",
        "notes",
    )
    return list(csv.DictReader(lines, fieldnames=fields, delimiter="\t"))


def run_json(*args: str) -> dict[str, object]:
    result = subprocess.run(
        ["rclone", *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return json.loads(result.stdout)


def verify_file(row: dict[str, str]) -> list[str]:
    remote = DRIVE_REMOTE + row["drive_path"]
    try:
        stat = run_json("lsjson", "--stat", "--hash", remote)
    except subprocess.CalledProcessError:
        return [f"{row['id']}: missing or unreadable file {remote}"]
    failures: list[str] = []
    expected_bytes = int(row["expected_bytes"])
    if stat.get("IsDir") or stat.get("Size") != expected_bytes:
        failures.append(
            f"{row['id']}: size {stat.get('Size')} != {expected_bytes}"
        )
    expected_hash = row["sha256"]
    hashes = stat.get("Hashes", {})
    actual_hash = hashes.get("sha256") if isinstance(hashes, dict) else None
    if expected_hash != "-" and actual_hash != expected_hash:
        failures.append(
            f"{row['id']}: sha256 {actual_hash} != {expected_hash}"
        )
    return failures


def verify_tree(row: dict[str, str]) -> list[str]:
    remote = DRIVE_REMOTE + row["drive_path"]
    try:
        size = run_json("size", "--json", "--fast-list", remote)
    except subprocess.CalledProcessError:
        return [f"{row['id']}: missing or unreadable tree {remote}"]
    failures: list[str] = []
    expected_count = int(row["expected_count"])
    expected_bytes = int(row["expected_bytes"])
    if size.get("count") != expected_count:
        failures.append(
            f"{row['id']}: count {size.get('count')} != {expected_count}"
        )
    if size.get("bytes") != expected_bytes:
        failures.append(
            f"{row['id']}: bytes {size.get('bytes')} != {expected_bytes}"
        )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-trees",
        action="store_true",
        help="also traverse trees and compare exact object counts/bytes",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        help="verify only the named manifest ids",
    )
    args = parser.parse_args()

    selected = set(args.ids or ())
    failures: list[str] = []
    checked = 0
    skipped_uploads = 0
    for row in read_manifest():
        if selected and row["id"] not in selected:
            continue
        if row["status"] != "verified":
            skipped_uploads += 1
            print(f"[drive] skip {row['id']}: status={row['status']}")
            continue
        if row["kind"] == "tree" and not args.include_trees:
            continue
        verify = verify_file if row["kind"] == "file" else verify_tree
        row_failures = verify(row)
        failures.extend(row_failures)
        checked += 1
        result = "FAIL" if row_failures else "ok"
        print(f"[drive] {result} {row['id']}")

    if failures:
        raise SystemExit("\n".join(failures))
    print(
        f"[drive] verification passed checked={checked} "
        f"pending_or_skipped={skipped_uploads} trees={args.include_trees}"
    )


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as error:
        print(f"missing required executable or manifest: {error}", file=sys.stderr)
        raise SystemExit(2) from error
