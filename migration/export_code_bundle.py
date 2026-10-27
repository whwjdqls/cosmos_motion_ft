#!/usr/bin/env python3
"""Export this working tree, including unpublished source, without data or .git."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New .tar.gz outside this checkout")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == REPO or REPO in output.parents:
        parser.error("Put the bundle outside the source checkout")
    if output.exists():
        parser.error("Refusing to overwrite an existing bundle")
    names = subprocess.check_output(
        ["git", "-C", str(REPO), "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
    ).decode().split("\0")
    manifest = {"format": 1, "git_head": subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "includes_working_tree_changes": True, "files": {}}
    selected = []
    for name in sorted(set(names) - {""}):
        path = REPO / name
        if path.suffix in (".out", ".log", ".mp4") or not path.exists():
            continue
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Review non-regular source before exporting: {name}")
        if path.stat().st_size > 20 * 1024**2:
            raise RuntimeError(f"Review large file before exporting: {name}")
        data = path.read_bytes()
        manifest["files"][name] = hashlib.sha256(data).hexdigest()
        selected.append((name, data, 0o755 if path.stat().st_mode & 0o111 else 0o644))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as raw:
        output.chmod(0o600)
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name, data, mode in selected + [("CODE_BUNDLE_MANIFEST.json", (json.dumps(manifest, indent=2) + "\n").encode(), 0o644)]:
                    member = tarfile.TarInfo("cosmos_motion_ft/" + name)
                    member.size, member.mode = len(data), mode
                    archive.addfile(member, io.BytesIO(data))
    with tarfile.open(output) as archive:
        for name, digest in manifest["files"].items():
            actual = archive.extractfile("cosmos_motion_ft/" + name).read()
            if hashlib.sha256(actual).hexdigest() != digest:
                raise RuntimeError(f"Bundle verification failed: {name}")
    print(json.dumps({"bundle": str(output), "files": len(selected), "bytes": output.stat().st_size,
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
