#!/usr/bin/env python3
"""Restore pinned upstream code and our patches; never install weights or packages."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

REPO = Path(__file__).resolve().parents[1]
LOCK = Path(__file__).with_name("dependencies.lock.json")


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def verify(root: Path, spec: dict) -> None:
    if not (root / ".git").is_dir():
        raise RuntimeError(f"Not a standalone checkout: {root}")
    if git(root, "rev-parse", "HEAD") != spec["commit"]:
        raise RuntimeError(f"Unexpected revision: {root}")
    expected = spec.get("patched_files", {})
    for name, digest in expected.items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Patched source differs: {path}")
    # Allow ignored runtime outputs, but never silently accept source drift.
    modified = set(git(root, "diff", "HEAD", "--name-only").splitlines())
    untracked = set(git(root, "ls-files", "--others", "--exclude-standard").splitlines())
    if (modified | untracked) - set(expected):
        raise RuntimeError(f"Unrecorded changes in {root}: {sorted((modified | untracked) - set(expected))}")
    if spec.get("patch"):
        patch = REPO / spec["patch"]
        if hashlib.sha256(patch.read_bytes()).hexdigest() != spec["patch_sha256"]:
            raise RuntimeError(f"Patch checksum differs: {patch}")
        subprocess.run(["git", "-C", str(root), "apply", "--reverse", "--check", str(patch)], check=True)


def restore(name: str, spec: dict, destination: Path, mirror_root: Path | None, check: bool) -> None:
    target = destination / name
    if target.exists():
        verify(target, spec)
        print(f"Verified {name} {spec['commit']}", flush=True)
        return
    if check:
        raise RuntimeError(f"Missing checkout: {target}")
    destination.mkdir(parents=True, exist_ok=True)
    # Stage beside the target; publish only a completely verified checkout.
    temporary = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=destination))
    try:
        mirror = mirror_root / name if mirror_root else None
        if mirror is not None and (mirror / ".git").is_dir():
            subprocess.run(["git", "clone", "--no-checkout", "--no-hardlinks", str(mirror), str(temporary)], check=True)
            subprocess.run(["git", "-C", str(temporary), "remote", "set-url", "origin", spec["url"]], check=True)
        else:
            subprocess.run(["git", "init", "-q", str(temporary)], check=True)
            subprocess.run(["git", "-C", str(temporary), "remote", "add", "origin", spec["url"]], check=True)
        found = subprocess.run(["git", "-C", str(temporary), "cat-file", "-e", spec["commit"] + "^{commit}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if found.returncode:
            subprocess.run(["git", "-C", str(temporary), "fetch", "--depth=1", "origin", spec["commit"]], check=True)
        subprocess.run(["git", "-C", str(temporary), "checkout", "--detach", spec["commit"]], check=True)
        if spec.get("patch"):
            patch = REPO / spec["patch"]
            if hashlib.sha256(patch.read_bytes()).hexdigest() != spec["patch_sha256"]:
                raise RuntimeError(f"Patch checksum differs: {patch}")
            subprocess.run(["git", "-C", str(temporary), "apply", "--check", str(patch)], check=True)
            subprocess.run(["git", "-C", str(temporary), "apply", str(patch)], check=True)
        verify(temporary, spec)
        temporary.rename(target)
        print(f"Restored {name} {spec['commit']}", flush=True)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    specs = json.loads(LOCK.read_text())["sources"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("components", nargs="*", help="Defaults to all locked dependencies")
    parser.add_argument("--destination", type=Path, default=REPO / ".deps")
    parser.add_argument("--mirror-root", type=Path, help="Optional local Git clones; commits still checked")
    parser.add_argument("--check", action="store_true", help="Verify only; no fetches or changes")
    args = parser.parse_args()
    components = args.components or list(specs)
    for name in components:
        if name not in specs:
            parser.error(f"Unknown component {name!r}; choose from {', '.join(specs)}")
    for name in components:
        restore(name, specs[name], args.destination.resolve(), args.mirror_root, args.check)


if __name__ == "__main__":
    main()
