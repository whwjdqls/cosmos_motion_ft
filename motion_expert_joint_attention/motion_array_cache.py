"""Opt-in, read-only NPY cache for raw UniEgo motion arrays.

The training path never builds or changes cache files. The offline CLI copies
NPY members out of source NPZ archives without changing their bytes or dtype.
Window selection and all motion transforms remain owned by the datasets.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

import numpy as np


SCHEMA_VERSION = 1
ARRAY_NAMES = ("features.npy", "neutral_joints.npy")
_OPEN: OrderedDict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = OrderedDict()
_NOTICES: set[tuple[str, str]] = set()
_PID = os.getpid()


def _absolute(path: str | Path) -> Path:
    # Do not resolve symlinks: logical WEKA paths must retain their identity
    # when a site moves physical data to scratch behind a symlink.
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _source_root(override: str | None = None) -> Path:
    value = override or os.environ.get("COSMOS_MOTION_CACHE_SOURCE_ROOT")
    if not value:
        from runtime_paths import WEKA_ROOT

        value = str(WEKA_ROOT)
    return _absolute(value)


def _identity(source: Path, source_root: Path) -> dict:
    relative = source.relative_to(source_root).as_posix()
    stat = source.stat()
    return {
        "schema_version": SCHEMA_VERSION,
        "source_relative_path": relative,
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }


def _entry_path(cache_root: Path, identity: dict) -> Path:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return cache_root / f"v{SCHEMA_VERSION}" / digest[:2] / digest


def _read_metadata(entry: Path, identity: dict) -> None:
    with (entry / "metadata.json").open() as handle:
        metadata = json.load(handle)
    if metadata != identity:
        raise ValueError(f"cache identity mismatch: {entry}")


def _notice(root: str, kind: str, message: str) -> None:
    key = (root, kind)
    if key not in _NOTICES:
        _NOTICES.add(key)
        print(f"[motion-cache] {message}", file=sys.stderr, flush=True)


def _cached_arrays(source: str | Path) -> tuple[np.ndarray, np.ndarray] | None:
    configured = os.environ.get("COSMOS_MOTION_CACHE_ROOT", "").strip()
    if not configured:
        return None

    global _PID
    if _PID != os.getpid():
        # A DataLoader fork gets its own bounded handle cache, not a copy of
        # mappings opened by visualization or indexing in the parent process.
        _OPEN.clear()
        _NOTICES.clear()
        _PID = os.getpid()

    try:
        limit = int(os.environ.get("COSMOS_MOTION_CACHE_MAX_OPEN", "64"))
        if limit < 1:
            raise ValueError("COSMOS_MOTION_CACHE_MAX_OPEN must be positive")
        root = _absolute(configured)
        identity = _identity(_absolute(source), _source_root())
        entry = _entry_path(root, identity)
        key = (str(root), str(entry))
        arrays = _OPEN.get(key)
        if arrays is None:
            _read_metadata(entry, identity)
            features = np.load(entry / ARRAY_NAMES[0], mmap_mode="r", allow_pickle=False)
            neutral = np.load(entry / ARRAY_NAMES[1], mmap_mode="r", allow_pickle=False)
            if features.ndim != 2 or features.shape[1] != 283 or neutral.shape != (30, 3):
                raise ValueError(f"invalid UniEgo cache shapes in {entry}")
            arrays = (features, neutral)
            _OPEN[key] = arrays
        _OPEN.move_to_end(key)
        while len(_OPEN) > limit:
            # Drop references instead of forcibly closing a mapping which a
            # caller may still be slicing. NumPy owns its mapping lifetime.
            _OPEN.popitem(last=False)
        _notice(configured, "enabled", f"using read-only arrays in {root}; max_open={limit} per process")
        return arrays
    except FileNotFoundError:
        _notice(configured, "missing", "cache entry missing or stale; falling back to original NPZ files")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        _notice(configured, "invalid", f"cache unavailable; falling back to original NPZ files: {exc}")
    return None


def load_motion_window(
    source: str | Path, start: int, stop: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the same writable float32 copies as the original NPZ loaders."""
    arrays = _cached_arrays(source)
    if arrays is not None:
        return arrays[0][start:stop].astype(np.float32), arrays[1].astype(np.float32)
    with np.load(source) as archive:
        return (
            archive["features"][start:stop].astype(np.float32),
            archive["neutral_joints"].astype(np.float32),
        )


def build_entry(source: Path, source_root: Path, cache_root: Path) -> tuple[bool, int]:
    """Publish one immutable entry atomically; return (created, NPY bytes)."""
    identity = _identity(source, source_root)
    entry = _entry_path(cache_root, identity)
    if entry.exists():
        _read_metadata(entry, identity)
        if not all((entry / name).is_file() for name in ARRAY_NAMES):
            raise RuntimeError(f"incomplete published cache entry: {entry}; use a new cache root")
        return False, 0

    entry.parent.mkdir(parents=True, exist_ok=True)
    staging: Path | None = None
    try:
        with zipfile.ZipFile(source) as archive:
            required_bytes = sum(archive.getinfo(name).file_size for name in ARRAY_NAMES)
            free_bytes = shutil.disk_usage(entry.parent).free
            if free_bytes < required_bytes + 16 * 1024 * 1024:
                raise OSError(errno.ENOSPC, f"insufficient filesystem space for {source}")
            staging = Path(tempfile.mkdtemp(prefix=".building-", dir=entry.parent))
            for name in ARRAY_NAMES:
                with archive.open(name) as reader, (staging / name).open("wb") as writer:
                    shutil.copyfileobj(reader, writer, length=1024 * 1024)

        if _identity(source, source_root) != identity:
            raise RuntimeError(f"source changed while building cache: {source}")
        with (staging / "metadata.json").open("w") as handle:
            json.dump(identity, handle, sort_keys=True)
            handle.write("\n")
        try:
            os.rename(staging, entry)
            staging = None
        except OSError as exc:
            # Independent builders may safely race on the same source. Never
            # overwrite published arrays that another worker could be mapping.
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            _read_metadata(entry, identity)
            if not all((entry / name).is_file() for name in ARRAY_NAMES):
                raise RuntimeError(f"incomplete concurrently published entry: {entry}") from exc
            return False, 0
        return True, required_bytes
    finally:
        if staging is not None:
            shutil.rmtree(staging)


def _iter_sources(trees: list[str], manifests: list[str]):
    seen: set[Path] = set()
    for tree in trees:
        root = _absolute(tree)
        if not root.is_dir():
            raise FileNotFoundError(f"motion source tree does not exist: {root}")
        for directory, subdirs, files in os.walk(root):
            subdirs.sort()
            for name in sorted(files):
                if name.endswith(".npz"):
                    path = _absolute(Path(directory) / name)
                    if path not in seen:
                        seen.add(path)
                        yield path
    if manifests:
        from runtime_paths import resolve_legacy_path

        for manifest in manifests:
            with _absolute(manifest).open() as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    path = _absolute(resolve_legacy_path(json.loads(line)["uniego_path"]))
                    if path not in seen:
                        seen.add(path)
                        yield path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", default=os.environ.get("COSMOS_MOTION_CACHE_ROOT"))
    parser.add_argument("--source-root", help="Logical source prefix; defaults to this site's WEKA_ROOT")
    parser.add_argument("--tree", action="append", default=[], help="UniEgo NPZ tree; repeatable")
    parser.add_argument("--pairs-jsonl", action="append", default=[], help="Motion pairs manifest; repeatable")
    args = parser.parse_args()
    if not args.cache_root:
        parser.error("set --cache-root or COSMOS_MOTION_CACHE_ROOT")
    if not args.tree and not args.pairs_jsonl:
        parser.error("provide --tree and/or --pairs-jsonl")
    root = _source_root(args.source_root)
    cache = _absolute(args.cache_root)
    print(f"[motion-cache] source_root={root} cache_root={cache}", flush=True)
    created = reused = total_bytes = 0
    for source in _iter_sources(args.tree, args.pairs_jsonl):
        built, size = build_entry(source, root, cache)
        created += int(built)
        reused += int(not built)
        total_bytes += size
        if (created + reused) % 100 == 0:
            print(f"[motion-cache] built={created} reused={reused} written_GiB={total_bytes / 2**30:.2f}", flush=True)
    print(f"[motion-cache] finished: built={created} reused={reused} written_GiB={total_bytes / 2**30:.2f}", flush=True)


if __name__ == "__main__":
    main()
