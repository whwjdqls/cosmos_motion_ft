"""Opt-in, exact-output caches for frozen TI2M conditioning.

Keep the version strings in sync with preprocessing changes. Cache storage is
disposable; a missing/unreadable entry always uses the original implementation.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import pickle
import queue
import threading
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch


_WARNED: set[str] = set()
_WRITE_AFTER: dict[str, float] = {}
_SPACE_AFTER: dict[str, float] = {}
_READ_POOL = None
_READ_POOL_KEY = None
_WRITE_QUEUE = None
_WRITE_QUEUE_PID = None


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(message, RuntimeWarning, stacklevel=3)


def _path(root: str, kind: str, identity: object, suffix: str) -> Path:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return Path(root).expanduser() / kind / digest[:2] / (digest + suffix)


def _write(path: Path, writer) -> None:
    """Publish without replacing another worker's entry or consuming RNG state."""
    root = str(path.parents[2])
    now = time.monotonic()
    if now < _WRITE_AFTER.get(root, 0):
        return
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if now >= _SPACE_AFTER.get(root, 0):
            space = os.statvfs(path.parent)
            minimum = float(os.environ.get("COSMOS_CONDITION_CACHE_MIN_FREE_GIB", "20"))
            if space.f_bavail * space.f_frsize < minimum * 1024**3:
                raise OSError("conditioning cache filesystem is low on free space")
            _SPACE_AFTER[root] = now + 60
        fd, temporary = tempfile.mkstemp(prefix=".cache-", dir=path.parent)
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
    except (OSError, RuntimeError) as error:
        _WRITE_AFTER[root] = now + 60
        _warn_once(root, f"Conditioning cache writes paused; original path remains available: {error}")
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def cache_reasoner_image(function):
    """Cache the final resized uint8 frame, not a different video decode path."""

    @functools.wraps(function)
    def wrapped(self, vis, s):
        root = os.environ.get("COSMOS_REASONER_IMAGE_CACHE_ROOT", "")
        if not root:
            return function(self, vis, s)
        try:
            source = Path(vis)
            stat = source.stat()
            size = int(self._reasoner_image_size)
            identity = {
                "version": "nymeria-pyav-bilinear-antialias-round-uint8-v1",
                "path": str(source.absolute()),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "frame": int(s),
                "fps": float(self._fps),
                "size": size,
            }
            path = _path(root, "frames-v1", identity, ".npy")
        except OSError:
            return function(self, vis, s)
        try:
            frame = np.load(path, allow_pickle=False)
            if frame.dtype != np.uint8 or frame.shape != (3, size, size):
                raise ValueError("unexpected cached frame layout")
            return torch.from_numpy(frame)
        except FileNotFoundError:
            pass
        except (OSError, ValueError, EOFError) as error:
            _warn_once(str(Path(root)), f"Ignoring unreadable frame cache entry: {error}")
        image = function(self, vis, s)
        # Do not let a future preprocessing/layout change silently reuse v1.
        if image.dtype == torch.uint8 and tuple(image.shape) == (3, size, size):
            array = image.detach().cpu().numpy()
            _write(path, lambda stream: np.save(stream, array, allow_pickle=False))
        return image

    return wrapped


def _condition_cache_location(self, text, image_chw, image_size):
    root = os.environ.get("COSMOS_EDGE_CONDITION_CACHE_ROOT", "")
    namespace = os.environ.get("COSMOS_EDGE_CONDITION_CACHE_ID", "")
    if not namespace:
        raise ValueError("COSMOS_EDGE_CONDITION_CACHE_ID is required when enabling frozen conditioning cache")
    cpu_image = image_chw.detach().cpu().contiguous()
    identity = {
        "version": "edge-serial-image-text-condition-v1",
        "frozen_weights_and_processor": namespace,
        "caption": text,
        "image": hashlib.sha256(cpu_image.numpy().tobytes()).hexdigest(),
        "shape": list(cpu_image.shape),
        "image_dtype": str(cpu_image.dtype),
        "image_size": int(image_size),
        "model_dtype": str(self.dtype),
        "processor": type(self.processor).__qualname__,
        "tokenizer": type(self.tokenizer).__qualname__,
    }
    return _path(root, "edge-conditions-v1", identity, ".pt"), identity


def _read_condition_cpu(path, identity):
    """CPU-only work: safe to prefetch while the main thread uses the GPU."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["identity"] != identity:
            raise ValueError("frozen conditioning cache identity mismatch")
        condition = payload["condition"]
        if not isinstance(condition, dict) or not {
            "input_ids", "inputs_embeds", "position_ids", "next_position_id"
        }.issubset(condition):
            raise ValueError("unexpected cached conditioning layout")
        return condition
    except FileNotFoundError:
        return None
    except (OSError, ValueError, EOFError, RuntimeError, KeyError, TypeError, pickle.UnpicklingError) as error:
        _warn_once(str(path.parents[2]), f"Ignoring unreadable frozen conditioning cache entry: {error}")
        return None


def prefetch_frozen_image_conditions(function):
    """Submit batch cache reads concurrently; preserve the serial GPU encoder."""
    @functools.wraps(function)
    def wrapped(self, captions, modes=None, reasoner_images=None, image_size=256):
        global _READ_POOL, _READ_POOL_KEY
        workers = int(os.environ.get("COSMOS_EDGE_CONDITION_PREFETCH_THREADS", "0"))
        edge = self._edge
        if (workers <= 0 or not os.environ.get("COSMOS_EDGE_CONDITION_CACHE_ROOT")
                or edge.backbone.training or edge.visual.training
                or modes is None or reasoner_images is None
                or len(captions) != len(modes) or len(captions) != len(reasoner_images)):
            return function(self, captions, modes=modes, reasoner_images=reasoner_images, image_size=image_size)
        key = (os.getpid(), min(workers, 8))
        if _READ_POOL_KEY != key:
            if _READ_POOL is not None and _READ_POOL_KEY[0] == os.getpid():
                _READ_POOL.shutdown(wait=False, cancel_futures=True)
            _READ_POOL = ThreadPoolExecutor(max_workers=key[1], thread_name_prefix="condition-read")
            _READ_POOL_KEY = key
        pending = {}
        for caption, mode, image in zip(captions, modes, reasoner_images):
            if mode == "textimg2motion" and image is not None:
                path, identity = _condition_cache_location(edge, caption, image, image_size)
                if str(path) not in pending:
                    pending[str(path)] = _READ_POOL.submit(_read_condition_cpu, path, identity)
        previous = getattr(edge, "_prefetched_condition_reads", None)
        edge._prefetched_condition_reads = pending
        try:
            return function(self, captions, modes=modes, reasoner_images=reasoner_images, image_size=image_size)
        finally:
            edge._prefetched_condition_reads = previous
            for future in pending.values():
                future.cancel()
    return wrapped


def _write_condition(path, identity, condition):
    """Queue only detached CPU tensors. Full queues drop fills, never batches."""
    global _WRITE_QUEUE, _WRITE_QUEUE_PID
    payload = {"identity": identity, "condition": condition}
    if os.environ.get("COSMOS_EDGE_CONDITION_ASYNC_WRITES", "0") != "1":
        _write(path, lambda stream: torch.save(payload, stream))
        return
    if _WRITE_QUEUE_PID != os.getpid():
        _WRITE_QUEUE = queue.Queue(maxsize=64)
        _WRITE_QUEUE_PID = os.getpid()
        def consume(work):
            while True:
                destination, contents = work.get()
                try:
                    _write(destination, lambda stream: torch.save(contents, stream))
                except Exception as error:
                    _warn_once("condition-writer", f"Skipping failed background cache fill: {error}")
                finally:
                    work.task_done()
        threading.Thread(target=consume, args=(_WRITE_QUEUE,), daemon=True,
                         name="condition-write").start()
    try:
        _WRITE_QUEUE.put_nowait((path, payload))
    except queue.Full:
        pass


def flush_condition_cache_writes(timeout=5.0):
    """Best-effort drain for diagnostics; training never needs to wait for fills."""
    if _WRITE_QUEUE_PID != os.getpid() or _WRITE_QUEUE is None:
        return True
    deadline = time.monotonic() + timeout
    while _WRITE_QUEUE.unfinished_tasks:
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


def cache_frozen_image_condition(function):
    """Cache the *existing* serial encoder output, including caption and positions.

    The explicit ID must identify immutable frozen weights and their processor.
    It intentionally does not use the trainable motion checkpoint as an ID.
    """

    @functools.wraps(function)
    def wrapped(self, text, image_chw, image_size=256):
        root = os.environ.get("COSMOS_EDGE_CONDITION_CACHE_ROOT", "")
        if not root:
            return function(self, text, image_chw, image_size=image_size)
        namespace = os.environ.get("COSMOS_EDGE_CONDITION_CACHE_ID", "")
        if not namespace:
            raise ValueError("COSMOS_EDGE_CONDITION_CACHE_ID is required when enabling frozen conditioning cache")
        # Cache only the frozen eval-mode path. Never suppress trainable grads.
        if self.backbone.training or self.visual.training:
            return function(self, text, image_chw, image_size=image_size)
        if not getattr(self, "_condition_cache_frozen_checked", False):
            if any(p.requires_grad for module in (self.backbone, self.visual) for p in module.parameters()):
                raise ValueError("Frozen conditioning cache cannot be used with trainable image/text weights")
            self._condition_cache_frozen_checked = True
        path, identity = _condition_cache_location(self, text, image_chw, image_size)
        pending = getattr(self, "_prefetched_condition_reads", None) or {}
        future = pending.get(str(path))
        condition = future.result() if future is not None else _read_condition_cpu(path, identity)
        if condition is not None:
            return {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in condition.items()
            }
        condition = function(self, text, image_chw, image_size=image_size)
        cpu_condition = {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in condition.items()
        }
        _write_condition(path, identity, cpu_condition)
        return condition

    return wrapped
