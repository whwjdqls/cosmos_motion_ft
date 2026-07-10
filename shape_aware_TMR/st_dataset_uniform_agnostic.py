"""Uniform BONES-SEED dataset for shape-agnostic TMR.

The uniform 20fps tree stores ``local_rot_mats``/``root_positions``/``posed_joints``
instead of proportional uniego ``features``/``neutral_joints``. This dataset
extracts the 30-joint SOMA subset and feeds it through the same 186-d
``TMRMotionRep`` used by the shape-aware model.
"""
from __future__ import annotations

import csv
import logging

import numpy as np
import torch

from kimodo.data.soma_text_motion import _BSEntry, _truthy
from kimodo.skeleton import SOMASkeleton30

from st_dataset import ShapeTMRDataset

log = logging.getLogger(__name__)


class UniformAgnosticTMRDataset(ShapeTMRDataset):
    """Shape-agnostic dataset over ``soma_uniform_motions_20fps``."""

    def __init__(self, *args, **kwargs):
        self.soma30 = SOMASkeleton30()
        super().__init__(*args, **kwargs)

    def _build_natural_pool(self, out, path_by_name):
        desc_cols = self.natural_desc_cols
        meta_by_file = {}
        no_motion = mirror = no_text = 0
        with open(self.natural_csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row["filename"]
                if not self.include_mirrored and (_truthy(row.get("is_mirror")) or fname.endswith("_M")):
                    mirror += 1
                    continue
                if fname not in path_by_name:
                    no_motion += 1
                    continue
                descs, seen = [], set()
                for k in desc_cols:
                    v = row.get(k, "")
                    if isinstance(v, str):
                        v = v.strip()
                    if v and v not in seen:
                        seen.add(v)
                        descs.append(v)
                if not descs:
                    no_text += 1
                    continue
                try:
                    nframes = int(float(row["move_duration_frames"]))
                except (KeyError, TypeError, ValueError):
                    no_text += 1
                    continue
                meta_by_file[fname] = (descs, nframes)

        long_seg = short_seg = bad_read = 0
        for fname, (descs, n_total) in meta_by_file.items():
            p = path_by_name[fname]
            t_sec = n_total / float(self.fps)
            if t_sec > self.max_segment_sec:
                long_seg += 1
                continue
            if t_sec * self.fps < self.min_frames:
                short_seg += 1
                continue
            for text in descs:
                out.append(_BSEntry(
                    source="natural", motion_path=p, filename=fname,
                    seg_start_sec=0.0, seg_end_sec=t_sec, text=text,
                ))
        log.info(
            "Pool 'natural' (uniform): %d entries (skipped no_motion=%d mirror=%d no_text=%d "
            "long>%.1fs=%d short=%d bad_read=%d)",
            len(out), no_motion, mirror, no_text, self.max_segment_sec, long_seg, short_seg, bad_read,
        )

    def __getitem__(self, index: int) -> dict:
        source = self.SOURCES[index % len(self.SOURCES)]
        pool = self._pools[source]
        entry = pool[(index // len(self.SOURCES)) % len(pool)]

        sf, ef = self._resolve_segment(entry)
        with np.load(entry.motion_path, mmap_mode="r") as d:
            n_total = int(d["local_rot_mats"].shape[0])
            sf = max(0, min(sf, n_total))
            ef = max(sf, min(ef, n_total))
            ef = min(ef, sf + self.max_frames)
            lrot77 = np.asarray(d["local_rot_mats"][sf:ef]).astype(np.float32)
            root = np.asarray(d["root_positions"][sf:ef]).astype(np.float32)

        T = lrot77.shape[0]
        if T < self.min_frames or not np.isfinite(lrot77).all() or not np.isfinite(root).all():
            return self[(index + 1) % len(self)]

        with torch.no_grad():
            lrot30 = self.soma30.from_SOMASkeleton77(torch.from_numpy(lrot77))
            feats = self.tmr_rep(
                local_joint_rots=lrot30.unsqueeze(0),
                root_positions=torch.from_numpy(root).unsqueeze(0),
                to_normalize=self.tmr_normalize,
                to_canonicalize=True,
                lengths=torch.tensor([T]),
            )[0]
        if not torch.isfinite(feats).all():
            return self[(index + 1) % len(self)]

        return {
            "features": feats.float(),
            "length": int(T),
            "text": entry.text,
            "neutral_joints": torch.zeros(30, 3, dtype=torch.float32),
            "source": entry.source,
        }
