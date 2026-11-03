"""Attach existing native camera actions without changing draws or captions."""
from collections import OrderedDict
import json
import os
from pathlib import Path

import numpy as np
import torch

from motion_expert_phase3_edge_image_joint.data import Phase3Dataset as BaseDataset
from motion_expert_phase3_edge_joint.data import collate as base_collate


class Phase3Dataset(BaseDataset):
    def __init__(self, cache, camera_cache=None):
        super().__init__(cache)
        self.camera_root = Path(camera_cache or os.environ["PHASE3_CAMERA_PREDICTION_CACHE"])
        self.camera_index = json.loads((self.camera_root / "index.json").read_text())
        if (not self.camera_index["full_training_index"]
                or self.camera_index["source"]["baseline_fingerprint"] != self.index["fingerprint"]
                or self.camera_index["source"]["rows"] != len(self)):
            raise ValueError("Camera cache must cover exactly the unchanged full image-joint cohort")
        self.camera_arrays = OrderedDict()

    def __setstate__(self, state):
        self.__dict__.update(state)
        from motion_expert_phase3_edge_image_joint.train import configure_local_paths
        from .policy import configure
        configure_local_paths()
        configure()

    def __getitem__(self, key):
        result = super().__getitem__(key)
        index = key[1] if isinstance(key, tuple) else int(key)
        row = self.rows[index]
        record = self.camera_index["records"][row["uuid"]]
        import motion_expert_phase3_edge_joint.data as base
        if base.identity(record["source"]) != record["source_identity"]:
            raise RuntimeError("Validated camera source identity changed")
        array = row["array"]
        if array not in self.camera_arrays:
            self.camera_arrays[array] = np.load(self.camera_root / record["file"], mmap_mode="r", allow_pickle=False)
        self.camera_arrays.move_to_end(array)
        while len(self.camera_arrays) > 16:
            self.camera_arrays.popitem(last=False)
        action = np.zeros((96, 64), dtype=np.float32)
        action[:, :9] = self.camera_arrays[array][row["row"]]
        if not np.isfinite(action).all():
            raise ValueError("Nonfinite camera input")
        return {**result, "camera_action": torch.from_numpy(action)}


def collate(rows):
    return {**base_collate(rows), "camera_action": torch.stack([r["camera_action"] for r in rows])}
