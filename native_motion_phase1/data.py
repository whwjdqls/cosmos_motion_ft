"""The immutable Phase-3 windows, represented through native action tokens.

The native VFM sampler requires a vision stream. Text-only samples therefore
use a fixed all-zero, observed latent, independent of the example's RGB.
No target video is loaded for text-only samples. A neutral-skeleton prefix
is always observed; the following 97 tokens are the unchanged motion features.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from cosmos_framework.data.generator.augmentors.text_tokenizer import TextTokenizerTransform
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.data.generator.action.utils.action_processing import (
    ActionProcessingRecord, make_batched_action_processing_fields,
)

TASKS = ("text2motion", "textimg2motion", "video2motion", "motion2video")
DIM, FRAMES, DOMAIN = 283, 97, 31
HEAD_DIM = 288  # grouped BF16 GEMM requires a multiple-of-eight channel stride


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def make_sample(motion, neutral, latent, task, caption=""):
    if task not in TASKS:
        raise ValueError(task)
    motion, neutral = torch.as_tensor(motion).float(), torch.as_tensor(neutral).float()
    if motion.shape != (97, 283) or neutral.shape != (30, 3):
        raise ValueError("Expected Phase-3 motion [97,283] and centered neutral [30,3]")
    if not torch.isfinite(motion).all() or not torch.isfinite(neutral).all():
        raise ValueError("Non-finite motion/shape")
    action = torch.zeros(98, HEAD_DIM)
    action[0, :90] = neutral.flatten()
    action[1:, :DIM] = motion
    if task == "text2motion":
        latent = torch.zeros(48, 1, 16, 16, dtype=torch.float16)
    else:
        latent = torch.as_tensor(latent)
        if latent.shape != (48, 25, 16, 16) or not torch.isfinite(latent).all():
            raise ValueError("Expected finite cached Wan latent [48,25,16,16]")
        if task == "textimg2motion":
            latent = latent[:, :1].clone()  # causal first latent uses only RGB frame 0
    vision_frames = latent.shape[1]
    plan = SequencePlan(
        has_text=True, has_vision=True, has_action=True,
        condition_frame_indexes_vision=([0] if task == "motion2video" else list(range(vision_frames))),
        condition_frame_indexes_action=(list(range(98)) if task == "motion2video" else [0]),
        action_start_frame_offset=-1,  # motion token 1 aligns with RGB frame 0
    )
    return dict(video=torch.zeros(3, 1 + (vision_frames - 1) * 4, 1, 1, dtype=torch.uint8),
                video_latents=latent, image_size=torch.tensor([256., 256., 256., 256.]),
                conditioning_fps=torch.tensor(20), action=action,
                raw_action_dim=torch.tensor(DIM), domain_id=torch.tensor(DOMAIN),
                sequence_plan=plan, mode=task,
                ai_caption=caption if task in TASKS[:2] else "")


def inference_batch(motion, neutral, latent, task, caption=""):
    sample = make_sample(motion, neutral, latent, task, caption)
    return dict(video=[sample["video"][None]], video_latents=[sample["video_latents"]],
                image_size=sample["image_size"][None], conditioning_fps=torch.tensor([20]),
                action=[sample["action"]], domain_id=[sample["domain_id"]],
                ai_caption=[sample["ai_caption"]], sequence_plan=[sample["sequence_plan"]],
                **make_batched_action_processing_fields(
                    ActionProcessingRecord(raw_action_dim=DIM, action_normalizer=None), 1))


class MotionDataset(Dataset):
    def __init__(self, cache, task, tokenizer_config, mean_path, std_path):
        if task not in TASKS:
            raise ValueError(task)
        self.root, self.task = Path(cache).resolve(), task
        if not str(self.root).startswith("/mnt/local/"):
            raise ValueError("Training cache must be node-local")
        index = json.loads((self.root / "index.json").read_text())
        if not index["full_training_index"] or index["source"]["limit"]:
            raise ValueError("A full validated Phase-3 cache is required")
        for key, path in (("mean", mean_path), ("std", std_path)):
            if sha256(path) != index["source"][key]:
                raise ValueError(f"Motion normalization changed: {key}")
        self.rows = index["rows"]
        self.arrays = OrderedDict()
        self.tokenize = TextTokenizerTransform(
            input_keys=["ai_caption"], output_keys=["text_token_ids"],
            args=dict(tokenizer_config=tokenizer_config, cfg_dropout_rate=.1 if task in TASKS[:2] else 0.))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        key = row["array"]
        if key not in self.arrays:
            self.arrays[key] = tuple(np.load(self.root / (key + suffix), mmap_mode="r", allow_pickle=False)
                                     for suffix in (".motion.npy", ".neutral.npy"))
        self.arrays.move_to_end(key)
        while len(self.arrays) > 16:
            self.arrays.popitem(last=False)
        motion, neutral = self.arrays[key]
        latent = None
        if self.task != "text2motion":
            path = Path(row["latent"])
            if not str(path.resolve()).startswith("/mnt/local/"):
                raise ValueError("Video must be node-local")
            st = path.stat()
            if {"size": st.st_size, "mtime_ns": st.st_mtime_ns} != row["latent_identity"]:
                raise ValueError(f"Validated latent changed: {path}")
            with np.load(path, allow_pickle=False) as archive:
                latent = archive["latents"]
        return self.tokenize(make_sample(motion[row["row"]].copy(), neutral.copy(),
                                         latent, self.task, row["caption"]))
