"""CPU contracts for batched Wan-latent preprocessing."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from motion_expert_joint_attention.precompute_latents import encode_window, encode_windows


class _FakePipeline:
    def __call__(self, sample, _resolution):
        video = sample["video"]
        height, width = video.shape[-2:]
        return {
            "video": video,
            "image_size": torch.tensor(
                [height, width, height, width], dtype=torch.float32
            ),
        }


class _FakeVAE:
    spatial_compression_factor = 2

    def __init__(self):
        self.batch_sizes: list[int] = []

    def encode(self, videos):
        self.batch_sizes.append(int(videos.shape[0]))
        return videos[:, :1, [0, 4], ::2, ::2]


class PrecomputeBatchingTest(unittest.TestCase):
    def setUp(self):
        base = np.arange(5 * 4 * 4 * 3, dtype=np.uint8).reshape(5, 4, 4, 3)
        self.frames = [base, np.flip(base, axis=0).copy(), np.flip(base, axis=1).copy()]

    def test_batch_matches_single_window_wrapper(self):
        pipe = _FakePipeline()
        batch_vae = _FakeVAE()
        batched = encode_windows(batch_vae, pipe, "256", self.frames, "cpu")
        self.assertEqual(batch_vae.batch_sizes, [3])

        singles = []
        single_vae = _FakeVAE()
        for frames in self.frames:
            singles.append(encode_window(single_vae, pipe, "256", frames, "cpu"))
        self.assertEqual(single_vae.batch_sizes, [1, 1, 1])

        for (batch_latent, batch_size), (single_latent, single_size) in zip(
            batched, singles, strict=True
        ):
            np.testing.assert_array_equal(batch_latent, single_latent)
            np.testing.assert_array_equal(batch_size, single_size)


if __name__ == "__main__":
    unittest.main()
