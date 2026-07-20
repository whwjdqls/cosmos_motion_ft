"""Check TI2M has 97 valid aligned frames inside the shared T=200 Phase-2 batch."""
from __future__ import annotations

from nymeria_joint_dataset import NymeriaJointDataset, collate_joint


def main() -> None:
    T = 200
    ti2m_frames = 97
    dataset = NymeriaJointDataset(
        split="train",
        num_frames=T,
        aligned_num_frames=ti2m_frames,
        task_weights={"textimg2motion": 1.0},
        bones_text2motion_frac=0.0,
        cfg_dropout=0.0,
        prefer_latents=False,
        reasoner_image_for_textimg=True,
        reasoner_image_size=256,
        train=False,
        max_samples=32,
        seed=0,
    )
    assert len(dataset._index) == 32
    assert len(dataset._t2m_index) == 32

    rows = []
    for index in range(4):
        row = dataset[index]
        aligned = dataset._index[index]
        valid = int((~row["motion_pad_mask"]).sum())
        assert row["mode"] == "textimg2motion"
        assert row["reasoner_image"] is not None
        assert tuple(row["reasoner_image"].shape) == (3, 256, 256)
        assert row["video_frames"] is None and row["video_latents"] is None
        assert row["camera_action"] is None
        assert tuple(row["motion"].shape) == (T, 283)
        assert valid == ti2m_frames
        rows.append(row)
        print(
            f"[ti2m-window] {aligned['uuid']}@{aligned['s']} "
            f"aligned_motion={valid} image={tuple(row['reasoner_image'].shape)}"
        )

    batch = collate_joint(rows)
    assert tuple(batch["motion"].shape) == (4, T, 283)
    assert len(batch["reasoner_image"]) == 4
    assert all(tuple(image.shape) == (3, 256, 256) for image in batch["reasoner_image"])
    print("TI2M 97-valid-frame aligned window padded to T=200 PASS")


if __name__ == "__main__":
    main()
