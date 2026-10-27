"""Derive correct world-root and CPF-relative metadata without changing sources."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from motion_expert_joint_attention.camera_head_recanonicalization import decode_uniego
from nymeria_baselines.body_shape import BODY_JOINTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    manifest_bytes = (args.cache / "index.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    records = {row["array"]: row["uuid"] for row in manifest["rows"]}
    output = args.cache / "egoallo_metadata_v1"
    output.mkdir(exist_ok=True)
    # y-up world coordinates = Y_FROM_Z @ z-up world coordinates.
    y_from_z = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    reports = []
    for key, uuid in records.items():
        source = args.cache / "records"
        motion = np.load(source / f"{key}.motion.npy", mmap_mode="r")
        camera = np.load(source / f"{key}.pose_zup.npy", mmap_mode="r")
        assert len(motion) == len(camera)
        decoded = decode_uniego(motion)
        root_rotation = y_from_z.T @ decoded.world_rotations[:, 0]
        xyzw = Rotation.from_matrix(root_rotation).as_quat()
        world_positions = np.einsum(
            "ij,tkj->tki", y_from_z.T, decoded.world_positions[:, BODY_JOINTS]
        )
        root = np.concatenate((xyzw[:, [3, 0, 1, 2]], world_positions[:, 0]), axis=-1)
        relative = world_positions[:, 1:] - camera[:, None, :3, 3]
        joints_wrt_cpf = np.einsum("tji,tkj->tki", camera[:, :3, :3], relative)
        head_distance = np.linalg.norm(
            world_positions[:, 6] - camera[:, :3, 3], axis=-1
        )
        for suffix, value in (("world_root_zup", root), ("joints_wrt_cpf", joints_wrt_cpf)):
            if not np.isfinite(value).all():
                raise ValueError(f"nonfinite {suffix}: {uuid}")
            destination = output / f"{key}.{suffix}.npy"
            temporary = output / f"{key}.{suffix}.tmp.npy"
            np.save(temporary, value.astype(np.float32))
            temporary.replace(destination)
        reports.append({
            "uuid": uuid, "array": key, "frames": len(motion),
            "head_camera_mean_m": float(head_distance.mean()),
            "head_camera_max_m": float(head_distance.max()),
        })
        if len(reports) % 50 == 0:
            print(f"prepared {len(reports)}/{len(records)} recordings", flush=True)
    payload = {
        "schema": 1,
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "world_root": "raw z-up world pose; subtract per-window floor from Z in loader",
        "joints_wrt_cpf": "body joints excluding root, expressed in the measured RGB camera frame",
        "records": reports,
    }
    temporary = output / "index.tmp.json"
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output / "index.json")
    print(json.dumps({"status": "complete", "recordings": len(reports), "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
