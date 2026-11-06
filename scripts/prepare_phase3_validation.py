"""Pin existing local metric assets and the reviewed fixed71 case manifest."""
import argparse
import json
from pathlib import Path

from motion_expert_phase3_edge_joint.settings import atomic_json, sha256
from motion_expert_phase3_edge_joint.validation import local_path, read_configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cases", "tmr_checkpoint", "tmr_stats", "text_cache", "body_adapter", "vae",
                 "i3d_weights", "i3d_source", "i3d_preprocess_source", "dreamsim_cache"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from motion_expert_phase3_edge_joint.site_paths import storage_path
    storage_path(args.out, must_exist=False)
    if args.out.exists():
        raise ValueError("Use a new site-local configuration file")
    config = {key: str(local_path(value)) for key, value in vars(args).items() if key != "out"}
    import torch
    import lpips
    config["lpips_alexnet"] = str(local_path(Path(torch.hub.get_dir()) / "checkpoints/alexnet-owt-7be5be79.pth"))
    config["lpips_linear"] = str(local_path(Path(lpips.__file__).parent / "weights/v0.1/alex.pth"))
    fingerprints = {key: sha256(value) for key, value in config.items() if Path(value).is_file()}
    fingerprints["tmr_stats"] = {p.name: sha256(p) for p in sorted(Path(config["tmr_stats"]).iterdir()) if p.is_file()}
    cache = Path(config["dreamsim_cache"])
    weights = [p for p in sorted(cache.rglob("*")) if p.is_file()
               and (p.suffix in {".pth", ".safetensors"} or p.name.endswith(".pth.tar") or p.name == "adapter_config.json")]
    fingerprints["dreamsim_files"] = {str(p.relative_to(cache)): sha256(p) for p in weights}
    config.update(schema=1, sha256=fingerprints)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, config)
    _, rows, _ = read_configuration(args.out)
    for row in rows:
        if sha256(local_path(row["bundle"])) != row["bundle_sha256"]:
            raise ValueError(f"Bundle checksum differs: {row['id']}")
    print(json.dumps(dict(config=str(args.out), windows=len(rows), assets_verified=True)))


if __name__ == "__main__":
    main()
