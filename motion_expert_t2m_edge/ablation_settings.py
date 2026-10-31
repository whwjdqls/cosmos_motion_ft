"""Opt-in output codecs/objectives; the default preserves old checkpoints."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

CODEC = os.environ.get("EDGE_PHASE2_FEATURE_CODEC", "uniego283")
OBJECTIVE = os.environ.get("EDGE_PHASE2_OBJECTIVE", "x0")
STATS_DIR = os.environ.get("PHASE2_KIMODO_STATS_DIR")


def validate():
    if CODEC not in {"uniego283", "kimodo369_v1"}:
        raise ValueError(f"unsupported Phase2 feature codec: {CODEC}")
    if OBJECTIVE not in {"x0", "velocity"}:
        raise ValueError(f"unsupported Phase2 objective: {OBJECTIVE}")


def motion_dim():
    validate()
    return 369 if CODEC == "kimodo369_v1" else 283


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats_manifest():
    if CODEC == "uniego283":
        return None
    if not STATS_DIR:
        raise RuntimeError("Kimodo369 requires prepared PHASE2_KIMODO_STATS_DIR")
    root = Path(STATS_DIR)
    data = json.loads((root / "summary.json").read_text())
    if data.get("codec") != "kimodo369_v1" or data.get("split") != "train":
        raise RuntimeError("Kimodo statistics must be a complete train-split codec manifest")
    if not data.get("complete") or data.get("limited"):
        raise RuntimeError("Limited/test/smoke statistics cannot authorize production")
    for key in ("mean", "std"):
        if sha(root / (key + ".npy")) != data[key + "_sha256"]:
            raise RuntimeError(f"Kimodo {key} hash mismatch")
    if sha(Path(__file__).with_name("kimodo_codec.py")) != data["codec_sha256"]:
        raise RuntimeError("Kimodo codec changed after statistics preparation")
    from kimodo_codec import dependency_identity
    if dependency_identity() != data.get('dependencies'):
        raise RuntimeError('Kimodo dependencies changed after statistics preparation')
    import config
    if data.get('population') != config.EXPECTED_STATS_POPULATION:
        raise RuntimeError('Kimodo statistics use a different source training population')
    if (data.get('actual_frames') != config.EXPECTED_STATS_POPULATION['frames']
            or data.get('actual_windows') != config.EXPECTED_STATS_POPULATION['stats_windows']):
        raise RuntimeError('Kimodo statistics did not finish the required population')
    for name, path in dict(mean=config.MOTION_STATS_MEAN, std=config.MOTION_STATS_STD,
                           manifest=config.NYMERIA_MANIFEST, split=config.NYMERIA_SPLIT_FILE,
                           calibration=config.FLOOR_CALIBRATION_JSON).items():
        if sha(path) != data.get('source_sha256', {}).get(name):
            raise RuntimeError(f'Kimodo source identity mismatch: {name}')
    return data


def contract():
    validate()
    if CODEC == "uniego283" and OBJECTIVE == "x0":
        return None
    data = stats_manifest()
    result = dict(version=1, feature_codec=CODEC, objective=OBJECTIVE)
    if data:
        result.update(stats_dir=str(Path(STATS_DIR)), mean_sha256=data["mean_sha256"],
                      std_sha256=data["std_sha256"], summary_sha256=sha(Path(STATS_DIR)/"summary.json"),
                      codec_sha256=data["codec_sha256"])
    return result


def configure_checkpoint(saved):
    """Resolve codec before heads construction; explicit conflicting flags fail."""
    global CODEC, OBJECTIVE, STATS_DIR
    saved = saved or dict(feature_codec="uniego283", objective="x0")
    if 'version' in saved and saved['version'] != 1:
        raise RuntimeError('Unsupported Phase2 ablation contract version')
    for env, name, default in [("EDGE_PHASE2_FEATURE_CODEC", "CODEC", "uniego283"),
                               ("EDGE_PHASE2_OBJECTIVE", "OBJECTIVE", "x0"),
                               ("PHASE2_KIMODO_STATS_DIR", "STATS_DIR", None)]:
        key = {"CODEC":"feature_codec", "OBJECTIVE":"objective", "STATS_DIR":"stats_dir"}[name]
        if name == 'STATS_DIR' and CODEC != 'kimodo369_v1':
            continue
        value = saved.get(key, default)
        if env in os.environ and globals()[name] != value:
            raise RuntimeError(f"Explicit {env} conflicts with checkpoint")
        globals()[name] = value
    validate()


def conditioning_partition(ti2m_fraction=.5, bones_within_t2m=.75,
                           caption_dropout=.1, ti2m_caption_dropout=.3):
    """Expected probabilities, including the historical double BONES dropout."""
    for x in (ti2m_fraction, bones_within_t2m, caption_dropout, ti2m_caption_dropout):
        if not 0 <= x <= 1:
            raise ValueError("mixture probabilities must be in [0,1]")
    bones = (1-ti2m_fraction)*bones_within_t2m
    ny = (1-ti2m_fraction)*(1-bones_within_t2m)
    keep = 1-caption_dropout
    return dict(bones_text=bones*keep**2, bones_null=bones*(1-keep**2),
                nymeria_text=ny*keep, nymeria_null=ny*caption_dropout,
                nymeria_image_text=ti2m_fraction*(1-ti2m_caption_dropout),
                nymeria_image_only=ti2m_fraction*ti2m_caption_dropout)
