"""Output-codec adapters; source datasets retain their original UniEgo guards."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

import ablation_settings as ablation


@lru_cache(maxsize=8)
def _arrays(mean, std):
    m, s = np.load(mean).astype(np.float32), np.load(std).astype(np.float32)
    if not np.isfinite(m).all() or not np.isfinite(s).all() or (s <= 0).any():
        raise ValueError("Invalid feature statistics")
    return m, s


def feature_statistics():
    import config
    return _feature_statistics(ablation.CODEC, ablation.STATS_DIR,
                               str(config.MOTION_STATS_MEAN), str(config.MOTION_STATS_STD))


@lru_cache(maxsize=8)
def _feature_statistics(codec, stats_dir, source_mean, source_std):
    if codec == "uniego283":
        return _arrays(source_mean, source_std)
    ablation.stats_manifest()
    root = Path(stats_dir)
    result = _arrays(str(root/'mean.npy'), str(root/'std.npy'))
    if result[0].shape != (369,) or result[1].shape != (369,):
        raise ValueError("Kimodo statistics must have369 channels")
    return result


def convert_source_row(row):
    if ablation.CODEC == "uniego283":
        return row
    import config
    from kimodo_codec import cached_encode
    source_mean, source_std = _arrays(str(config.MOTION_STATS_MEAN), str(config.MOTION_STATS_STD))
    valid = ~row['motion_pad_mask']
    values = row['motion'][valid].float()
    physical = values * torch.from_numpy(source_std) + torch.from_numpy(source_mean)
    converted = cached_encode(physical, config.FPS)
    mean, std = feature_statistics()
    result = torch.zeros((len(valid),369), dtype=torch.float32)
    result[valid] = (converted-torch.from_numpy(mean))/torch.from_numpy(std)
    return {**row, 'motion': result}


def decode_joints(features):
    if features.shape[-1] == 369:
        from kimodo_codec import decode_joints as decode
    else:
        from decode_uniego_torch import decode_joints as decode
    return decode(features)


def head_rotation_from_features(features):
    if features.shape[-1] == 369:
        from kimodo_codec import head_rotation
        value = torch.as_tensor(features).float()
        return head_rotation(value[0] if value.ndim == 2 else value).cpu().numpy()
    from render_motion import head_rotation_from_features as original
    return original(features)
