"""Camera-only CFG for untouched HF Edge or a saved Phase-1 checkpoint.

The sampler and guidance implementation are reused. The first scale-1 sample
also verifies native-versus-hook final latents with identical restored RNG.
"""
import copy
from functools import wraps
import json
import os


def install_checked_camera_cfg():
    import torch
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
    from native_phase_training.camera_cfg_inference_config import install_camera_cfg
    native = OmniMoTModel.generate_samples_from_batch
    install_camera_cfg()
    guided = OmniMoTModel.generate_samples_from_batch
    # Both branches and the native acceptance call use the same cache policy.
    OmniMoTModel._can_reuse_inference_text_kv = lambda self, *a, **kw: False
    checked = False

    @wraps(guided)
    def compare(self, *args, **kwargs):
        nonlocal checked
        if checked or kwargs.get('guidance') != 1.0:
            return guided(self, *args, **kwargs)
        checked = True
        cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        a = native(self, *copy.deepcopy(args), **copy.deepcopy(kwargs))
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state_all(cuda)
        b = guided(self, *copy.deepcopy(args), **copy.deepcopy(kwargs))
        records = []
        for key in a:
            if a[key] is None:
                assert b[key] is None
                continue
            assert len(a[key]) == len(b[key])
            for x, y in zip(a[key], b[key], strict=True):
                equal = bool(torch.equal(x, y))
                records.append(dict(key=key, equal=equal, max_abs=float((x-y).abs().max())))
                assert equal, 'Camera guidance1 differs from native uncached inference'
        print('ZERO_CAMERA_NATIVE_PARITY '+json.dumps(records), flush=True)
        return b

    OmniMoTModel.generate_samples_from_batch = compare


def make_config():
    from native_phase_training.inference_config import make_config as make_native
    config = make_native()
    install_checked_camera_cfg()
    return config
