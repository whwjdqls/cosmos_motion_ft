"""Focused CPU contract checks for native-schedule x0 motion training/sampling."""
from __future__ import annotations

import torch

import bs_native_flow
import flow
from nymeria_joint_dataset import uses_native_motion_index


def main() -> None:
    shift = 3.0
    n_train = 1000

    generator_actual = torch.Generator().manual_seed(123)
    generator_expected = torch.Generator().manual_seed(123)
    actual = flow.sample_sigma_native_logitnormal(
        64,
        "cpu",
        shift=shift,
        generator=generator_actual,
    )
    raw = torch.sigmoid(torch.randn((64,), generator=generator_expected))
    expected = flow.shift_sigma(1.0 - raw, shift)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert bool(((actual > 0) & (actual < 1)).all())

    sigmas, timesteps = flow.native_inference_schedule(
        50,
        shift=shift,
        num_train_timesteps=n_train,
    )
    poc_sigmas, poc_timesteps = bs_native_flow.inference_schedule(
        50,
        shift=shift,
        num_train_timesteps=n_train,
    )
    torch.testing.assert_close(sigmas, poc_sigmas, rtol=0, atol=0)
    torch.testing.assert_close(timesteps, poc_timesteps, rtol=0, atol=0)

    from cosmos_framework.model.vfm.diffusion.samplers.fm_solvers_unipc import (
        FlowUniPCMultistepScheduler,
    )

    official = FlowUniPCMultistepScheduler(
        num_train_timesteps=n_train,
        shift=1.0,
        use_dynamic_shifting=False,
    )
    official.set_timesteps(50, device="cpu", shift=shift)
    torch.testing.assert_close(sigmas, official.sigmas, rtol=0, atol=0)
    torch.testing.assert_close(timesteps, official.timesteps, rtol=0, atol=0)

    target = torch.linspace(-1.0, 1.0, 2 * 7 * 5).reshape(2, 7, 5)

    def perfect_x0(_x: torch.Tensor, _sigma: torch.Tensor) -> torch.Tensor:
        return target

    euler = flow.sample_x0_native(
        perfect_x0,
        T=7,
        motion_dim=5,
        steps=12,
        batch=2,
        device="cpu",
        generator=torch.Generator().manual_seed(7),
    )
    torch.testing.assert_close(euler, target, rtol=1e-5, atol=1e-5)

    unipc = flow.sample_x0_native_unipc(
        perfect_x0,
        T=7,
        motion_dim=5,
        steps=12,
        batch=2,
        device="cpu",
        generator=torch.Generator().manual_seed(7),
    )
    torch.testing.assert_close(unipc, target, rtol=1e-4, atol=1e-4)

    fixed_noise = torch.randn(2, 7, 5, generator=torch.Generator().manual_seed(91))
    unipc_fixed_a = flow.sample_x0_native_unipc(
        perfect_x0,
        T=7,
        motion_dim=5,
        steps=12,
        batch=2,
        device="cpu",
        generator=torch.Generator().manual_seed(1),
        initial_noise=fixed_noise,
    )
    unipc_fixed_b = flow.sample_x0_native_unipc(
        perfect_x0,
        T=7,
        motion_dim=5,
        steps=12,
        batch=2,
        device="cpu",
        generator=torch.Generator().manual_seed(2),
        initial_noise=fixed_noise,
    )
    torch.testing.assert_close(unipc_fixed_a, unipc_fixed_b, rtol=0, atol=0)
    try:
        flow.sample_x0_native_unipc(
            perfect_x0,
            T=7,
            motion_dim=5,
            steps=2,
            batch=2,
            device="cpu",
            initial_noise=torch.zeros(1, 7, 5),
        )
    except ValueError as error:
        assert "initial_noise" in str(error)
    else:
        raise AssertionError("native UniPC must reject an initial-noise shape mismatch")

    assert flow.motion_sampler("x0", "legacy", "euler") is flow.sample_x0
    assert flow.motion_sampler("x0", "native", "euler") is flow.sample_x0_native
    assert flow.motion_sampler("x0", "native", "unipc") is flow.sample_x0_native_unipc
    assert flow.motion_sampler("x0", "native") is flow.sample_x0_native_unipc
    assert uses_native_motion_index("text2motion", needs_video=False, needs_camera=False)
    assert not uses_native_motion_index("textimg2motion", needs_video=False, needs_camera=False)
    assert not uses_native_motion_index("textimg2motion", needs_video=True, needs_camera=False)
    assert not uses_native_motion_index("video2motion", needs_video=True, needs_camera=False)

    print(
        "native motion flow PASS: training sigma, POC/native ladder, official UniPC ladder, "
        "perfect-x0 Euler/UniPC, fixed initial noise, and T2M/TI2M index routing"
    )


if __name__ == "__main__":
    main()
