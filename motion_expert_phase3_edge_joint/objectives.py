"""Specialist losses and native noise sampling, not new phase-3 loss formulas."""
from __future__ import annotations

import torch
from losses import t2m_reconstruction_loss
from cosmos_framework.model.generator.algorithm.loss.flow_matching import compute_flow_matching_loss
from cosmos_framework.model.generator.diffusion.rectified_flow import RectifiedFlow, TrainTimeSampler
from motion_expert_phase3_edge_joint.settings import RECIPE


def noise_batch(batch, device):
    motion = batch["motion"].to(device)
    video = batch["video"].to(device).float()
    modes = batch["modes"]
    count = len(modes)
    shifts = torch.full((count,), 3.0)
    ms = TrainTimeSampler("logitnormal")(count, shifts=shifts).to(device)
    vs = TrainTimeSampler("waver")(count, shifts=shifts).to(device)
    if set(modes) - set(RECIPE["tasks"]):
        raise ValueError("Unknown Phase-3 task")
    mt = torch.tensor([m != "motimg2video" for m in modes], device=device)
    vt = torch.tensor([m != "video2motion" for m in modes], device=device)
    motion_noise, video_noise = torch.randn_like(motion), torch.randn_like(video)
    ms = ms * mt
    vs = vs * vt
    x_motion = torch.lerp(motion, motion_noise, ms[:, None, None])
    x_video = torch.lerp(video, video_noise, vs[:, None, None, None, None])
    x_video[:, :, 0] = video[:, :, 0]
    return {
        "motion": x_motion, "video": x_video, "neutral": batch["neutral"].to(device),
        "motion_sigma": ms, "video_sigma": vs, "captions": batch["captions"], "modes": modes,
    }, {"motion": motion, "velocity": video_noise - video}


def specialist_loss(output, target, inputs, mean, std):
    modes = inputs["modes"]
    batch = len(modes)
    device = mean.device
    mi = [i for i, mode in enumerate(modes) if mode in {"video2motion", "joint"}]
    vi = [i for i, mode in enumerate(modes) if mode in {"motimg2video", "joint"}]
    total = output["anchor"]
    metrics = {"video": total.detach().new_zeros(()), "motion": total.detach().new_zeros(())}
    if mi:
        pred = output["motion"][mi]
        valid = torch.ones(pred.shape[:2], device=device, dtype=torch.bool)
        loss, terms = t2m_reconstruction_loss(
            pred, target["motion"][mi], valid, mean, std,
            **RECIPE["motion_weights"], joint_loss_mode="centered", fps=20.0,
            # Reuse the exact Phase-2 formula; V2M changes conditioning, not
            # the supervised motion representation or reconstruction objective.
            modes=["text2motion"] * len(mi), reduction="per_task", decoded_nonfinite="skip",
        )
        total = total + loss * len(mi) / batch
        metrics.update({"motion/" + key: value.detach() for key, value in terms.items()})
        metrics["motion"] = loss.detach()
    if vi:
        flow = RectifiedFlow(None, train_time_distribution="waver", train_time_weight_method="uniform", shift=3)
        masks = [output["packed"].vision.condition_mask[i] for i in vi]
        loss, _ = compute_flow_matching_loss(
            pred=[output["video"][i].float() for i in vi],
            target=[target["velocity"][i] for i in vi], condition_mask=masks,
            timesteps=inputs["video_sigma"][vi, None] * 1000.0,
            has_valid_tokens=True, rectified_flow=flow,
            tensor_kwargs_fp32={"device": device, "dtype": torch.float32},
            normalize_by_active=False,
        )
        total = total + 10.0 * loss * len(vi) / batch
        metrics["video"] = loss.detach()
    metrics["video_weighted"] = 10.0 * metrics["video"]
    metrics["motion_count"] = total.detach().new_tensor(len(mi))
    metrics["video_count"] = total.detach().new_tensor(len(vi))
    metrics["joint_count"] = total.detach().new_tensor(sum(m == "joint" for m in modes))
    metrics["total"] = total.detach()
    return total, metrics
