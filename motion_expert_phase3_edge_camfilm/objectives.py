"""Call the original specialist loss formulas, with more motion conditions."""
from types import SimpleNamespace

from motion_expert_phase3_edge.objectives import noise_batch as base_noise, specialist_loss as base_loss
from .settings import effective_caption


def target_modes(modes):
    return ["motimg2video" if mode == "motimg2video" else "video2motion" for mode in modes]


def noise_batch(batch, device):
    inputs, target = base_noise({**batch, "modes": target_modes(batch["modes"])}, device)
    inputs["modes"] = batch["modes"]
    inputs["captions"] = [effective_caption(mode, caption) for mode, caption in zip(batch["modes"], batch["captions"], strict=True)]
    inputs["camera_action"] = batch["camera_action"].to(device)
    return inputs, target


def specialist_loss(output, target, inputs, mean, std):
    masks = [None] * len(inputs["modes"])
    for item, sample in enumerate(output["vision_sample_indices"]):
        masks[sample] = output["packed"].vision.condition_mask[item]
    aligned = {**output, "packed": SimpleNamespace(vision=SimpleNamespace(condition_mask=masks))}
    return base_loss(aligned, target, {**inputs, "modes": target_modes(inputs["modes"])}, mean, std)
