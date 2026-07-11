"""Native-compatible Cosmos camera training helpers.

This package is intentionally outside ``motion_expert_joint_attention``.  It
keeps Phase-1 generator LoRA training close to NVIDIA's native Cosmos trainer
while allowing cached Nymeria Wan-VAE latents to bypass repeated video encoding.
"""

