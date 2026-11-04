"""One or two GPUs, preserving the original two 16-sample loss groups."""


def topology(world, microbatch):
    if world not in (1, 2) or microbatch <= 0 or 32 % (world * microbatch):
        raise ValueError("Use one or two GPUs with microbatch * GPUs dividing global batch32")
    return dict(world_size=world, microbatch=microbatch, global_batch=32,
                accumulation=32 // (world * microbatch))


def require_acceptance(receipt, fingerprint, world, microbatch):
    topology(world, microbatch)
    checks = {
        'camera_predictions_all_three_tasks_no_clean_camera_tokens',
        'zero_camera_loss_and_no_camera_readout_gradient',
        'joint_caption_invariant_camera_motion_video_outputs',
    }
    if (not receipt.get('passed') or receipt.get('fingerprint') != fingerprint
            or receipt.get('world_size') != world
            or receipt.get('microbatch') != microbatch
            or not checks.issubset(receipt.get('checks', []))):
        raise RuntimeError("Missing or stale camera preflight for this exact GPU count and microbatch")
