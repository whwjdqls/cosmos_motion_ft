"""Rectified-flow noising + both objectives + samplers (cosmos env).

Self-contained: NO Cosmos imports. Operates on dense [B,T,D] / [B,N,D] tensors;
mask / padding is handled entirely by the caller. Two objective families,
selected by train.py / sample.py via --objective:

7-TASK EXTENSION (DESIGN_7TASK.md section 3): the velocity path is generalized to
ALL modalities (video latents / camera actions / motion) via a single per-token
``condition_mask`` (True == CLEAN, matching ``task_plan.ModalityResolved``):
``add_noise_velocity_masked`` applies ``sigma_eff = sigma*(1-condition_mask)`` so
clean condition frames pass through untouched (no timestep bias, no loss) while
noised frames are supervised; ``flow_loss_masked`` + the named wrappers
(``vision_flow_loss`` all channels / ``camera_flow_loss`` channels [:9] x~10 /
``motion_flow_loss`` 283-d) and ``compute_gen_flow_loss`` dispatch the per-modality
loss the trainer sums; ``sample_velocity_masked`` is the matching inference sampler
that pins the conditioning tokens clean while integrating the noised target. The
existing motion-only ``add_noise_velocity`` / ``sample_velocity`` /
``add_noise`` / ``sample_x0`` are UNCHANGED so the text2motion path is untouched.

(1) velocity (DEFAULT, Cosmos-native rectified flow — matches
    train_motion_ft.forward_loss):
        forward   x_t = (1-t)*x0 + t*noise,   t ~ U(0,1)
        target    v   = noise - x0
        net regresses v_hat; sampling = Euler ODE t:1->0, x -= dt*v_hat.

(2) x0 (PoC-compatible, copied verbatim from motion_expert/flow.py; the proven
    working recipe of motion_expert/bs_train.py):
        logit-normal sigma; x_sigma = sigma*eps + (1-sigma)*x0;
        net predicts x0; DDIM-in-sigma sampler.
    The masked 7-task twins exist for this objective too: ``add_noise_x0_masked``
    (per-token condition_mask gate, TARGET = x0 itself; same tuple structure as
    ``add_noise_velocity_masked`` so train.step_loss switches symmetrically) and
    ``sample_x0_masked`` (DDIM-in-sigma while pinning the conditioning tokens
    clean, the inference twin of ``add_noise_x0_masked``).

Both samplers are agnostic to the model's call signature: they take a
`predict(x, t_or_sigma_b) -> target_hat` closure (built in sample.py around
joint_motion_model) plus an optional `predict_null` closure for CFG. This
generalizes the PoC sampler, whose positional H_R signature does NOT apply to
the joint model.

No torch.compile.
"""
from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# (1) Cosmos velocity path (DEFAULT)
# ---------------------------------------------------------------------------
def add_noise_velocity(x0: torch.Tensor):
    """Rectified-flow forward noising with velocity target.

    x0 [B,T,D] -> (x_t [B,T,D], t [B], target_v [B,T,D]) where
        t        ~ U(0,1)
        x_t      = (1-t)*x0 + t*noise
        target_v = noise - x0
    Matches train_motion_ft.forward_loss exactly (t=1 pure noise, t=0 clean).
    """
    B = x0.shape[0]
    noise = torch.randn_like(x0)
    t = torch.rand(B, device=x0.device, dtype=x0.dtype)
    tb = t.view(-1, *([1] * (x0.dim() - 1)))
    x_t = (1.0 - tb) * x0 + tb * noise
    target_v = noise - x0
    return x_t, t, target_v


# ---------------------------------------------------------------------------
# (1b) Masked velocity noising for the 7-task joint model (all modalities).
#
# DESIGN_7TASK.md section 3: per modality apply a single per-token condition_mask
# (T-first; True == CLEAN, False == NOISED, matching task_plan.ModalityResolved)
# to drive the rectified-flow forward process:
#       sigma_eff = sigma * (1 - condition_mask)          # 0 on clean tokens
#       x_t       = (1 - sigma_eff) * x0 + sigma_eff * eps
#       target_v  = eps - x0
# Clean tokens (condition_mask=True) therefore pass through UNTOUCHED (x_t == x0,
# sigma_eff == 0) and get NO timestep bias; they are excluded from the loss. This
# is the identical objective used for motion above (add_noise_velocity), only with
# the noise gated per-token so condition frames stay clean.
#
# `condition_mask` is True == CLEAN (the task_plan contract). The internal
# multiplier is `(~condition_mask)` so noise is applied only on NOISED tokens.
# `t` is sampled per-sample (one scalar per batch row) exactly like the motion
# path; the per-token gate selects which tokens that t actually noises.
# ---------------------------------------------------------------------------
def add_noise_velocity_masked(
    x0: torch.Tensor,
    condition_mask: torch.Tensor,
    t: torch.Tensor | None = None,
):
    """Masked rectified-flow forward noising with velocity target.

    Args:
        x0 [B, N, D]      : clean modality tokens (video latents / camera actions /
                            motion features), N = per-modality token count.
        condition_mask    : [B, N] bool, True == CLEAN (do NOT noise, no loss),
                            False == NOISED (supervised). Matches task_plan's
                            per-token condition_mask. Broadcast over channels D.
        t [B]             : optional pre-sampled flow time (shared with sibling
                            modalities of the same sample so a multi-target task
                            uses ONE t per sample). If None, sampled U(0,1).

    Returns:
        x_t [B, N, D]      : x_t = (1-sigma_eff)*x0 + sigma_eff*eps; clean tokens
                             keep x0 exactly (sigma_eff = 0 there).
        t [B]              : the per-sample flow time used.
        target_v [B, N, D] : eps - x0 (the regression target; loss masks to NOISED).
        noised_mask [B, N] : ~condition_mask (True == this token was noised /
                             supervised); convenience for the loss reductions.
    """
    B = x0.shape[0]
    if t is None:
        t = torch.rand(B, device=x0.device, dtype=x0.dtype)
    else:
        t = t.to(device=x0.device, dtype=x0.dtype)
    noise = torch.randn_like(x0)

    noised_mask = ~condition_mask.bool()                       # [B, N] True == noise it
    tb = t.view(-1, *([1] * (x0.dim() - 1)))                   # [B,1,1]
    gate = noised_mask.to(x0.dtype)
    while gate.dim() < x0.dim():
        gate = gate.unsqueeze(-1)                              # [B,N,1] broadcast over D
    sigma_eff = tb * gate                                      # 0 on clean tokens

    x_t = (1.0 - sigma_eff) * x0 + sigma_eff * noise
    target_v = noise - x0
    return x_t, t, target_v, noised_mask


def flow_loss_masked(
    pred_v: torch.Tensor,
    target_v: torch.Tensor,
    noised_mask: torch.Tensor,
    loss_channels: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Masked flow-matching MSE for ANY modality (vision / camera / motion-feat).

    Mirrors train.masked_mse but with the optional channel slice the camera loss
    needs (supervise only channels [:9], ignoring the zero-pad to max_action_dim).

    Args:
        pred_v / target_v [B, N, D] : predicted vs target velocity (eps - x0).
        noised_mask [B, N]          : True == token is NOISED + supervised (the
                                       ~condition_mask returned by add_noise_velocity_masked).
        loss_channels (lo, hi)      : restrict the channel dim to [lo:hi] before the
                                       MSE (camera -> (0, CAMERA_RAW_DIM)). None == all.

    Returns a scalar mean over supervised (noised) tokens x kept channels. Returns
    0 when no token is noised (the all-clean / condition-only fast path), so a
    condition-only modality contributes exactly zero loss by construction.
    """
    if loss_channels is not None:
        lo, hi = loss_channels
        pred_v = pred_v[..., lo:hi]
        target_v = target_v[..., lo:hi]
    m = noised_mask.to(pred_v.dtype)
    while m.dim() < pred_v.dim():
        m = m.unsqueeze(-1)                                    # [B,N,1] broadcast over channels
    se = ((pred_v - target_v) ** 2) * m
    denom = m.expand_as(pred_v).sum().clamp(min=1)
    return se.sum() / denom


@torch.no_grad()
def sample_velocity(
    predict,
    T: int,
    motion_dim: int,
    steps: int = 50,
    guidance: float = 1.0,
    predict_null=None,
    batch: int = 1,
    device="cuda",
    dtype=torch.float32,
    generator=None,
):
    """Euler ODE sampler for the velocity objective (t: 1 -> 0).

    predict(x [B,T,D], t_b [B]) -> v_hat [B,T,D]   (conditional velocity)
    predict_null(...)           -> v_u   [B,T,D]   (unconditional, for CFG)

    CFG:  v = v_u + guidance * (v_cond - v_u).
    At each step x <- x - dt * v_hat. Returns clean x0 [B,T,motion_dim].
    """
    x = torch.randn(batch, T, motion_dim, device=device, dtype=dtype, generator=generator)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
    for i in range(steps):
        t_b = ts[i].expand(batch)
        dt = (ts[i] - ts[i + 1])  # > 0; we step x -= dt * v
        v_hat = predict(x, t_b).to(dtype)
        if guidance != 1.0 and predict_null is not None:
            v_u = predict_null(x, t_b).to(dtype)
            v_hat = v_u + guidance * (v_hat - v_u)
        x = x - dt * v_hat
    return x


# ---------------------------------------------------------------------------
# (2) PoC x0 path (copied from motion_expert/flow.py; sampler generalized)
# ---------------------------------------------------------------------------
def sample_sigma_logitnormal(batch: int, device, m: float = 0.0, s: float = 1.0) -> torch.Tensor:
    return torch.sigmoid(m + s * torch.randn(batch, device=device))


def add_noise(x0: torch.Tensor, sigma: torch.Tensor):
    """x0 [B,T,D], sigma [B] -> (x_sigma, eps). (x0 itself is the prediction target.)"""
    eps = torch.randn_like(x0)
    s = sigma.view(-1, *([1] * (x0.dim() - 1)))
    x_sigma = s * eps + (1.0 - s) * x0
    return x_sigma, eps


def add_noise_x0_masked(
    x0: torch.Tensor,
    condition_mask: torch.Tensor,
    sigma: torch.Tensor | None = None,
):
    """Masked x0-objective forward noising (the x0 twin of ``add_noise_velocity_masked``).

    Mirrors the PROVEN motion_expert/bs_train.py x0 recipe -- sigma ~ logit-normal
    (m=0, s=1), x_sigma = sigma*eps + (1-sigma)*x0, the net predicts x0 DIRECTLY --
    generalized with the same per-token ``condition_mask`` gate as the velocity path:
        sigma_eff = sigma * (1 - condition_mask)           # 0 on clean tokens
        x_t       = (1 - sigma_eff) * x0 + sigma_eff * eps
        TARGET    = x0 itself                              # NOT eps - x0
    Clean tokens (condition_mask=True) pass through UNTOUCHED (x_t == x0, sigma_eff
    == 0, no timestep bias) and are excluded from the loss -- identical mask
    semantics to the velocity variant, only the regression target differs.

    Args:
        x0 [B, N, D]      : clean modality tokens (video latents / camera actions /
                            motion features), N = per-modality token count.
        condition_mask    : [B, N] bool, True == CLEAN (do NOT noise, no loss),
                            False == NOISED (supervised). Broadcast over channels D.
        sigma [B]         : optional pre-sampled noise level (shared with sibling
                            modalities of the same sample -- ONE sigma per sample
                            across a multi-target task). If None, sampled
                            logit-normal via ``sample_sigma_logitnormal``.

    Returns (SAME tuple structure as add_noise_velocity_masked, target=x0 in
    place of target_v, so the trainer can switch symmetrically):
        x_t [B, N, D]       : (1-sigma_eff)*x0 + sigma_eff*eps; clean tokens keep
                              x0 exactly.
        sigma [B]           : the per-sample noise level used. INVARIANT: feed this
                              SAME tensor to model.forward's ``t_or_sigma`` (a past
                              bug fed a different t than the noising used).
        target_x0 [B, N, D] : x0 itself (the regression target; loss masks to
                              NOISED tokens).
        noised_mask [B, N]  : ~condition_mask (True == this token was noised /
                              supervised); convenience for the loss reductions.
    """
    B = x0.shape[0]
    if sigma is None:
        sigma = sample_sigma_logitnormal(B, x0.device).to(x0.dtype)
    else:
        sigma = sigma.to(device=x0.device, dtype=x0.dtype)
    eps = torch.randn_like(x0)

    noised_mask = ~condition_mask.bool()                       # [B, N] True == noise it
    sb = sigma.view(-1, *([1] * (x0.dim() - 1)))               # [B,1,1]
    gate = noised_mask.to(x0.dtype)
    while gate.dim() < x0.dim():
        gate = gate.unsqueeze(-1)                              # [B,N,1] broadcast over D
    sigma_eff = sb * gate                                      # 0 on clean tokens

    x_t = (1.0 - sigma_eff) * x0 + sigma_eff * eps
    return x_t, sigma, x0, noised_mask


@torch.no_grad()
def sample_x0(
    predict,
    T: int,
    motion_dim: int,
    steps: int = 50,
    guidance: float = 1.0,
    predict_null=None,
    batch: int = 1,
    device="cuda",
    dtype=torch.float32,
    generator=None,
    sigma_eps: float = 1e-3,
):
    """DDIM-in-sigma sampler from an x0-prediction model (sigma: 1 -> 0).

    Generalized from motion_expert/flow.py: the model invocation is a passed-in
    closure (the PoC's positional H_R signature does NOT apply to the joint model).

    predict(x [B,T,D], sigma_b [B]) -> x0_hat [B,T,D]   (conditional)
    predict_null(...)               -> x0_u   [B,T,D]   (unconditional, for CFG)

    Each step: predict x0_hat, derive eps_hat = (x_sigma - (1-sigma)*x0_hat)/sigma,
    re-noise to the next sigma. CFG applied on x0_hat. Returns clean x0.
    """
    x = torch.randn(batch, T, motion_dim, device=device, dtype=dtype, generator=generator)
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        s = sigmas[i].clamp(min=sigma_eps).expand(batch)
        x0_hat = predict(x, s).float()
        if guidance != 1.0 and predict_null is not None:
            x0_u = predict_null(x, s).float()
            x0_hat = x0_u + guidance * (x0_hat - x0_u)
        si = sigmas[i].clamp(min=sigma_eps)
        eps_hat = (x - (1.0 - si) * x0_hat) / si
        snext = sigmas[i + 1]
        x = (1.0 - snext) * x0_hat + snext * eps_hat
    return x.to(dtype)


# ---------------------------------------------------------------------------
# (3) Per-modality named flow losses + multi-modality dispatch (7-task model).
#
# Thin wrappers over flow_loss_masked that bake in each modality's loss channels
# and default weight per DESIGN_7TASK.md section 3 / task_plan.py:
#   motion -> 283-d feature MSE (all channels), weight w_feat (the geometric
#             joint/smooth terms stay in train.step_loss, computed on x0_hat).
#   vision -> latent flow-matching MSE on noised latent frames only (all channels).
#   camera -> action flow-matching MSE on channels [:9] only, up-weighted (~10).
# Condition-only modalities (condition_mask all True) yield zero loss for free.
# ---------------------------------------------------------------------------
W_VISION_DEFAULT = 1.0
W_CAMERA_DEFAULT = 10.0          # == config.ACTION_LOSS_WEIGHT
CAMERA_RAW_DIM = 9               # == task_plan.CAMERA_RAW_DIM (supervise chan [:9])


def vision_flow_loss(pred_v, target_v, noised_mask, weight: float = W_VISION_DEFAULT):
    """Vision flow-matching MSE on noised latent frames only (all latent channels).

    pred_v / target_v [B, N_lat, C_lat]; noised_mask [B, N_lat] True == supervised
    latent frame. Returns weight * masked MSE (0 when no frame is noised).
    """
    return weight * flow_loss_masked(pred_v, target_v, noised_mask, loss_channels=None)


def camera_flow_loss(pred_v, target_v, noised_mask, weight: float = W_CAMERA_DEFAULT):
    """Camera flow-matching MSE on channels [:9] only, up-weighted (~10).

    pred_v / target_v [B, N_cam, D] (D may be the zero-padded action dim);
    noised_mask [B, N_cam] True == supervised action frame. The loss restricts to
    the first CAMERA_RAW_DIM raw channels (3 pos + 6 rot6d), ignoring the pad.
    """
    return weight * flow_loss_masked(
        pred_v, target_v, noised_mask, loss_channels=(0, CAMERA_RAW_DIM)
    )


def motion_flow_loss(pred_v, target_v, noised_mask, weight: float = 1.0):
    """Motion 283-d feature flow-matching MSE on noised valid frames (all channels).

    The decoded joint / smoothness geometric terms are NOT here -- they live in
    train.step_loss because they need decode_joints(x0_hat). This is only the feat
    (velocity-MSE) term, made mask-aware so it matches the other modalities. For
    text2motion the existing train.masked_mse path is kept unchanged; this wrapper
    is for the gen-present motion tasks (5,7) routed through the unified noiser.
    """
    return weight * flow_loss_masked(pred_v, target_v, noised_mask, loss_channels=None)


def compute_gen_flow_loss(
    modality: str,
    pred_v: torch.Tensor,
    target_v: torch.Tensor,
    noised_mask: torch.Tensor,
    weight: float | None = None,
) -> torch.Tensor:
    """Dispatch a modality name -> its flow loss (vision / camera / motion).

    Single entry point for train.step_loss to sum the supervised modalities of a
    multi-target task (e.g. policy = video + camera). `weight` overrides the per-
    modality default when given (so train.py can thread its --w_vision / --w_camera
    / --w_feat without re-deriving channels). Unknown / condition-only modalities
    raise; the caller should only pass modalities flagged `supervised` by the plan.
    """
    if modality == "video":
        return vision_flow_loss(pred_v, target_v, noised_mask,
                                weight=W_VISION_DEFAULT if weight is None else weight)
    if modality == "camera":
        return camera_flow_loss(pred_v, target_v, noised_mask,
                                weight=W_CAMERA_DEFAULT if weight is None else weight)
    if modality == "motion":
        return motion_flow_loss(pred_v, target_v, noised_mask,
                                weight=1.0 if weight is None else weight)
    raise ValueError(f"compute_gen_flow_loss: no flow loss for modality {modality!r}")


@torch.no_grad()
def sample_velocity_masked(
    predict,
    x0_clean: torch.Tensor,
    condition_mask: torch.Tensor,
    steps: int = 50,
    guidance: float = 1.0,
    predict_null=None,
    device="cuda",
    dtype=torch.float32,
    generator=None,
):
    """Euler ODE sampler for ONE target modality while holding conditioning clean.

    Generalizes sample_velocity to the masked 7-task setting: clean tokens
    (condition_mask=True) are pinned to their given x0 at every step (they are the
    conditioning image/video/camera/motion frames), and only the noised tokens are
    integrated t: 1 -> 0. This is the inference twin of add_noise_velocity_masked.

    Args:
        predict(x [B,N,D], t_b [B]) -> v_hat [B,N,D] : velocity for the WHOLE packed
            target modality (the model still sees the clean tokens via condition_mask).
        x0_clean [B, N, D]   : the clean values for the conditioning tokens; noised
            token slots are ignored (overwritten by sampled noise at init).
        condition_mask [B, N]: True == CLEAN (pinned), False == NOISED (integrated).
        guidance / predict_null : CFG (only the text content differs between passes;
            the conditioning modalities are identical in both, per DESIGN section 6).

    Returns x0 [B, N, D] with clean tokens equal to x0_clean and noised tokens the
    integrated estimate.
    """
    clean = condition_mask.bool()
    keep = clean.to(dtype)
    while keep.dim() < x0_clean.dim():
        keep = keep.unsqueeze(-1)                              # [B,N,1] broadcast over D

    x = torch.randn(x0_clean.shape, device=device, dtype=dtype, generator=generator)
    x = keep * x0_clean.to(dtype) + (1.0 - keep) * x          # pin clean tokens at t=1

    B = x0_clean.shape[0]
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
    for i in range(steps):
        t_b = ts[i].expand(B)
        dt = (ts[i] - ts[i + 1])
        v_hat = predict(x, t_b).to(dtype)
        if guidance != 1.0 and predict_null is not None:
            v_u = predict_null(x, t_b).to(dtype)
            v_hat = v_u + guidance * (v_hat - v_u)
        x = x - dt * v_hat
        x = keep * x0_clean.to(dtype) + (1.0 - keep) * x      # re-pin clean tokens each step
    return x


@torch.no_grad()
def sample_x0_masked(
    predict,
    x0_clean: torch.Tensor,
    condition_mask: torch.Tensor,
    steps: int = 50,
    guidance: float = 1.0,
    predict_null=None,
    device="cuda",
    dtype=torch.float32,
    generator=None,
    sigma_eps: float = 1e-3,
):
    """DDIM-in-sigma sampler for ONE target modality while holding conditioning clean.

    The x0-objective twin of ``sample_velocity_masked`` (and the inference twin of
    ``add_noise_x0_masked``): the model predicts x0_hat DIRECTLY; each step mirrors
    ``sample_x0``'s update -- derive eps_hat = (x - (1-sigma)*x0_hat)/sigma, re-noise
    to the next sigma -- while clean tokens (condition_mask=True) stay pinned to
    their given x0 at every step. CFG applied on x0_hat (matching sample_x0).

    Args:
        predict(x [B,N,D], sigma_b [B]) -> x0_hat [B,N,D] for the WHOLE packed
            target modality (the model sees the clean tokens via condition_mask).
        x0_clean [B, N, D]   : clean values for the conditioning tokens; noised
            slots are ignored (overwritten by sampled noise at init).
        condition_mask [B, N]: True == CLEAN (pinned), False == NOISED (integrated).

    Returns x0 [B, N, D] with clean tokens equal to x0_clean and noised tokens the
    integrated estimate.
    """
    clean = condition_mask.bool()
    keep = clean.to(torch.float32)
    while keep.dim() < x0_clean.dim():
        keep = keep.unsqueeze(-1)                              # [B,N,1] broadcast over D

    x = torch.randn(x0_clean.shape, device=device, dtype=torch.float32, generator=generator)
    x = keep * x0_clean.float() + (1.0 - keep) * x            # pin clean tokens at sigma=1

    B = x0_clean.shape[0]
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        si = sigmas[i].clamp(min=sigma_eps)
        x0_hat = predict(x.to(dtype), si.expand(B)).float()
        if guidance != 1.0 and predict_null is not None:
            x0_u = predict_null(x.to(dtype), si.expand(B)).float()
            x0_hat = x0_u + guidance * (x0_hat - x0_u)
        eps_hat = (x - (1.0 - si) * x0_hat) / si
        snext = sigmas[i + 1]
        x = (1.0 - snext) * x0_hat + snext * eps_hat
        x = keep * x0_clean.float() + (1.0 - keep) * x        # re-pin clean tokens each step
    return x.to(dtype)
