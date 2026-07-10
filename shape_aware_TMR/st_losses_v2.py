"""SnapMoGen-evaluator losses (TMR-paper faithful): SmoothL1 cross-modal recon,
unit + bidirectional cross-modal KL, latent alignment, InfoNCE with sentence-similarity
false-negative filtering. Mirrors SnapMoGen `model/evaluator/losses.py` +
`trainers/evaluator_trainer.py` (config config/evaluator.yaml: lambda_rec=1, lambda_kl=1e-5,
lambda_latent_align=1e-5, lambda_contrast=0.1, infoNCE_temp=0.10, infoNCE_thre=0.80).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def smooth_l1_recon(pred: torch.Tensor, target: torch.Tensor,
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Masked SmoothL1 (mask True = valid frame)."""
    diff = F.smooth_l1_loss(pred, target, reduction="none").sum(-1)  # (B,T)
    if mask is None:
        return diff.mean()
    m = mask.float()
    return (diff * m).sum() / m.sum().clamp(min=1.0)


def kl_to_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(torch.sum(logvar.exp() + mu.pow(2) - 1.0 - logvar, dim=-1))


def kl_between(mu_p, logvar_p, mu_q, logvar_q) -> torch.Tensor:
    """Mean KL( N(mu_p, var_p) || N(mu_q, var_q) ) over the batch."""
    var_p, var_q = logvar_p.exp(), logvar_q.exp()
    kl = 0.5 * torch.sum(logvar_q - logvar_p + (var_p + (mu_p - mu_q).pow(2)) / var_q - 1.0, dim=-1)
    return kl.mean()


def latent_align(z_t: torch.Tensor, z_m: torch.Tensor) -> torch.Tensor:
    """SmoothL1 between the text and motion latent codes (TMR's L_E)."""
    return F.smooth_l1_loss(z_t, z_m)


def info_nce_with_filtering(
    zm: torch.Tensor,                       # (B,D) motion latents (with grad)
    zt: torch.Tensor,                       # (B,D) text latents (with grad)
    sent_emb: Optional[torch.Tensor] = None,  # (B,768) frozen sentence embeddings
    temperature: float = 0.10,
    threshold_selfsim: float = 0.80,
) -> torch.Tensor:
    """SnapMoGen/TMR InfoNCE: symmetric CE over the batch sim matrix; candidate pairs whose
    RAW sentence embeddings are near-duplicates (cos > 2*threshold-1) are masked -inf so the
    model isn't penalized for semantically-equivalent captions. Batch-only (no memory bank).
    """
    x = F.normalize(zt, dim=-1)
    y = F.normalize(zm, dim=-1)
    sim = (x @ y.t()) / temperature                      # (B,B) text->motion
    B = sim.shape[0]
    labels = torch.arange(B, device=sim.device)
    if sent_emb is not None and threshold_selfsim < 1.0:
        real_thre = 2.0 * threshold_selfsim - 1.0        # SnapMoGen's [-1,1] conversion
        s = F.normalize(sent_emb, dim=-1)
        selfsim = s @ s.t()
        selfsim.fill_diagonal_(-1.0)                     # never mask the positive
        sim = sim.masked_fill(selfsim > real_thre, float("-inf"))
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))


def info_nce_membank_frozen(
    zm: torch.Tensor,                 # (B,D) motion emb, L2-normalized, with grad
    zt: torch.Tensor,                 # (B,D) text emb, L2-normalized, with grad
    sent: torch.Tensor,               # (B,S) FROZEN raw sentence emb (e.g. llm2vec 4096)
    queue_zm=None, queue_zt=None, queue_sent=None,   # detached past (Q,D)/(Q,S)
    temperature: float = 0.1,
    threshold_selfsim: float = 0.9,   # cosine on the FROZEN sent embs
) -> torch.Tensor:
    """Memory-bank symmetric InfoNCE with UNGAMEABLE false-negative masking.

    TAP's `info_nce_membank` masks by cosine in the *learnable projected* text space — a
    degenerate optimum exists where the text tower collapses all projections above the
    threshold so every negative is masked and the loss -> 0 (observed: run c5 collapsed).
    Here the mask comes from the FROZEN raw sentence embeddings, which the model cannot move.
    """
    B = zm.shape[0]
    device = zm.device
    if queue_zm is not None and queue_zm.numel() > 0:
        keys_m = torch.cat([zm, queue_zm], dim=0)
        keys_t = torch.cat([zt, queue_zt], dim=0)
        keys_s = torch.cat([sent, queue_sent], dim=0)
    else:
        keys_m, keys_t, keys_s = zm, zt, sent

    pos = torch.arange(B, device=device)
    s_q = F.normalize(sent, dim=-1)
    s_k = F.normalize(keys_s, dim=-1)
    dup = (s_q @ s_k.t()) > threshold_selfsim           # (B, B+Q) frozen-space near-duplicates
    dup[pos, pos] = False                                # never mask the positive
    neg_inf = torch.finfo(zm.dtype).min

    logit_t2m = (zt @ keys_m.t()) / temperature
    loss_t2m = F.cross_entropy(logit_t2m.masked_fill(dup, neg_inf), pos)
    logit_m2t = (zm @ keys_t.t()) / temperature
    loss_m2t = F.cross_entropy(logit_m2t.masked_fill(dup, neg_inf), pos)
    return 0.5 * (loss_t2m + loss_m2t)
