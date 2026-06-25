# SPDX-License-Identifier: OpenMDW-1.1
"""Text -> motion flow-matching SAMPLER + decode + stick-figure render for the
finetuned Cosmos3-Nano motion model.

Reuses train_motion_ft.py for the model build / weight load / text tokenizer /
packing, then runs the rectified-flow ODE backward in time (t: 1 -> 0) to turn
gaussian noise into a clean normalized 369-d motion sequence, decodes it to world
joints via motion_decode.py, and renders a matplotlib 3D stick-figure mp4.

FLOW-MATCHING CONVENTION (verified against train_motion_ft.forward_loss):
    x_t      = (1 - t) * x0 + t * noise          # t=1 -> pure noise, t=0 -> clean
    v        = noise - x0                          # the target the net regresses
    dx_t/dt  = d/dt[(1-t)x0 + t*noise] = noise - x0 = v
So integrating the ODE from t=1 down to t=0 with step dt>0:
    x_{t-dt} = x_t - dt * v_hat(x_t, t)           # Euler, moving DOWN in t
and at any t the clean-signal estimate is:
    x0_hat   = x_t - t * v_hat                     # since x_t = x0 + t*v

CFG (optional): v = v_uncond + scale * (v_cond - v_uncond), computed per step
with a parallel empty-prompt forward.

Run from cosmos-framework (so `import cosmos_framework` resolves), cosmos env.
    python sample_motion.py --ckpt <ckpt.pt> --prompt "a person walks forward" \
        --frames 120 --steps 50 --cfg 2.5 --out samples/walk
This writes <out>.npy (normalized 369-d), <out>_joints.npy (T,30,3), <out>.mp4.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motion_decode as md  # noqa: E402
import train_motion_ft as T  # noqa: E402  (reuse build/load/tokenize/pack helpers)

from cosmos_framework.data.vfm.sequence_packing import (  # noqa: E402
    SequencePlan,
    pack_input_sequence,
)
from cosmos_framework.model.vfm.utils.data_and_condition import (  # noqa: E402
    GenerationDataClean,
)
from cosmos_framework.model.vfm.mot.attention import build_packed_sequence  # noqa: E402
from cosmos_framework.model.vfm.mot.context_parallel_utils import (  # noqa: E402
    get_context_parallel_last_hidden_state,
)

MOTION_DIM = T.MOTION_DIM
TIMESTEP_SCALE = T.TIMESTEP_SCALE
SPECIAL_TOKENS = T.SPECIAL_TOKENS


# --------------------------------------------------------------------------------------
# model: base build + gen weights + finetuned overlay
# --------------------------------------------------------------------------------------
def load_model(ckpt_path, dtype=torch.bfloat16, verbose=True, lora=None):
    """Build the real net, load BASE gen weights, then overlay the finetuned ckpt.

    train_motion_ft.save_checkpoint stores payload["model"] keyed by
    net.named_parameters() names (e.g. 'motion2llm.weight',
    'language_model.model.layers.0.self_attn.q_proj_moe_gen.weight') -- the FULL
    (unsharded) tensors of every requires_grad param. We copy each key that exists
    in net.named_parameters() with a matching shape.

    LoRA: if the ckpt was trained with --lora, its tensors are LoRA adapters
    ('...lora_A'/'...lora_B') that only exist once LoRA is injected. We auto-detect
    lora keys in the ckpt (or honor lora=True/False) and inject LoRA BEFORE overlay
    so the adapters load and actually affect the generator.
    """
    payload = torch.load(ckpt_path, map_location="cpu")
    state = payload["model"]
    if lora is None:
        lora = any("lora_" in k for k in state)

    net, _ = T.build_network(tiny=False, dtype=dtype)
    if lora:
        net = T.inject_lora_pre_fsdp(
            net, lora_rank=16, lora_alpha=32, lora_target_modules=T.LORA_TARGETS,
        )
    net = T.materialize(net, dtype)
    if lora:
        T.init_lora_weights_post_materialization(net)
    T.load_gen_weights(net, verbose=verbose)
    if verbose:
        print(f"[load_model] lora={'ON' if lora else 'off'}")
    own = dict(net.named_parameters())
    loaded, skipped = 0, 0
    with torch.no_grad():
        for k, v in state.items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v.to(own[k].dtype).to(own[k].device))
                loaded += 1
            else:
                skipped += 1
    if verbose:
        n_moe = sum(1 for k in state if "_moe_gen" in k)
        n_head = sum(1 for k in state if T.is_motion_head(k))
        print(f"[overlay] ckpt={os.path.basename(ckpt_path)} step={payload.get('step')} "
              f"keys={len(state)} loaded={loaded} skipped={skipped} "
              f"(_moe_gen={n_moe} motion_heads={n_head})")
    net.eval()
    return net


# --------------------------------------------------------------------------------------
# packing at sample time: like build_pack_from_batch but with OUR x_t and OUR t,
# and NO internal re-noising (we feed the noisy motion directly).
# --------------------------------------------------------------------------------------
def build_sample_pack(text_ids_list, x_t_list, t_per_sample):
    """Pack B samples for a single ODE step.

    x_t_list      : list of [T_i, 369] float32 -- the CURRENT noisy motion (used
                    only as the action layout payload; we overwrite the embedding
                    from it in the forward, exactly like forward_loss does with x0).
    t_per_sample  : [B] float in [0,1] -- the current ODE time for each sample.

    Mirrors build_pack_from_batch's packing call verbatim (same SequencePlan,
    GenerationDataClean, mrope settings) but stores timesteps = t/scale (the
    network multiplies by scale -> recovers t). Returns the cuda PackedSequence.
    """
    B = len(x_t_list)
    x_t_motion = [m.to(torch.float32) for m in x_t_list]

    plans = [
        SequencePlan(
            has_text=True,
            has_action=True,
            condition_frame_indexes_action=[],
        )
        for _ in range(B)
    ]
    gen_data = GenerationDataClean(
        batch_size=B,
        is_image_batch=False,
        x0_tokens_action=x_t_motion,
    )
    input_timesteps_packed = t_per_sample.to(torch.float32) / TIMESTEP_SCALE

    ps = pack_input_sequence(
        sequence_plans=plans,
        input_text_indexes=text_ids_list,
        gen_data_clean=gen_data,
        input_timesteps=input_timesteps_packed,
        special_tokens=SPECIAL_TOKENS,
        latent_patch_size=1,
        include_end_of_generation_token=False,
        position_embedding_type="unified_3d_mrope",
        unified_3d_mrope_reset_spatial_ids=True,
        unified_3d_mrope_temporal_modality_margin=15000,
        action_dim=MOTION_DIM,
    )
    ps.to_cuda()
    return ps


@torch.no_grad()
def predict_velocity(net, ps, t_per_sample, dtype=torch.bfloat16):
    """Sample-time forward: encode the action tokens (= our x_t) at timestep t and
    return predicted flow-matching velocity v_hat for every motion token.

    This is forward_loss WITHOUT the re-noising: action.tokens already hold x_t, so
    we encode them directly (no `(1-t)x0 + t*noise`). Everything else (motion2llm,
    modality embed, time_embedder(t), two-way packed MoT forward, llm2motion) is
    identical to forward_loss. Returns v_hat [N_motion, 369] float32, in the SAME
    token order as action.tokens (= concatenation of per-sample frames).
    """
    device = "cuda"
    action = ps.action
    assert action is not None and action.tokens is not None

    packed_sequence, _ = net._encode_text(ps)  # frozen reasoner encodes the caption

    x_t = torch.cat([tok.to(device) for tok in action.tokens], dim=0).to(torch.float32)

    # per-sample t -> per-token t (one t per sample's frames), same as forward_loss
    t_per_sample = t_per_sample.to(device).to(torch.float32)
    T_each = [s[0] for s in action.token_shapes]
    t_tokens = torch.cat(
        [t_per_sample[i].repeat(T_each[i]) for i in range(len(T_each))], dim=0
    )

    h_motion = net.motion2llm(x_t.to(dtype))
    h_motion = h_motion + net.motion_modality_embed.view(1, -1)
    with torch.autocast("cuda", enabled=True, dtype=torch.float32):
        ts_emb = net.time_embedder(t_tokens)  # network rescales internally; we pass t
    h_motion = h_motion + ts_emb.to(dtype)

    packed_sequence = packed_sequence.clone()
    packed_sequence[action.sequence_indexes] = h_motion.to(packed_sequence.dtype)

    input_pack, attention_meta, natten_md = build_packed_sequence(
        "two_way",
        packed_sequence=packed_sequence,
        attn_modes=ps.attn_modes,
        split_lens=ps.split_lens,
        sample_lens=ps.sample_lens,
        packed_und_token_indexes=ps.text_indexes,
        packed_gen_token_indexes=action.sequence_indexes,
        num_heads=net.num_heads,
        head_dim=net.head_dim,
        num_layers=net.num_hidden_layers,
        is_image_batch=ps.is_image_batch,
    )
    packed_outputs, _ = net.language_model(
        input_pack,
        attention_mask=attention_meta,
        position_ids=ps.position_ids,
        natten_metadata_list=natten_md,
        memory=None,
    )
    last_hidden = get_context_parallel_last_hidden_state(
        packed_outputs=packed_outputs, parallel_dims=None
    )
    v_hat = net.llm2motion(last_hidden[action.mse_loss_indexes].to(dtype)).to(torch.float32)
    return v_hat  # [N_motion, 369]


# --------------------------------------------------------------------------------------
# the ODE sampler
# --------------------------------------------------------------------------------------
@torch.no_grad()
def _rf_shift(t, s):
    """Cosmos rectified-flow shift applied to a noise level t in [0,1].
    sigma = s*t_bar/(1 + (s-1)*t_bar), t_bar = 1 - t ; s=1 => identity (current behavior),
    s>1 biases the schedule toward higher noise. Returns the shifted noise level."""
    if s == 1.0:
        return t
    tb = 1.0 - t
    sigma = s * tb / (1.0 + (s - 1.0) * tb)
    return 1.0 - sigma  # back to the same t-convention (t=1 noise, t=0 clean)


def sample(net, tokenize, prompt, frames, steps, cfg, dtype=torch.bfloat16, seed=0, shift=1.0):
    """Integrate the rectified-flow ODE from t=1 (noise) to t=0 (clean motion).

    Single sample of length `frames`. Returns x0 [frames, 369] normalized float32.
    With cfg>1, runs a parallel empty-prompt (unconditional) forward each step and
    combines: v = v_uncond + cfg * (v_cond - v_uncond).
    `shift` (s>=1) applies Cosmos's rectified-flow schedule shift; shift=1.0 is the
    original uniform schedule (unchanged); Cosmos action sampling uses shift~5.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    # x lives on cuda (v_hat comes back on cuda); seed on CPU for reproducibility.
    x = torch.randn(frames, MOTION_DIM, generator=g).to(torch.float32).cuda()  # x_{t=1}=noise

    cond_ids = tokenize([prompt])
    uncond_ids = tokenize([""]) if cfg != 1.0 else None

    # t grid from 1 -> 0; shift=1.0 -> uniform (original). shift>1 warps toward high noise.
    ts = torch.linspace(1.0, 0.0, steps + 1)  # [steps+1]
    if shift != 1.0:
        ts = torch.tensor([_rf_shift(float(t), float(shift)) for t in ts], dtype=torch.float32)
    for i in range(steps):
        t_cur = ts[i].item()
        dt = (ts[i] - ts[i + 1]).item()  # positive
        t_vec = torch.tensor([t_cur], dtype=torch.float32)

        ps = build_sample_pack(cond_ids, [x], t_vec)
        v_cond = predict_velocity(net, ps, t_vec, dtype)

        if cfg != 1.0:
            ps_u = build_sample_pack(uncond_ids, [x], t_vec)
            v_uncond = predict_velocity(net, ps_u, t_vec, dtype)
            v = v_uncond + cfg * (v_cond - v_uncond)
        else:
            v = v_cond

        x = x - dt * v  # Euler step DOWN in t: x_{t-dt} = x_t - dt * v_hat

    return x  # x_{t=0} = clean x0, normalized [frames, 369]


# --------------------------------------------------------------------------------------
# render: stick-figure 3D matplotlib animation
# --------------------------------------------------------------------------------------
def render_stick_figure(joints, parents, out_mp4, fps=20, title=""):
    """joints: [T,J,3] world positions. parents: [J] long (root parent = -1).

    Writes an mp4 (ffmpeg) if available; else a gif (pillow); else per-frame PNGs.
    Returns (path, kind). kimodo uses Y-up; we plot X/Z on the floor, Y up.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    joints = np.asarray(joints, dtype=np.float32)
    Tn, J, _ = joints.shape
    parents = np.asarray(parents).astype(int)
    bones = [(j, parents[j]) for j in range(J) if parents[j] >= 0]

    # axis limits from the whole sequence (kimodo: Y up, X/Z floor)
    xs, ys, zs = joints[..., 0], joints[..., 1], joints[..., 2]
    cx, cz = xs.mean(), zs.mean()
    r = max(xs.max() - xs.min(), zs.max() - zs.min(), ys.max() - ys.min(), 1.0) * 0.6

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    def draw(frame):
        ax.clear()
        ax.set_xlim(cx - r, cx + r)
        ax.set_zlim(ys.min(), ys.min() + 2 * r)
        ax.set_ylim(cz - r, cz + r)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_zlabel("y")
        ax.set_title(f"{title}\nframe {frame}/{Tn}", fontsize=8)
        P = joints[frame]  # (J,3) [x,y,z]; plot as (x, z, y) so Y is vertical
        ax.scatter(P[:, 0], P[:, 2], P[:, 1], c="tab:blue", s=12)
        for c, p in bones:
            ax.plot([P[c, 0], P[p, 0]], [P[c, 2], P[p, 2]], [P[c, 1], P[p, 1]],
                    c="tab:red", lw=2)
        ax.view_init(elev=12, azim=-70)

    anim = FuncAnimation(fig, draw, frames=Tn, interval=1000 / fps)

    # try ffmpeg -> gif -> pngs
    try:
        anim.save(out_mp4, writer=FFMpegWriter(fps=fps, bitrate=1800))
        plt.close(fig)
        return out_mp4, "mp4"
    except Exception as e:
        print(f"[render] ffmpeg writer failed ({e}); trying gif")
    try:
        gif = os.path.splitext(out_mp4)[0] + ".gif"
        anim.save(gif, writer=PillowWriter(fps=fps))
        plt.close(fig)
        return gif, "gif"
    except Exception as e:
        print(f"[render] gif writer failed ({e}); dumping per-frame PNGs")
    png_dir = os.path.splitext(out_mp4)[0] + "_frames"
    os.makedirs(png_dir, exist_ok=True)
    for f in range(Tn):
        draw(f)
        fig.savefig(os.path.join(png_dir, f"frame_{f:04d}.png"), dpi=80)
    plt.close(fig)
    return png_dir, "pngs"


# --------------------------------------------------------------------------------------
# decode helper (shared)
# --------------------------------------------------------------------------------------
def decode_and_render(feat_369, out_prefix, skeleton_path, stats_dir, fps=20,
                      title="", is_normalized=True):
    """feat_369: [T,369] (normalized if is_normalized). Decodes to joints, saves
    joints .npy, renders a stick figure. Returns (joints, render_path, kind)."""
    skel = md.load_skeleton(skeleton_path)
    mean, std = md.load_stats(stats_dir)
    feat = torch.as_tensor(np.ascontiguousarray(feat_369), dtype=torch.float32)
    joints = md.decode_features_to_joints(
        feat, skel, is_normalized=is_normalized, stats=(mean, std)
    ).cpu().numpy()  # [T,30,3]
    np.save(out_prefix + "_joints.npy", joints)
    path, kind = render_stick_figure(
        joints, skel["parents"].cpu().numpy(), out_prefix + ".mp4", fps=fps, title=title
    )
    return joints, path, kind


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="finetuned ckpt .pt")
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=2.5,
                    help="classifier-free guidance scale (1.0 = no CFG)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True,
                    help="output prefix; writes <out>.npy/_joints.npy/.mp4")
    ap.add_argument("--skeleton", type=str,
                    default="/home/jungbin_cho/cosmos_motion_ft/skeleton_soma30.npz")
    ap.add_argument("--stats_dir", type=str,
                    default="/weka/jungbin/seed/stats/soma_uniform_motions_20fps/")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--no_render", action="store_true", help="skip decode+render")
    ap.add_argument("--shift", type=float, default=1.0,
                    help="rectified-flow schedule shift (1.0=original uniform; Cosmos action uses ~5)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    dtype = torch.bfloat16

    net = load_model(args.ckpt, dtype=dtype, verbose=True)
    proc = T.build_text_processor()
    print(f"[proc] {type(proc).__name__}")

    def tokenize(texts):
        ids = []
        for t in texts:
            tid = proc.tokenize_text(t)
            if len(tid) == 0:
                tid = [SPECIAL_TOKENS["eos_token_id"]]
            ids.append(tid)
        return ids

    print(f"[sample] prompt={args.prompt!r} frames={args.frames} steps={args.steps} "
          f"cfg={args.cfg} seed={args.seed}")
    x0 = sample(net, tokenize, args.prompt, args.frames, args.steps, args.cfg,
                dtype=dtype, seed=args.seed, shift=args.shift)
    x0_np = x0.cpu().numpy().astype(np.float32)
    np.save(args.out + ".npy", x0_np)
    finite = np.isfinite(x0_np).all()
    print(f"[sample] x0 shape={x0_np.shape} mean={x0_np.mean():.4f} std={x0_np.std():.4f} "
          f"min={x0_np.min():.3f} max={x0_np.max():.3f} finite={finite} -> {args.out}.npy")

    if not args.no_render:
        joints, path, kind = decode_and_render(
            x0_np, args.out, args.skeleton, args.stats_dir, fps=args.fps,
            title=args.prompt, is_normalized=True,
        )
        sz = os.path.getsize(path) if os.path.isfile(path) else -1
        print(f"[render] joints shape={joints.shape} -> {args.out}_joints.npy")
        print(f"[render] wrote {kind}: {path} ({sz} bytes)")


if __name__ == "__main__":
    main()
