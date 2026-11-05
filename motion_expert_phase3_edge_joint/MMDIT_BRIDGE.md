# Phase3 bridge with separate modality parameters

Implemented on Foundary sep-1 on 2026-11-05. No training job was launched or
restarted. The new Phase3 LoRA is disabled by default and is independent of
this experiment.

## Three stages

Select `--bridge-parameterization` on the joint/image-joint/pretrained-camera
trainers. The image-joint Foundary launcher forwards it to preflight, disposable
fresh/reload probes and production.

| Value | LayerNorm | Q/K/V | Attention output | Seven bridge parameters |
|---|---|---|---|---:|
| `shared` (legacy default) | Shared weights | Shared | Shared | 117,469,198 |
| `separate_norm` | Per modality | Shared | Shared | 117,497,870 |
| `separate_norm_out` | Per modality | Shared | Per modality | 146,857,998 |
| `mmdit` | Per modality | Per modality | Per modality | 234,938,382 |

The counts use hidden2048,16 heads,head_dim128 at the existing seven bridge
layers. Parameter counts were checked with meta-device construction, without
allocating full model weights or a GPU. The full split approximately doubles
the trainable bridge parameter/optimizer storage. It does not double the
number of tokens in attention. Runtime speed has not been benchmarked.

These are three cumulative ablations, corresponding to the user's requested
order. `mmdit` enables all three changes. Native generator and motion-expert
parameters stay frozen; the changes are inside the trainable modality bridge.
LayerNorm already normalizes each token over hidden channels; separate norms
untie the learned scale/bias for the two modalities.

## Full bridge computation

For generator states G and temporal motion states M (shape is excluded):

```text
HG = LN_G(G)                      HM = LN_M(M)
QG, KG, VG = WQ_G(HG), ...        QM, KM, VM = WQ_M(HM), ...
Q = concat(QG, QM)               K = concat(KG, KM)
V = concat(VG, VM)
Y = softmax(Q K^T / sqrt(head_dim) + existing_mask) V
G_out = G + gate_G * WO_G(Y_G)
M_out = M + gate_M * WO_M(Y_M)
```

Each modality has its own trainable norm, Q/K/V, and output projection. Q/K/V
are concatenated **before** attention, so the modalities interact through one
joint attention operation. The shape token bypasses the bridge entirely:
it enters no bridge LayerNorm, Q/K/V, attention or output projection and is
restored bit-for-bit at its original motion-stream row. Shape conditioning
inside the motion expert remains unchanged. See [shape exclusion](SHAPE_EXCLUSION.md).
The two residual gates remain independent zero-initialized scalars, making
every stage an exact identity at initialization. Norm/projection weights remain FP32
as in the existing bridge; each output returns to its own stream's dtype.

The [temporally local mask](TEMPORALLY_LOCAL_ATTENTION.md) now restricts both
within-modality and cross-modal attention to causal Wan groups. Shape is
excluded from bridge queries and keys/values. The configured
directional/bidirectional policy is retained. This implements the requested
MMDiT-style attention parameter split. No new FFNs, adaptive timestep norm
conditioners or backbone attention layers are introduced.
The camera-prediction variant continues to bridge video and body-motion tokens;
native camera tokens remain in the existing native attention path.

## Training and inference

Use `--bridge-parameterization mmdit --phase3-lora-rank 0 --ema-decay 0` for
the full bridge ablation before trying additional LoRA. The last two flags
are already defaults. Leave LR, weight decay, task mix, video/motion losses,
noise schedules, batching and conditioning settings matched to the intended
baseline. EMA and the additional LoRA remain independently opt-in for later
experiments.

Use a new local output directory and a fresh matching acceptance receipt.
The architecture is recorded in the recipe, provenance and source fingerprint.
New architectures save schema2 checkpoints. Sampling reads the saved bridge
mode automatically before constructing the model, for both raw and optional
EMA weights. An explicit conflicting `--bridge-parameterization` is rejected.
Strict state loading also rejects mismatched bridge architectures or missing
modality weights, before copying any trainable state.

Legacy checkpoints without a bridge mode load the original shared architecture.
No automatic conversion of shared weights/Adam moments is performed. Existing
runs must use their original source snapshots for exact fingerprinted resume.
The historical scratch-motion recipe retains its shared bridge and rejects
the new architecture. Camera launchers retain their baseline source guards;
new camera experiments require a reviewed matching contract/preflight rather
than bypassing old guards. Supported camera trainers accept the new flag.

The cached-mask execution policy dispatches the new modes to the same modality
projection implementation, so the optimization cannot silently restore shared
Q/K/V/output weights.

## Verification

All 11 bridge tests plus 10 additional-LoRA tests passed on CPU. They cover:

- independent parameter storage and routing for every modality projection;
- the explicit joint-attention equations, including hidden/head width mismatch;
- exact zero-gate identity with FP32/BF16 inputs;
- nonzero cross-modal influence and gradients to all private blocks;
- cached/uncached forward, backward and mask parity across all stages/tasks;
- exact legacy shared initialization/keys, and projection/residual outputs and
  gradients against HEAD when applying the same new mask to the reference;
- architecture restoration/rejection and raw/EMA save-resume-continue;
- compatibility with optional additional LoRA;
- real two-process Gloo DDP with different target tasks and checkpoint recomputation.

All 12 existing camera/scratch regression tests also passed. Trainer, sampler
and launcher argument parsing was checked without initializing a GPU job.
Native GPU/data acceptance remains pending. No GPU sampling/training was done.
