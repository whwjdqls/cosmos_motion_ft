# Nymeria random20 v1

One frozen random subset for sep-1, sep-2 and Yonsei: **1,420 physical windows,
20 from every one of the71 held-out recordings**. Parent:11,677 accepted unique
(recording,start) windows, after the existing floor/finite-motion guards.
Duplicate annotation entries do not add weight. Selection uses no model scores.

Use `manifest.json` as the authoritative ordered cohort and `indices.csv` for
inspection. Both contain UUID, start/end frame, eligible recording/global
indices, random selection rank, caption, split, and paired sampling seed.
The metadata are portable: no node-local input paths are required.

A start is an index on the **20FPS recording timeline**, not an index into a
compressed video packet stream. Decode97 frames at start..start+96; start/20
is the first-frame timestamp. Use native source preprocessing and matching
normalization/shape/camera conventions when constructing evaluation inputs.
Do not substitute a nearby window when an input fails; report the failure.

Selection seed:20261110. Rank each recording's eligible starts by the full
SHA256 digest of UTF8 `nymeria-random20-v1|20261110|uuid|start`; retain the20
lowest digests without replacement. The selected rows are stored chronologically.
This avoids dependencies on NumPy/Python RNG implementations. No first window
is forced into the sample. Distinct physical windows can overlap in time;
no temporal rejection is applied because it would change the uniform draw.

Captions use the lexicographically first humanized valid annotation for each
physical window. Freeze the supplied captions as well as physical indices.
The paired per-case sampling seed is stored explicitly in the manifest and
must be reused across conditions/models. Its derivation is documented there.

Run `python verify_selection.py` after copying the directory. It verifies the
parent key hash, exact random draw,20-per-recording counts, frame indices,
IDs and sampling seeds. `SHA256SUMS` verifies transferred files. Every node
should report the same manifest hash:

`47288f230f6b21835f30d5874641b24b0129ab669f99e2f0d03cb1dcf5ced2df`

## Offline evaluation defaults

Read `evaluation_defaults.json`: whole-bridge-dropped reference CFG;
M2V1.1, V2M1.5, video+text→motion1.5. These defaults were chosen before random20
results from the earlier directional W1 EMA150k/fixed71 sweep. M2V1.1 favors
DreamSim/PSNR/SSIM; FVD alone favored1.5. Motion1.5 favors jerk/acceleration
with similar position error to2. There is no universal metric winner.

Native UniPC35, shifts video10/motion3,97frames,20FPS,256px, EMA weights,
provided neutral shape. Video generator captions stay empty. Only the explicit
video+text condition sends the supplied text to the frozen motion expert,
identically in both CFG branches. This does not change training captions.
Future offline evaluations use these values unless explicitly overridden.
The existing periodic fixed71 training-monitor recipe is a separate protocol.

The first sep-1 evaluation uses directional W1 EMA200k, with no camera prediction,
50% V2M /50% M2V /0% joint training. It evaluates M2V, V2M and video+text→motion.
Virtual shards4, cohort rows rank::4,64-case input chunks, M2V batches32,
V2M batches64, decode4; actual backoffs are logged. Preserve this virtual batch
layout where possible across machines; different devices/batch shapes can
introduce BF16 rounding differences even with the same per-case RNG seeds.

Rescore already generated predictions only after checking physical window,
caption/shape, checkpoint/EMA, conditioning, seed and CFG provenance. Do not
relabel a fixed71 or full-test aggregate as this benchmark. Scalar per-window
metrics can be reaggregated; FID/FVD/retrieval require subset-level recomputation
using the matched evaluator and selected GT/features. Per-shard FID/FVD must
not be averaged. Retain per-case scores, TMR embeddings and I3D features so
future rescoring does not require regeneration or decoding.
