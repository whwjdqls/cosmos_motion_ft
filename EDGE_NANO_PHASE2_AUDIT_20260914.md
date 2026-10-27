# Nano versus Edge Phase-2 implementation audit

Date: 2026-09-14. Site: Yonsei datacenter. Status: IN PROGRESS.

This is a coverage/evidence ledger, not a declaration that the implementation is
correct. Earlier smoke tests and frozen-text parity did not inspect every line.
No training weights, training configuration, or allocation is changed here.

## Scope and completion gates

| Area | Evidence required | Current state |
| --- | --- | --- |
| H200 data provenance | Archive manifests, producer code, current hashes and tree identity | 17 critical hashes match; original Nano configuration/log recovered from a checksum-verified Drive archive; exact historical source remains incomplete |
| All loading paths | Pairs, split, source/task routing, retries, crop, masks, grounding, stats, worker RNG | Shared/historical loader comparison and all BONES pair-finiteness checks complete; remaining helper/exposure audit open |
| Raw/cache equality | Complete source-member integrity and cached-member equality with coverage counts | Complete three-tree scan: zero integrity/content errors; 9,294 BONES sources lack cache entries and use fallback |
| Recanonicalization | Coordinate basis, calibration and actual stored-array invariants | All 728 paired sequences compared; body/contact preservation established, learning equivalence not established |
| Architecture | Every projection, activation, norm, initialization, residual and dtype | Both top-level models and motion layers reviewed; Native constructor/freeze/decode and both motion heads covered; remaining dependencies, historical identity and causal validation open |
| Reasoner conditioning | Tokens, positions, image/deep-stack features, every cross-path K/V and mask | Matched 200k caption/attention diagnostic complete; image helper/cache source audit advanced; causal intervention incomplete |
| Training | Noise/target, loss masks/reductions, gradients, optimizer, LR, DDP, resume | Both trainer files reviewed through the update loop; DDP visualization bug found; dependency/history and causal audit remain open |
| Sampling | Checkpoint load, sigma/time, CFG, UniPC, padding, statistics and decode | Both sampler/flow files, Edge fixed-suite builder and shared renderer fully reviewed; active UniPC contracts agree; cache/overlay validation gaps and historical reconstruction remain open |
| Historical Nano | Code actually used to train the archived checkpoint | Original 8-rank configuration, resume log and 200k completion recovered; archive has no source files, so exact code revision remains unproven |
| Documentation | Explicit coverage, snippets, contradictions and unexamined paths | This living report |

Completion requires the entire requested scope, not a small probe or a plausible
explanation. Historical code is not silently replaced with today's shared code.

## Established benchmark evidence

Same BONES repetition/overview manifest: 2,374 cases. Same full retrieval pool,
C45, UniPC35, CFG2, shift3. Each model uses its own training statistics and
caption-conditioning policy; evaluator captions are shared.

| Model | Step | R@3 | FID | Mean penetration, cm |
| --- | --- | --- | --- | --- |
| GT | - | 93.93% | 0, self-reference | 0.353681 |
| Nano contact | 200000 | 82.31% | 0.009299 | 0.356118 |
| Edge BONES-only | 200000 | 14.74% | 0.082723 | 1.721386 |
| Edge legacy / Nano stats | 200000 | 5.39% | 0.124291 | 6.069622 |
| Edge Nano-parity GBS256 | 85000 | 13.98% | 0.114300 | 1.742590 |
| Edge Nano-parity GBS256 | 100000 | 15.00% | 0.104371 | 2.941368 |

Evidence: completed summaries below
`/scratch2/whwjdqls99/cosmos/operations/phase2_quant200k_yonsei_20260914/results`.

The 85k/100k versus 200k comparisons are not training-step matched. Legacy-stat Edge also
performs poorly, so recanonicalization alone does not explain all bad runs.
These results establish a gap, not a specific causal code bug.

## Findings and open risks

### F01. BONES receives two caption-dropout opportunities

Confirmed in inspected shared source; historical impact unresolved.

`motion_expert_joint_attention/nymeria_joint_dataset.py:450` passes the
outer dropout probability into BONES. The inner
`motion_expert_joint_attention/dataset.py:204` does:

~~~python
caption = humanize_caption(r["caption"])
if self.train and self._active_rng().random() < self.cfg_dropout:
    caption = ""
~~~

The outer BONES route at
`motion_expert_joint_attention/nymeria_joint_dataset.py:625` then does:

~~~python
b = self._bones[idx % len(self._bones)]
cap = self._apply_caption_policy("text2motion", b["caption"])
~~~

That policy performs another dropout at line 519. For initially nonempty
captions and independent draws, p=0.1 gives `1-(1-0.1)**2=0.19` empty
captions, not 0.1. Nymeria receives the outer check once.

Edge inherits this route. Current Nano uses the same shared route, so this is
NOT yet an Edge-specific explanation. Historical source and loader-distribution
checks are required before causal attribution. No behavior was changed.

### F02. Camera-RGB world-basis documentation contradicts the producer

Confirmed documentation inconsistency, not a demonstrated recanonicalizer bug.

`PROVENANCE.md`, section 3, labels camera-RGB world poses Kimodo Y-up and
describes a Z-to-Y conversion. The archived producer at
`external/nymeria_kimodo_pipeline/camera/preprocess_camera_rgb.py:112` does:

~~~python
Twr = Twd @ T_device_rgb
Twu = Twr @ T_RGB_UPRIGHT
upos = Twu[:, :3, 3].astype(np.float32)
urot = Twu[:, :3, :3].astype(np.float32)
~~~

These are camera-local right multiplications, not a world-basis conversion.
They preserve the raw Aria world basis. This supports the recanonicalizer's
explicit `ARIA_Z_UP_TO_KIMODO_Y_UP @ rotations` conversion. Actual
restored arrays must still be tied to the archived source. Do not modify the
recanonicalizer merely to match the conflicting overview document.

### F03. Current shared code is not proof of historical training parity

Both current paths use shared loaders and contact losses, but those modules
have received portability, caching, RNG, representation and performance edits.
An archived checkpoint's arguments do not prove every historical default.
The Nano-parity launcher already says historical loss-reduction revision is
unproven. A historical-source audit is still required.

### F04. Frozen text parity does not cover motion-query conditioning

The earlier test compared custom versus native reasoner states for 28 layers
on three prompts, final cosine about 0.99997. It did not validate motion Q/K/V,
cross-path reasoner K normalization, softmax mass, time bias, image/deep-stack
conditioning or gradients.

The new probe observes actual trained motion queries, records per-head
text-attention mass/norms, and compares kernel output with a dense FP32
reference using identical projected/rotated Q/K/V. Low text-attention mass
alone will not be called a bug without Nano comparison and controlled tests.

## H200 and Google Drive lineage

`migration/SITE_STORAGE.md` identifies `/home/jungbin_cho` and
`/weka/jungbin` as H200/a3ultra paths, and
`/mnt/shared/jungbin_cho` as historical A100 paths. Grasp and Yonsei
are distinct restored sites.

`PROVENANCE.md` records the a3ultra hand-off, Nano framework revision
`82f8229` plus archived local patches, and derived motion producers.
`migration/DRIVE_ARTIFACT_MANIFEST.tsv` is the exact archive contract.

| Dependency | Archive contract |
| --- | --- |
| Derived Nymeria core | 4,474 files; 350,742,352,390 bytes |
| BONES proportional UniEgo | 142,231 files; 24,121,032,440 bytes |
| BONES training pairs | 222,686,895 bytes; SHA-256 f4c36ecfd5989b28c7cd5c2a1e3073a45000c4c5e719ac084e80736c922f1630 |
| BONES validation pairs | 6,301,597 bytes; SHA-256 1a03e43fd49aa11ea92f403180fca0cc12363997c7940c78956faa36b98a3fd9 |
| Nymeria video manifest | 64,062,374 bytes; SHA-256 d9e4a6f369b13f89ea3f6dcb91f43dc5f10ff4d5c04fc0142cc53a0372c8c59e |
| Floor calibration | 489,507 bytes; SHA-256 ad27208f9ad8e472d16bcda14e8fdc765b3496248a8763ed95619a9a2f3693c8 |
| Nano contact checkpoint | 6,186,577,167 bytes; SHA-256 a21db649e6402977723af0ee0ba1de55c220ad42c55a01163caca0ea3fc9ac61 |

The storage document's SYNCING/PENDING and scratch statements are dated
2026-09-04, not live status. Some heavy physical data later moved behind
scratch symlinks. Logical active roots are under Yonsei's Lustre weka tree.

`runtime_paths.resolve_legacy_path` preserves an existing original path
before applying a finite prefix map. Active pair resolutions must be checked,
not assumed correct from a run name.

## Architecture details inspected

| Component | Nano | Edge | Interpretation |
| --- | --- | --- | --- |
| Residual width | 4096 | 2048 | Backbone geometry differs |
| Motion blocks | 12, stride3 / 36 | 7, stride4 / 28 | Not depth-matched |
| Motion parameters | About 994M | About 231M | Capacity difference, not a proven cause |
| Motion FFN | Fresh Linear-SiLU/SwiGLU-Linear, width3072 | Same three-linear topology, width3072 | Private motion activation matches |
| Projection initialization | Random normal std0.02 | Random normal std0.02 | Neither clones pretrained motion weights |
| Motion heads | FP32 reference | BF16 older runs; FP32 parity | 85k evaluation kept FP32 |
| Shape head | Linear-GELU-LayerNorm-Linear | Same topology, smaller width | Runtime scale/dtype audit open |
| Output head initialization | Zero linear | Zero linear | Not the final learned values |
| Time embedder | cos/sin, Linear-SiLU-Linear | Same family; cached nonpersistent frequencies | Normalized sigma confirmed earlier; construction-dependent buffer precision detailed below, historical buffer equality unproven |
| Reasoner query mask | Causal reasoner-only | Causal reasoner-only | Prior parity covered only this branch |
| Motion query mask | Full within sample | Full within sample | Dense reference probe prepared |
| Text K read by motion | Qwen-specific normalization | Separate k_norm_und_for_gen before RoPE | Native contracts, not names, determine correctness |

Edge's full attention omits `is_causal`; the inspected frontend defaults
it to False. That omission is not a causal-motion-mask bug.

The Edge cross-path K copy is separate from its causal-reasoner K:

~~~python
k_text_projected = f.k_proj(text_norm).view(-1, self.num_kv_heads, self.head_dim)
k_text_raw = f.k_norm(k_text_projected)
k_text_for_motion = f.k_norm_for_motion(k_text_projected)
~~~

Motion Q/K have trainable norms. Correct frozen text output does not prove the
second K copy is appropriate. Nano heads scale incoming time; Edge accepts
normalized sigma. Actual model call sites and hooks, not API comments, decide
whether the embedder inputs match.

## Data paths inspected

- Pair loader: resolve path, load annotated window, optional ground shift,
  within-pair crop, frame-zero canonicalization, normalization, centered
  neutral skeleton, caption dropout.
- Nymeria loader: native windows, per-window floor plus per-sequence correction,
  guards, padding, task/source routing and worker RNG. Runtime distributions
  and historical equivalence remain open.
- Edge adds BONES actor rewriting. BONES has no reasoner image; TI2M is Nymeria.
- Raw cache identity: logical relative path, bytes and nanosecond mtime.
  Missing/stale entries fall back to NPZ. Windows return writable float32 copies.
- Image cache: source identity, frame, fps, resolution, version. Frozen-condition
  cache additionally keys caption, image bytes, processor/tokenizer types,
  model dtype and namespace. Runtime equivalence/freshness remains open.
- Recanonicalization: decode world transforms, replace Head rotation, rebuild
  yaw-only coordinates, re-encode. Positions and non-Head world rotations are
  passed through by construction; actual stored-output invariants remain open.
- Shared contact helper: raw contact BCE, GT-contact-masked horizontal velocity,
  local raw foot-Y reconstruction. Same weights do not prove same reductions.

## Evidence and outstanding work

Artifacts:
`/scratch2/whwjdqls99/cosmos/operations/edge_nano_full_audit_20260914`.

- `archive_contract.json`: 17 exact critical-file archive records.
- `audit_data.py`: read-only full member/cache and critical-hash scan.
- `critical_hashes.json`: progressive current hash results when running.
- `data_progress.json` and `data_summary.json`: measured coverage.
- `*_members.jsonl`: source member hashes, CRC errors and cache equality.
- `probe_edge_attention.py`: observational 85k checkpoint hooks.
- `edge85k_attention_report.json`: probe output, not yet a passing check.
- `source_capture_inventory.json`: initial capture list and truncation flags.

Next: finish unreviewed model/trainer/native sections; compare real Nano/Edge
time inputs, positions, activations, norms, conditioning and gradients; inspect
caption/window/source/retry distributions; verify stored recanonicalization;
reconcile original Nano source and optimizer state. Source capture is not
equivalent to completed line review. This audit remains incomplete.

## Runtime evidence update: critical hashes, motion-query attention, norms

The previous "prepared" entries above describe the initial launch, not the latest
status. These observations supersede them; they do not close the full audit.

### Full archive/cache scan

All 17 critical archived-file SHA256 and byte-size checks matched. This includes
the contact Nano checkpoint, Nymeria and BONES pair/index metadata, splits, floor
metadata and fixed evaluation windows. This is not a per-file remote checksum
proof for every raw motion archive.

The complete original Nymeria tree scan inspected 732 NPZs (733 total files),
15,755,082,687 bytes. The head-aligned tree inspected 728 NPZs (729 total files),
15,669,537,398 bytes. Both had zero member/cache errors, zero stale cache identities,
and complete cache coverage for their NPZs. Why four original sequences are absent
from the recanonicalized tree remains a separate coverage/producer question.
At the latest live Slurm-step check, BONES had passed 60,500 NPZs with zero errors.
Its scan was still running as step 2219279.27 on node36.

### Edge real motion-query attention

The 85k FP32-I/O checkpoint completed eight observational forwards on BONES cases
0, 1110 and 2300: sigma 1, .99, .9 and .5, each with conditional and empty text.
All 168 layer/sample attention records compared the real motion output with an
independent dense FP32 softmax reference. Maximum relative RMS error was 0.002005,
approximately 0.20%; peak allocated GPU memory was 7.5005 GiB. The probe returned
the original attention result unchanged.

At sigma 1 the mean text attention mass at motion layers 3,7,11,15,19,23,27 was
0.198, 0.787, 0.860, 0.887, 0.563, 0.641, 0.528. Reasoner conditioning is therefore
not entirely masked/disconnected. Attention mass includes any special tokens and
does not prove that useful caption semantics are transmitted. Text V RMS was
substantially below motion V RMS in several layers; this needs a Nano reference
rather than an arbitrary rescaling fix.

The actual Edge time-embedder inputs were sigma itself in FP32, not sigma/1000.
The hook's inherited layer counter is not an execution-layer label for the
top-level head; use its sigma/input fields only.

### Actual checkpoint norm and optimizer tensors

Every inspected motion-block norm tensor remained exactly one in BF16:
28/28 for Edge85k and 48/48 for Nano200k. Their small BF16 optimizer states have
nonzero moments. This is consistent with BF16 updates rounding away near gamma=1,
but it is shared behavior, not evidence of an Edge-only regression. Final motion
norm weights are FP32 and learned in both models (mean Edge .277609, Nano .083339).
No optimizer or model weights were changed.

### Native call-site review and matched probe

Newly captured native sources are joint_motion_model.py (1574 lines) and
mot_joint_layer.py (566 lines); capture is not a claim of finished line review.
The actual model pre-divides sigma by TIMESTEP_SCALE at lines 885-886, then
MotionHeads multiplies it back at motion_heads.py:228. Source therefore predicts
normalized sigma at the native embedder too, unlike the misleading isolated
head default of .001. A matched runtime hook is now prepared to test that.

The native model's frame-position helper at joint_motion_model.py:409-453 uses
the same T x 1 x 1 shape/frame construction as Edge. Native generator defaults
and real position tensors, especially image-conditioned sequences, still need
direct comparison. Native sparse layers explicitly gather only reasoner/generator
rows, then preserve motion rows; full motion layers read the reasoner but never
allow reasoner queries to read motion.

New read-only artifacts:
- probe_nano_attention.py: same three BONES cases and seeded noise as Edge.
- nano_probe_env.sh and launch_nano_probe.sh: legacy Nano stats and owned
  PRO6000 overlap, with 46 GiB initial free-memory guard and an 8 GiB reserve.
- nano200k_attention_report.json: terminal result when the Nano probe succeeds.

The probe uses each checkpoint's trained normalization/caption policy and is not
a step-matched training experiment. Single-forward feature MSE uses GT=0 self-MSE
as reference; it is not the full motion-quality benchmark. Gradient paths,
image/deep-stack equivalence, transform invariants, historical source revisions,
and remaining trainer/loader lines remain open.

## Full scan completion and diagnostic import-path correction

The full compute-node scan completed in 1320.494 seconds, exit 0. It inspected
143,680 NPZs: 732 original Nymeria, 728 head-aligned Nymeria, and 142,220 BONES.
There were zero archive-member errors, cache-content mismatches, stale cache
identities, or critical-file hash mismatches. BONES has exactly 142,231 total
files and 24,121,032,440 bytes, matching the archived inventory. Its 132,926 cached
NPZs matched their source members; 9,294 uncached NPZs use the loader's fallback.
An absent cache is not an invalid source sample.

The actual four missing camera-aligned filenames match the local build manifest's
four missing_camera records:
- S02/20230728_s0_lauren_mayer_act3_m63vl0.npz
- S06/20230818_s0_amy_padilla_act3_51vqz9.npz
- S16/20230829_s0_ray_humphrey_act2_knawby.npz
- S16/20230928_s1_barbara_sandoval_act0_sp1rel.npz

The Yonsei build manifest records all 642 training and 71 test sequences covered,
13,553,128 frames converted, and calibration SHA256
17bd180d105f447b6aa05eaba5c12339a78605a8f2d53e01380000c1f2c01404.
This resolves the inventory discrepancy; independent stored-transform numeric
checks and independent split-membership checks still remain distinct gates.

The first Nano activation probe failed before model/checkpoint loading with
AttributeError: module 'sample' has no attribute 'load_joint_model'.
The scratch-hosted diagnostic lacked a script-directory import priority:
restored_env.sh prepends motion_expert before motion_expert_joint_attention.
Normal native training/evaluation scripts reside in the joint-attention directory,
so their sys.path[0] already selects their native config/sample/flow/heads.
The user authorized correcting the probe. Its startup now explicitly prepends
the joint-attention directory. This is a diagnostic bug, not evidence that a
training checkpoint was evaluated by the wrong backend in the completed benchmark.
The original failed-attempt log is retained separately before retry.

### Additional source differences, not established root causes

1. Native T2M uses raw tokenizer IDs (single EOS for empty text); Edge uses
   BOS + raw text + EOS + start_of_generation (three tokens for empty text).
   See native train.py:966-973 and Edge edge_loader.py:229-259. This changes
   context content and rotary offsets. Edge follows its generation token contract,
   so removing the prefix is not automatically a correct fix.
2. With reasoner images, native joint_motion_model.py:1067-1069 offsets motion
   by reasoner token count; Edge uses next_position_id, which is maximum actual
   reasoner rotary position plus one. These need not coincide after image grids.
   This is TI2M-specific and cannot explain the BONES-only T2M failure.
3. Native module-registration comments are misleading: assigning embed_tokens and
   norms directly registers those nn.Modules. Its named_all_parameters explicitly
   de-duplicates parameter identities and freeze disables those references, so this
   does not establish an optimizer ownership bug.
4. Native periodic checkpoint labels use a zero-based update index; Edge uses
   completed one-based update count. Native final checkpoint/resume increments
   the final count again, skipping one numbered update on extension. This small
   bookkeeping difference cannot explain the large quality gap.
5. The current two dataset classes use worker-specific crop/dropout RNGs based
   on DataLoader worker.seed plus a dataset-seed offset. Current source therefore
   does not simply clone one RNG stream into all workers. Historical behavior and
   exact mid-epoch/resume coverage still require evidence.

Full current native trainer (2350 lines) and Edge trainer (1059 lines) have now
been captured without missing lines. Motion noising, loss reductions and optimizer
update loops have been examined; complete line review of all remaining setup,
visualization, distributed and recovery branches is still in progress.

## Completed Nano runtime comparison and next numerical gates

The corrected Nano probe completed successfully (exit 0, former Slurm step
2215401.17 on node58). It loaded the actual contact200k checkpoint with 149
overlay tensors and zero skipped tensors. Peak probe memory was 30.259 GiB.
The three fixed BONES cases and CPU noise seed match the Edge probe. Each
checkpoint retains its trained statistics and caption policy; this is not a
step-matched experiment.

Across 288 real Nano motion-query attention records, the maximum relative RMS
error against independent dense FP32 attention was 0.00197444 (about 0.197%).
The actual native time-embedder input is normalized sigma, just like Edge:
1, .990000069, .900000036 and .5. No 1000-fold runtime timestep discrepancy was
observed. Real native motion rotary positions also match the source T x 1 x 1
construction; no frame-four compression appeared in this T2M path.

At sigma 1, native layer-average text attention mass ranged from .360 to .946.
Text keys read by motion had RMS 1.76-3.86, whereas Edge's corresponding keys
were approximately .95-.99. Native intermediate layers demonstrably attended
caption action/body words, not only the first token. This is a measured
conditioning difference, not proof that changing key scale would fix Edge.
Native early text V values were even smaller than Edge's; low unweighted text
V RMS alone is therefore not an Edge-specific failure explanation.

Single-forward normalized-feature MSE (GT self-MSE = 0):
| sigma | Nano conditional | Nano null |
| --- | --- | --- |
| 1 | .526641 | .810643 |
| .99 | .433311 | .732309 |
| .9 | .090117 | .096977 |
| .5 | .009820 | .009877 |

These are GT-noised forward diagnostics, not sampled-motion quality or retrieval
metrics. Different generator statistics prevent treating raw MSE scale as a
fair cross-model quality score.

The observed Nano null input is token ID 1, decoded by its tokenizer as a
quotation mark, despite the source calling the configured token EOS.
Both historical native training and its current sampling route use that
configured fallback. Do not silently substitute the tokenizer's actual EOS
for an existing checkpoint. Edge uses its structural BOS/EOS/SOG null prefix.
The exact historical special-token configuration remains a provenance gate.

Read-only Git history shows the September 11 f408565 loader changes replaced
NPZ window reads with the cache helper and added the reasoner-image cache.
The September 2 ddd0954 changes added portable path resolution and shared-stat
overrides, and changed Nymeria's subject wording from "a person" to
"The camera wearer". Neither of those commits introduced the two CFG drops.
The earlier introduction and exact checkpoint source revision remain open.

### Explicit loss assumption requiring measurement

motion_losses.py:34-36 describes local-pose Y as equivalent to decoded height.
That holds for ground-truth yaw-only, zero-vertical canonical transforms.
decode_uniego_torch.py actually reconstructs full 3D canonical rotations and
translations; it does not enforce those constraints on predicted features.
Thus a predicted global Y/tilt error can change decoded foot height while
leaving the supervised local Y channels unchanged. This mechanism is shared
by Native and Edge, so it cannot by itself establish an Edge implementation
regression. Centered joint reconstruction and horizontal planted-foot velocity
do not directly penalize a constant global vertical offset. Feature and
temporal-joint losses still provide other constraints.

New scratch-only diagnostic artifacts are prepared to quantify, not assume,
these effects:
- probe_edge_tokens_gradients.py records per-token/special-token attention,
  weighted text and motion value contributions, padding/batching/train-eval
  invariance, and all trainable/frozen backward-gradient participation.
  It constructs no optimizer and performs no parameter update.
- audit_numeric.py scans all features/neutral values in the complete 143,680
  source inventory for finite values, shapes, contacts and 6D rotation
  degeneracy, and independently computes current Nymeria split coverage.
- The existing compare_camera_head_aligned_uniego.py is run on all 728 actual
  Yonsei old/new/camera pairs into a fresh camhead_numeric_full output,
  checking preserved metadata, contacts, world joints, feet and non-head
  rotations. Every frame contributes to means/extrema; global percentiles
  use the explicitly labelled stride-20 sample.
- diagnostic_env.sh and launch_diagnostic.sh constrain these diagnostics to
  retained owned Slurm allocations. Their result JSON/log/exit files, not
  this preparation note, establish completion.

No training settings, checkpoint tensors or retained allocations were changed.
The all-line architecture/data/history audit remains incomplete; in particular
image-conditioned runtime parity, data-window/caption distributions and
remaining historical trainer branches are still open.

## Completed preservation, backward, and height diagnostics

### Full stored Nymeria recanonicalization comparison

The fresh Yonsei comparison completed with exit 0: all 728 old/new/camera pairs
and all 13,553,128 frames were processed, with no metadata/contact failures.
It independently grouped 642 train, 71 test and 15 other sequences.
The numerical source scan separately found zero non-finite, shape, contact or
degenerate-6D issues in all 732 original and all 728 head-aligned Nymeria NPZs.

All-frame maximum changes after recanonicalization:
| Preserved quantity | Maximum change |
| --- | --- |
| Any joint position | 0.0000467525 m, approximately 0.0468 mm |
| Foot height | 9.43e-12 m |
| Bone length | 1.26e-7 m |
| Non-head joint rotation | 4.36e-6 degrees |

Thus recanonicalization did not introduce material ground-height or skating
changes in the stored corpus. It intentionally changed head rotation and
canonical heading; those changes can still alter learning difficulty.
The proxy camera position inferred using the old calibrated head lever arm
became less accurate (mean 1.55 cm to 3.68 cm), while camera orientation and
relative action rotation improved. This calibration/lever-arm caveat is not
a displacement of the preserved physical joints.

The physical-window comparator used 133,702 usable, calibrated windows across
713 manifest sequences; 7,887 floor-filtered windows were excluded. Stored
contact-foot floating >10 cm and penetration <-5 cm fractions were unchanged,
as were skating ratios. These GT data metrics are not generated-model metrics.
Percentiles use stride-20 sampling. Frame/transition aggregates use every stored
observation; physical means average per-window summaries, not all frames globally.

Full evidence:
`/scratch2/whwjdqls99/cosmos/operations/edge_nano_full_audit_20260914/camhead_numeric_full/comparison_report.json`

### Real Edge backward and token-content participation

The Edge85k diagnostic completed, exit 0. All 94 trainable tensors received
finite, nonzero gradients; frozen parameters received none. No optimizer was
constructed and every trainable parameter version stayed unchanged.
Perturbing only padded inputs produced exactly zero change to valid outputs;
padded-input gradients were zero. Train/eval outputs were identical.
Batched-versus-single output difference was RMS .002894, max .071806 in
normalized-feature space, so batching is not claimed bit-identical.

At sigma 1, caption-content attention mass (excluding structural BOS/EOS/SOG
positions) across the seven Edge motion layers was approximately:
.155, .635, .610, .622, .431, .554, .440.
Weighted caption-value contributions are nonzero, with action/body words among
high-attention tokens. The reasoner is not simply disconnected or entirely
replaced by special tokens. This does not establish that conditioning is strong
enough to explain the retrieval target.

Evidence:
`/scratch2/whwjdqls99/cosmos/operations/edge_nano_full_audit_20260914/edge85k_token_gradient_report.json`

### Shared height-loss weakness, secondary to the Nano/Edge question

The synthetic +1 cm canonical Y-per-transition example accumulated almost
1.99 m of vertical displacement while centered joint loss remained numerical
zero and foot-height loss stayed zero. Feature loss increased by 3.50e-7 and
smoothness by 3.33e-5; total weighted loss rose only .00016697 above GT.
Both original and camhead statistics set the constant global-Y std to 1.
This confirms weak world-height supervision, but is shared with Nano.

In the previously generated fixed 64-case Edge legacy/nanostats 200k sample:
| Lowest-foot height, mean | Value |
| --- | --- |
| GT world | +1.896 cm |
| Generated local pose | +2.672 cm |
| Generated decoded world | -6.223 cm |

The all-foot world-minus-local height discrepancy was 12.241 cm RMS.
Predicted canonical Y increments averaged -1.083 mm per frame.
GT any-foot-below-minus1cm frame fraction was .599%; generated world was 70.03%.
These are the fixed 64-case diagnostic, not the full 2374-case benchmark.
The source sample artifact is
`/scratch2/whwjdqls99/cosmos/operations/phase2_quant200k_yonsei_20260914/debug_sampling_v1/samples.npz`.

### BONES numerical findings are not automatically training contamination

The numerical scan is still in progress. Its first five NaN-containing source
NPZs were independently re-opened directly, not merely inferred from a cache.
Those sources appear in 64 training pairs, but none of those pair intervals
intersect the NaN frames. More sources appeared later in the running scan, so
the complete train/val interval intersection audit remains open. A source-file
NaN count cannot be substituted for the rate of invalid emitted windows.

## Prioritized Nano-versus-Edge differences and matched-200k gate

The user explicitly clarified that shared defects are not the explanatory
target: Nano has strong R@3/FID and better physical motion. Shared-loss findings
remain secondary until a differential mechanism is supported.

The source confirms a specific reasoner interface difference:
- Nano mot_joint_layer.py:327 supplies reasoner K through its ordinary learned
  k_norm to motion queries as well as reasoner queries.
- Edge layer.py:250-254 has Identity for causal text K normalization and a
  separate k_norm_und_for_gen RMSNorm for the copy read by motion.
  The actual motion-read text K RMS is about 1 in Edge versus 1.8-3.9 in Nano.
  This follows the respective backbones; it is not automatically an Edge bug.

The real 200k checkpoint metadata additionally records:
- Edge legacy: batch_size=128, grad_accum=1, LR=2e-4, warmup=1000,
  cosine floor ratio=.1, original statistics, camera-wearer BONES wording.
- Nano contact: batch_size=32 per rank, DDP enabled, LR=2e-4, warmup=1000,
  cosine-to-zero, original statistics, historical raw BONES wording.
  Metadata alone does not establish world size or lifetime training exposure.
- Both record T200, TI97, task mixture .75 T2M/.25 TI2M and BONES fraction .5
  within T2M, with feature/joint/smooth weights 1/1/5 and contact .05/1/10.

New probe_matched200k_conditioning.py compares those actual 200k checkpoints on
the same fixed 64 BONES cases, exact original mean/std hashes, per-case noise,
neutral skeletons and batch size 4. It evaluates correct, shuffled and null
captions, using both original and camera-wearer wording. It records checkpoint
optimizer settings, I/O/parameter dtypes, input hashes, per-feature denoising
errors and observational motion-query attention. No optimizer updates occur.
Equal checkpoint step is explicitly not a claim of matched training exposure.
Completion is established only by each backend's report and launcher exit file.

Historical snapshots b2c307f (July 14) confirm that worker-unique dataset RNG,
BONES-within-T2M selection and double caption-dropout already existed there.
The July 14 trainer snapshot predates the contact terms present in the actual
contact200k checkpoint; the later pre-migration training revision still needs
inspection before historical contact-loss parity is claimed.

Training scripts, training processes and retained allocations remain unchanged.

## Completed matched-200k comparison: Nano is the working reference

The user's priority is explanatory Nano-vs-Edge differences, not a shared
loss weakness. Completed reports are in:
`/scratch2/whwjdqls99/cosmos/operations/edge_nano_full_audit_20260914/`
as `nano_200k_conditioning_report.json` and
`edge_200k_conditioning_report.json`.

Both reports completed all 64 BONES cases and 1280 forward records each.
Case IDs/indices, captions/permutation, mean/std/manifest hashes, and every
per-batch normalized GT, noise and neutral-skeleton fingerprint match exactly.
Both checkpoints are labelled 200k, but training exposure is not matched.
This is GT-noised denoising, not R@3/FID or a free-running motion benchmark.
GT self-MSE is zero.

At sigma=1, frame-weighted normalized feature MSE:
| Condition | Nano contact200k | Edge legacy/nanostats200k |
| --- | ---: | ---: |
| Raw correct caption | 0.6848815 | 1.0948293 |
| Raw shuffled caption | 1.4592065 | 1.1700791 |
| Camera-wearer correct caption | 0.7153434 | 1.0476548 |
| Camera-wearer shuffled caption | 1.4063003 | 1.2146826 |
| Null caption | 1.0697429 | 1.1879902 |
| GT self-MSE | 0 | 0 |

Correct-vs-shuffled improvement is 53.06% for Nano/raw, 6.43% for Edge/raw,
and 13.75% for Edge/training-policy wording. Nano still improves 49.13%
with the camera-wearer wording. Correct captions beat shuffled captions in
61/64 Nano cases versus 48/64 Edge/raw and 50/64 Edge/policy cases.
At sigma=.99 the raw improvements are 61.36% Nano versus 8.15% Edge.
This is evidence of much weaker learned semantic conditioning in the old Edge
checkpoint, not proof that a particular implementation line caused it.

On the first four cases at sigma=1/raw-correct, normalized within-text
attention entropy is .825-.960 across Edge's seven joint layers versus
.376-.773 across Nano's twelve layers. One means uniform over text tokens.
Motion-read text-key RMS is .9446-.9915 Edge versus 1.7497-3.8702 Nano.
Different tokenizers/prefixes and learned models remain confounds. Some Edge
layers have large weighted text-value contributions, so the result is not
simply absence of text attention or uniformly tiny text values.

### Actual saved optimizer, precision and capacity

| Item | Nano contact200k | Old Edge legacy200k |
| --- | --- | --- |
| Trainable motion parameters | 993,733,915 | 230,508,571 |
| Joint layers / hidden width | 12 / 4096 | 7 / 2048 |
| Private motion FFN | SwiGLU, intermediate 3072 | SwiGLU, intermediate 3072 |
| Saved Adam betas | .9, .99 | .9, .95 |
| Saved LR | approximately 0 | 2e-5 |
| Motion I/O, shape/modality heads, final norm | FP32 | BF16 |
| Time embedder | FP32 | FP32 |
| Motion-block parameters/norms | BF16 | BF16 |

Nano has about 4.31 times the trainable motion capacity. Both saved Adam
groups use eps=1e-8, weight_decay=0 and fused=True; both probe runtimes
have TF32 disabled. The historical Nano launcher explicitly requests eight
GPUs with batch32/rank, supporting intended GBS256. Checkpoint metadata alone
cannot prove the entire run's world size, data exposure or uncommitted code.

Do not attribute the old Edge settings to the newer Nano-parity run.
`motion_expert_t2m_edge/train_contact_nanoparity_yonsei.sh:57-62` already
sets FP32 motion I/O, Adam beta2=.99, per-task loss reduction, Nano loss
guards, TF32 off and zero LR floor. It remains a seven-layer, 2048-wide
Edge expert with camera/head-aligned representation/statistics. Its earlier
85k evaluation is not exposure-matched to Nano200k.

Historical `f81b1aa` (July 20) contact helper is identical to the current
helper ignoring outer whitespace. That trainer computes the contact and
decoded feature/joint/smooth terms per task and weights by task count/B.
The July 14 snapshot predates contact loss and is not the contact200k reference.

### Diagnostic limitations, not training failures

Nano's launcher exited 0. Edge saved the complete report before a final
console-print error: `json.dumps` could not serialize NumPy RNG arrays in
checkpoint-extra metadata. The saved atomic report uses a converter and is
valid. Edge's exit1 is therefore a diagnostic reporting failure after all
forwards, not a model inference failure. The user was informed; the script
has not been silently fixed and no inference rerun was requested for this.

That diagnostic's saved-dtype counter also inspects `model` rather than
Edge's `motion_state`, yielding an empty Edge saved-state counter. Do not
use it as dtype evidence. The live loaded parameter dtype comparison above
is populated and valid. No training code or checkpoint was affected.

### Completed corpus and recanonicalization evidence

The full numeric scan completed: 732 original Nymeria files / 13,627,365
frames, 728 camhead Nymeria files / 13,553,128 frames, and 142,220 BONES
files / 20,818,566 frames. Both Nymeria trees have zero numeric issues.
BONES has 679 source files containing 18,689,682 nonfinite feature values;
neutral skeletons have no nonfinite values. This is not evidence of
nonfinite training windows: the pair builder explicitly checks finite
features in each selected interval at `build_bones_pairs.py:297-310`.
The first five sources already had zero intersections with 64 listed pairs.
A full source-to-pair intersection is a separate diagnostic below.

The full scan skips rotation/contact tests for a 2048-frame chunk containing
any nonfinite values. Its zero other-invariant counters therefore do not
prove that finite portions of those skipped BONES chunks satisfy them.

Independent Nymeria split checks found 642 train and 71 test records,
zero duplicates or train/test overlap, and no missing original/camhead
files for either split. The four original-only sequences are outside both.

The completed original-vs-camhead comparison checks all 728 sequences,
13,553,128 frames and 13,552,400 transitions. Maximum joint displacement
is 4.67525e-5 m (0.04675 mm); maximum foot-height difference is 9.43e-12 m.
There are zero metadata/contact mismatches. Thus the recanonicalization
does not directly lift or skate the GT body in these compared sequences.
Changing orientation features and normalization can still change learning;
geometry preservation alone does not establish training equivalence.

### Next bounded differential intervention

`probe_edge_text_key_scale.py` is a scratch-only inference diagnostic for
the same old Edge200k checkpoint and fixed 64 cases. Predeclared scales
.5, 1, 2 and 4 apply only to the text keys read by motion queries. Frozen
causal reasoner keys/outputs, text values, motion keys and all weights stay
unchanged. Scale1 is compared with the saved matched-input control MSE.
It must not be presented as a training fix or selected using final R@3/FID.
The result goes to `edge_text_key_scale_report.json`.

`audit_bones_pair_finiteness.py` directly rechecks all 679 NaN-tainted source
NPZs on a compute node, intersects every materialized train/val interval,
and records unresolved paths, bounds, source and mirror-family split
overlap separately. Output: `bones_nonfinite_pair_intersection_report.json`.

The new diagnostics run only inside owned compute allocations. Active
training, checkpoint weights and retained allocations remain unchanged.

## Further differential audit: normalization arithmetic and exact sampler source

### F05. FP32 final-norm weights do not imply Nano-equivalent activation precision

This is a previously missed numerical difference in the motion decode path,
including the newer Edge FP32-I/O parity option. It is not yet an established
cause of the generated-motion quality gap.

Nano's actual framework source
`cosmos_framework/model/vfm/vlm/qwen3_vl/qwen3_vl.py:388-393`
normalizes in FP32, casts the normalized activation to the incoming dtype,
and then multiplies by the gain:

```python
input_dtype = hidden_states.dtype
hidden_states = hidden_states.to(torch.float32)
variance = hidden_states.pow(2).mean(-1, keepdim=True)
hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
return self.weight * hidden_states.to(input_dtype)
```

The Edge framework's pinned `d4599e2` implementation in
`cosmos_framework/model/generator/reasoner/nemotron_3_dense_vl/nemotron_3_dense_vl.py:55-60`
instead casts the product back to the incoming dtype:

```python
input_dtype = hidden_states.dtype
hidden_states = hidden_states.to(torch.float32)
variance = hidden_states.pow(2).mean(-1, keepdim=True)
hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)
```

Both experts feed BF16 packed motion rows into their final norm. With FP32
gain vectors, Nano's elementwise multiplication produces FP32 output, whereas
Edge's explicit final cast produces BF16 output. Edge subsequently casts that
already-rounded activation to FP32 for its output linear layer; the cast does
not recover the lost precision. Relevant repo calls are
`motion_expert_t2m_edge/model.py:87-92,237-245` and
`motion_expert_joint_attention/joint_motion_model.py:1387-1402`.

This is not a severed gradient path. Casts still propagate gradients, and the
earlier real-model backward audit found finite, nonzero gradients. It also
does not mean that all Edge normalization errors are larger than Nano's.

An operator-only diagnostic completed successfully on node36 inside job
2219279. It instantiated the exact captured norm classes with identical
synthetic BF16 inputs, identical gains, and epsilon held equal at 1e-5 to
isolate the arithmetic-order difference. It read no checkpoint, constructed
no optimizer, and changed no training process. Artifact:
`/scratch2/whwjdqls99/cosmos/operations/edge_nano_full_audit_20260914/final_norm_operator_precision_report.json`.

For FP32 gains:
- Nano output dtype FP32; Edge output dtype BF16.
- Output RMS difference 0.0013823; maximum difference 0.0181231.
- RMS error against the full-FP32 reference: Nano 0.00098330,
  Edge 0.00097324. Edge is slightly closer in this synthetic comparison.
- GT/reference self-MSE is zero.
- Weight-gradient relative RMS difference is 0.00031545, about 0.0315%.
- Adding 1e-5 to every gain changes 100% of Nano output elements but only
  0.593% of Edge elements; adding 1e-4 changes 100% versus 5.941%.
  The Edge changes are sparse, larger quantization jumps, not zero gradients.

These are synthetic operator diagnostics, not R@3/FID, skating measurements
or evidence that a normalization change will improve a trained checkpoint.
A controlled real-checkpoint/training ablation is still needed. Do not alter
the frozen reasoner norms or an existing training run on this evidence alone.

### Exact UniPC source comparison

The complete scheduler files were captured from both active framework paths:
- Nano: `/home/whwjdqls99/cosmos-framework/cosmos_framework/model/vfm/diffusion/samplers/fm_solvers_unipc.py` (773 lines).
- Edge: `/home/whwjdqls99/cosmos-framework-edge/cosmos_framework/model/generator/diffusion/samplers/fm_solvers_unipc.py` (768 lines).

Their only textual differences are copyright/header comments; executable
code is identical. This includes timestep construction, model-output
conversion, UniP/UniC updates, correction history, final-step behavior,
and `step`. Both default to order2, flow_prediction, predict_x0=True,
bh2, lower_order_final=True, thresholding=False and final sigma zero.
The repo wrappers use the same x0-to-flow-velocity conversion and CFG
expression. Matching sampler names alone was weak evidence; this full
implementation comparison is stronger.

The July20 `f81b1aa` historical Native `flow.py` is also text-identical
to the current Native `flow.py`. Its noising distribution and scheduler
wrapper have therefore not acquired a committed post-handoff change.

### Backbone configuration and activation distinctions

The actual loader configuration files establish:
| Backbone property | Nano | Edge |
| --- | --- | --- |
| Reasoner family | Qwen3-VL | Nemotron-3 Dense VL |
| Frozen residual width / depth | 4096 / 36 | 2048 / 28 |
| Frozen FFN intermediate width | 12288 | 9216 |
| Frozen FFN | gated SiLU/SwiGLU | two-linear ReLU-squared |
| Motion FFN | fresh SwiGLU, 3072 intermediate | fresh SwiGLU, 3072 intermediate |
| RMS epsilon | 1e-6 | 1e-5 |
| Rotary theta | 5,000,000 | 100,000,000 |
| mRoPE sections | 24,20,20 | 24,20,20 |
| Interleaving | interleaved | interleaved in the pinned implementation |

Matching motion position IDs does not make the rotary frequencies identical.
The Edge implementation explicitly interleaves axes even though its JSON
does not have Nano's `mrope_interleaved` field. There is no evidence here of
an accidental chunked-versus-interleaved axis mismatch.

The separate Edge motion-read text-key normalization is explicitly requested
by the official Edge model configuration:
`qk_norm_for_text=False`, `use_und_k_norm_for_gen=True`.
It is not an accidental extra normalization added only by the motion port.
Its measured scale nevertheless differs substantially from Nano's pretrained
text keys, so matching the official Edge backbone does not establish
Nano-equivalent conditioning strength for a newly trained motion expert.

The official Edge configuration's video diffusion loss_scale=10 and native
video-resolution settings are not applied to the custom Phase2 loss: the
loader only uses the VLM construction, and Phase2 has its own explicit
1/1/5 contact recipe and shift3 motion noising. Do not multiply the Phase2
loss by ten merely because that unrelated configuration field differs.

### DDP and remaining meaning of batch-size parity

The July20 Native trainer uses `torch.manual_seed(rank)`, passes `seed=rank`
to the dataset, and has worker-unique RNGs. It broadcasts initial trainable
parameters from rank0, averages gradients across ranks, then clips them.
The Edge trainer also seeds ranks separately, and its DDP reduction happens
before clipping. No missing world-size division or identical-rank sampling
stream was found in these inspected paths.

However, equal global batch size does not guarantee exactly the same
variable-length loss weighting across different rank-local batch sizes.
Both current per-task implementations compute a masked frame mean within
each rank/task, multiply it by that task's sample count/local batch, then
average across ranks. Omitting the common feature-dimension factor:

```text
L = (1 / world) * sum_r sum_task [
      (n_rank_task / local_batch)
      * sum_valid_frame_errors(rank, task) / valid_frames(rank, task)
    ]
```

Changing from 8x32 to 2x128 can therefore change the denominators assigned
to individual variable-length examples, even for the same global batch.
Equal lengths within each task remove this distinction. This is a residual
exact-parity caveat, not evidence that it explains the large caption-sensitivity
gap. It should not be silently changed in the existing ablation.

### Historical coverage and completed data follow-through

The complete July20 Native Nymeria loader was compared with the current
source. The 25 added/deleted lines comprise path portability, NPZ/image
caching, caption-comment wording, and explicit shared-statistics overrides
for BONES. Existing window selection, floor correction, frame0
canonicalization, padding, task/source routing and retry logic are unchanged
between those captured versions.

The first committed contact trainer and launcher appear together in
`f81b1aa`. That snapshot contains head-camera-alignment CLI arguments absent
from the actual contact200k checkpoint. Thus the exact uncommitted training
revision cannot be inferred from the nearest Git snapshot. The local Nano
run directory currently lacks `config.json` and `train.log`; both direct
reads returned missing-file errors. Archive restoration, if available, is
still needed to strengthen lifetime world-size/resume provenance. The
checkpoint's recorded arguments/optimizer remain stronger evidence of
actual saved settings than current launcher defaults.

The full BONES source-to-pair check also completed with exit0:
- All 679 NaN-tainted source NPZs were directly rechecked.
- All 585,312 training and 18,459 validation pair intervals were checked.
- Zero nonfinite intersections, unresolved paths, invalid bounds, or failed source rechecks.
- 1,170 training pairs and 26 validation pairs reference otherwise
  NaN-tainted files, but their selected intervals are entirely finite.
- Zero train/validation source overlap or mirror-family overlap.

The text-key-scale intervention remains terminal before any comparisons:
its diagnostic guard tried to read a version counter from inference tensors.
The user was informed and permission to fix/retry was requested. It has not
been silently patched or restarted. The complete earlier matched200k
reports remain usable. All operator/data diagnostics in this update are
terminal; active training and retained allocations were not changed.

The full audit remains incomplete. Capturing a full dependency source file
is not a claim that every unrelated vision/training branch in it has been
reviewed, and neither operator parity nor the current synthetic test proves
the causal origin of the generated-motion quality gap.

## Image-conditioning follow-through: source, local artifacts, and cache

This continuation inspected new framework dependencies and the installed
SigLIP2 patch converter, rather than rerunning a motion smoke test. The Edge
framework HEAD is `d4599e2e43fbd06168e9884205b9b66c3902d8f6`, matching
the revision named in the conditioning-cache namespace. HEAD alone is not
a whole-worktree cleanliness check or proof of every historical process's
loaded code. No new model/data computation, training edit, or job change
was performed.

### Local Edge processor and patch-layout contract

The local `/lustre/whwjdqls99/cosmos/weka/Cosmos3-Edge/config.json` declares
`model_type=cosmos3_edge`. Framework `processors/__init__.py` routes that
local directory to `Nemotron3DenseVLProcessor`; `processors/base.py:125`
then explicitly constructs the native bridge processor rather than allowing
a bare-tokenizer AutoProcessor fallback.

The local image configuration supplies mean/std `[0.5, 0.5, 0.5]`,
patch size 16, merge size 2, and a minimum area of 65,536 pixels. The installed
SigLIP2 image processor enables rescaling and normalization. The custom
image path consumes those settings before patchification; it does not
reuse motion mean/std or arbitrarily substitute Qwen normalization.

`cosmos3_edge_processing.py:226` deliberately calls the installed
`convert_image_to_patches`. Its actual implementation in the active
environment's `transformers/models/siglip2/image_processing_siglip2_fast.py:49` is:

```python
patched_image = image.reshape(C, patch_rows, patch_size, patch_cols, patch_size)
patched_image = patched_image.permute(1, 3, 2, 4, 0)
patched_image = patched_image.reshape(patch_rows * patch_cols, -1)
```

Thus rows are raster ordered and each row is flattened as patch-y,
patch-x, channel. This matches the layout expected by the framework's
SigLIP2 linear patch embedding. The known warning about a different
transformers-main Cosmos3Edge patch layout is not evidence that our
inspected path uses that incorrect layout. The module's claim of bit-exact
upstream-reference equivalence has not been independently tested here.

For the 256-square call, the repo supplies explicit equal min/max pixel
budgets. The outer Nemotron wrapper first resizes by a factor-32 budget;
the inner bridge extracts and forwards those same per-image budgets.
The local patch/merge sizes are consistent with that factor. This produces
a source-derived 16x16 patch grid and 8x8 merged token grid, not a silently
expanded high-resolution grid. No fresh runtime token-count test was run.

### Vision activations, positions, and DeepStack

Both inspected vision configurations have 27 layers, width 1152, 16 heads,
and a 4304-wide GELU-tanh vision MLP. Their architectures are nevertheless
different, and their pretrained vision weights are not interchangeable:

- Nano uses a temporal Conv3d patch embed, learned interpolated positions
  plus visual rotary positions, and Qwen patch mergers.
- Edge uses a linear patch embed, interpolated learned positions, a
  post-vision LayerNorm, and a separate 11,520-wide GELU projector to width
  2048. Its loaded vision attention implementation is explicitly eager;
  softmax is computed in FP32, dropout is disabled in eval, and image
  chunks have separate attention.
- Nano collects visual layers 8, 16, and 24 into three projected DeepStack
  tensors. These are added to visual-token rows after language layers
  0, 1, and 2, not after language layers 8, 16, and 24.
- The repo's `joint_motion_model.py:1310` and layer-loop call sites implement
  that post-layer addition, matching `qwen3_vl.py:944`. Sparse motion-layer
  placement does not suppress the additions on intervening reasoner layers.
- Edge's official multimodal helper intentionally returns an empty
  DeepStack list. The repo rejects unexpected nonempty values rather than
  silently discarding them. Absence of Nano-style DeepStack is a backbone
  difference, not an accidentally omitted Edge tensor.

The actual `get_rope_index` and `get_placeholder_mask` function bodies
in the two framework utility modules are text-identical. Each helper
validates that the number of projected image features matches placeholder
slots before scattering those features into token embeddings.

The previously identified downstream offset difference remains:
`joint_motion_model.py:1067` starts Nano motion positions at the number
of reasoner tokens, while Edge uses `max(reasoner_position_ids)+1`.
For a single 8x8 merged image grid these two quantities differ by 56
positions, assuming the same surrounding text and no padding. This is a
source-derived illustration, not a new checkpoint experiment. It changes
relative image/text-to-motion rotary geometry and could matter for TI2M.
It does not apply to pure T2M, whose sequential reasoner positions make the
two quantities equal for a given token sequence.

### Cache semantics and remaining robustness limits

`conditioning_cache.py` caches the resized frame and prepared multimodal
input embeddings, not all 28 layers of reasoner hidden states or K/V.
The frozen reasoner still runs during the motion forward pass.

The frame key includes source path, byte count, mtime, frame index, fps,
size, and preprocessing version. The condition key includes the actual
image-byte hash, caption, shape/dtype, requested size, model dtype,
processor/tokenizer classes, and an explicit frozen-artifact namespace.
An unchanged image with a different caption therefore cannot legitimately
reuse the same prepared-condition entry.

The active GPU path is frozen and kept in eval by `model.py:105`.
Reads are prefetched on CPU; cache misses fall back to the serial encoder.
Queued writes carry detached CPU tensors. A full write queue drops cache
fills, not training examples. Atomic publication does not expose a
partially written new cache entry.

Remaining robustness limitations, not demonstrated quality failures:

- The condition reader checks identity and required keys, but not every
  tensor's shape, dtype, or finiteness. The frame reader does check its
  exact uint8 shape.
- Frozen weights/processor identity relies on the explicit namespace; it
  is not recomputed from all weight contents for every lookup.
- An unreadable existing entry falls back to recomputation, but hard-link
  publication refuses to replace the existing destination. Such an entry
  is not automatically healed and can cause repeated cache misses.

BONES-only T2M never requests an image condition. Consequently, none of
these image-only differences can directly explain that run's poor
text-only benchmark. Mixed-task training can still couple TI2M behavior
into the shared expert; that is a separate, unmeasured training effect.

### Initialization clarification and coverage limit

Both motion experts initialize their private projection/FFN weights
fresh with normal std 0.02 and zero biases. Nano's
`mot_joint_layer.py:221` explicitly resets them; Edge calls its own reset
at `layer.py:177`. Loader comments describing a generator-cloned pathway
must not be read as evidence that Nano inherited pretrained motion weights
while Edge started randomly.

This update covers active image preparation, the inspected vision-forward
operators, feature injection, and the full shared condition-cache source.
It does not certify every unrelated video/AR/generator branch in the newly
captured 2,726-line Edge `unified_mot.py`, every third-party operator, or
historical image-cache contents. The overall audit remains incomplete,
and no new causal explanation for the T2M gap is claimed.

## Training-step follow-through and an Edge-specific DDP completion bug

This pass closed the remaining source ranges in the current 2,350-line
Nano trainer and 1,059-line Edge trainer, including argument dispatch,
visualization helpers and finalization. It also read the actual shared
Edge launcher. This is source coverage of these files, not certification
of every transitively imported dependency or of the exact historical Nano
working tree.

### F06. Edge's duplicate-visualization decision can diverge across DDP ranks

Severity: operational correctness. This is not an explanation for poor
motions in a checkpoint already saved.

In `motion_expert_t2m_edge/train.py`:

- `run_visualization` returns before any collective when
  `step in completed_viz_steps` at line 680.
- Only rank 0 adds a newly completed step to that set at line 747.
  The later status broadcast does not update peer ranks' sets.
- The last optimizer step can trigger periodic visualization at line 1030,
  followed by the unconditional final-visualization call at line 1046 when
  `no_viz_final` is false.
- Therefore, after successful periodic visualization of the final step,
  rank 0 can skip the duplicate call while peers enter its collectives.
  Rank 0's later cleanup barrier may match a peer's visualization-entry
  barrier, but peers still expect a status broadcast from rank 0 afterward.
  A collective mismatch, hang or process-group error can follow.

The checked Nano-parity wrapper sets visualization every 5,000 steps.
Its shared launcher leaves `no_viz_final=False`, and the trainer defaults
to 200,000 steps. The source-level failure conditions therefore apply to
that configured path when it runs with more than one rank and successfully
renders the final periodic suite. This is not a report that an active job
has already reached or hit that condition.

The final regular checkpoint is saved before these visualization calls,
so the bug does not retroactively alter its trained weights. A fix should
make completion bookkeeping and the duplicate-skip decision collective;
merely relying on the shared manifest file does not synchronize the
already-created in-memory sets. No fix or distributed reproducer was run
during this review.

The current Nano trainer has neither this duplicate-skip branch nor the
same final-step numbering pattern. It also uses a separate two-hour
Gloo visualization-control group. Edge waits for rendering and W&B uploads
on the normal NCCL training group, introducing a separate timeout risk for
long visualizations. That risk also concerns availability, not a direct
change to motion loss or inference semantics.

The potential concern about ordinary recovery-signal handling was checked:
Edge lines 915-925 already all-reduce update-validity and recovery flags
before the normal checkpoint decision. It should not be reported as
missing synchronization without evidence of a distinct asynchronous race.

### What matches, and what no longer explains BONES-only T2M

The saved Nano contact200k arguments and old Edge legacy200k arguments
agree on these intended training quantities:

| Quantity | Recorded setting |
| --- | --- |
| T2M / TI2M task mass | 0.75 / 0.25 |
| BONES fraction within T2M | 0.5 |
| Expected BONES / Nymeria T2M / Nymeria TI2M mass | 0.375 / 0.375 / 0.25 |
| Motion output cap / TI2M valid frames | 200 / 97 |
| Motion target | Clean normalized x0 |
| Motion time distribution | Shift-3 logit-normal |
| Feature / centered decoded-joint / smooth weights | 1 / 1 / 5 |
| Contact / planted-foot velocity / foot-height weights | 0.05 / 1 / 10 |
| Contact slope / motion fps | 2 / 20 |
| Peak LR / warmup / gradient clip | 2e-4 / 1,000 / 1 |

These are not claims of equal lifetime exposure or bitwise identical
training. Old Edge still differs in LR floor, Adam beta2, precision and
pooled reduction as documented above; the newer parity wrapper already
overrides those settings.

For a batch containing only `text2motion`, the Edge per-task grouping has
exactly one group with fraction 1. Its loss is mathematically identical
to pooled reduction, including the contact helper. Variable clip lengths
do not change that single-group equivalence. Thus the reduction switch
cannot directly explain BONES-only T2M's poor benchmark.

For the compared mixed-task recipe, the newer Edge per-task path matches
the current Nano formulas: one shared decode, centroid-relative joint
MSE, adjacent-frame decoded displacement MSE, the same contact helper,
then task sample-count weighting of each task's masked means. The
previously documented rank-local variable-length reduction distinction
between 8x32 and 2x128 remains; equal global batch size is not bitwise
or exactly frame-weight-equivalent training.

With the saved Nano pure-DDP setting (`fsdp=False`), both trainers use a
single `torch.nn.utils.clip_grad_norm_` call over the plain trainable set
after averaging gradients across ranks. Nano's DTensor-vs-plain clipping
split is inactive for this reference. There is no missing world-size
division or separate clipping of Nano's heads versus blocks in that path.

Nano draws unused random motion/gen time tensors before selecting its
active motion sigma; Edge omits unused generator work. This changes RNG
consumption and prevents equal seeds alone from proving matched inputs.
It does not change the active marginal noise distribution. The earlier
64-case comparison used matching tensor fingerprints, not that assumption.

### Loader reachability and resume limits

The shared dataset's length explicitly includes `len(self._bones)`
(`nymeria_joint_dataset.py:469`). Routing BONES by
`idx % len(self._bones)` therefore does not cap it to a short Nymeria-only
prefix. Both active trainers refuse to silently proceed without BONES
when a positive BONES fraction was requested.

Current Edge saves data epoch and RNG state but not the exact consumed
batch cursor or every persistent worker's private RNG state. Recreating
the iterator on resume is not an exact continuation of the data stream.
Native checkpoints also do not preserve that full loader state.
Their sampler-epoch conventions differ: Nano uses the current step at
each new pass, while Edge increments an epoch counter. These are
reproducibility/exposure differences, not demonstrated causal failures.

The inherited shell launcher's Slurm comments describe Grasp resources,
but the Yonsei wrapper executes it with Bash inside an existing
allocation and exports the Yonsei site/paths. Those `#SBATCH` comments
do not request Grasp resources in that execution. The launcher correctly
forwards the wrapper's loss, precision-policy and DDP process-count
settings. Running it without the wrapper's environment would not reproduce
the recorded contact recipe.

### Remaining work after this pass

No new generation, gradient test, or benchmark was run. No training code,
checkpoint, allocation, or Git state was changed.

Remaining source dependencies include the shared camera/video loader and
checkpoint helpers not yet separately captured. Exact archived Nano
world-size/resume provenance is also unresolved. The already-inspected
`UPLOAD_PATH_MAP.md:18` identifies
`data:cosmos_run_metadata_archives/` as the remote source of run
configurations/logs, which is a concrete next lead rather than assuming
that the missing local files never existed.

The failed text-key-scale diagnostic remains unmodified pending explicit
approval to fix its inference-tensor version-counter guard. The broad
audit stays open; finding an operational DDP bug does not establish the
cause of the text-conditioning and physical-quality gap.

## Recovered original Nano run metadata

This resolves the earlier missing run-configuration/log lead, not the entire
historical-source requirement. The documented `data:` alias is not configured
on Yonsei; the existing `gggdrive:` remote contains the run metadata archive.
No OAuth configuration was changed and no bulk restore was performed.

Archive under `gggdrive:cosmos_run_metadata_archives/`:

`ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d_w1_1_5_contact_c0p05_v1_h10_s2_unipc35.tar.zst`

The 39,080,614-byte archive was streamed and inspected on node36 in allocation
2219279, without extracting into any live run directory. Its compressed
SHA-256 matched the remote sidecar:

`4c76b08f2c833169a19a4ee3a741d299a8fd0028c34e5c288c56f4365bf51ce8`

The diagnostic exited zero. The captured report is
`/scratch2/whwjdqls99/cosmos/operations/edge_nano_full_audit_20260914/nano_contact_archived_metadata_report.json`.

The archived `config.json` explicitly records:

~~~json
{
  "world": 8,
  "batch_size": 32,
  "ddp": true,
  "fsdp": false,
  "steps": 200000,
  "lr": 0.0002,
  "lr_schedule": "cosine",
  "warmup": 1000,
  "init_gen": null,
  "init_motion": null
}
~~~

Together with the audited per-rank DataLoader semantics, this confirms the
recorded Nano recipe uses global batch 256. The task weights are T2M 0.75 and
TI2M 0.25, with BONES fraction 0.5 within T2M. The loss policy explicitly
records feature/joint/smooth 1/1/5 and contact/foot-velocity/foot-height
0.05/1/10, contact scale 2, GT contact masks and 20 fps. The recorded motion
statistics are the original H200 repository's UniEgo283 mean/std paths.

The archived log records a fresh start, then an optimizer-restored resume
from checkpoint 100000 to step 100001. It reports 149 loaded weight entries,
zero missing/shape-mismatched entries and `fsdp_sharded=False`. Logged step
numbers drop from 102280 to 100020 across this restart, so the log is not
one uninterrupted monotonically increasing trajectory. Its final records
include checkpoint 200000, final visualization and `[train] done`.
All 10,114 parsed step records report zero skipped steps; no bomb-guard or
error record was found in the captured log. This establishes what the
rank-zero archive reports, not that every unlogged worker event is known.

There are 126 tar members and no Python/shell source members. A saved
configuration can reflect the last launch; it does not independently prove
unchanged rank count for every earlier segment. The exact historical
worktree revision and uncommitted modifications remain open.

## Additional shared helper source coverage

The following previously pending files were captured and read in full:
`nymeria_world/nymeria_camera_rgb_dataset.py`,
`nymeria_world/camera_to_action.py`,
`nymeria_world/nymeria_camera_dataset.py`, and
`motion_expert_joint_attention/checkpoint_utils.py`.

The active Phase-2 path imports camera/video helpers, not these standalone
camera dataset classes. The shared decoder seeks backward to a keyframe,
indexes decoded frames by rounded PTS times the configured FPS, and returns
the exact requested indices in order. Missing requested frames raise rather
than being silently duplicated or padded. Its constant-FPS/zero-origin
assumption and possible long decode when an index is missing are shared
robustness risks, not evidence of an Edge-specific quality difference.

The upright-RGB loader reads float32 camera positions/rotations, resolves
the paired RGB path, and computes backward framewise relative poses.
The absolute-to-relative transform uses higher precision internally.
The default device-to-OpenCV branch is not an extra remap on the active
upright-RGB helper path; no double-remap was found here.

Both standalone camera dataset modules import `resolve_legacy_path`
after their `__main__` blocks. Direct CLI instantiation can therefore use
the name before that import executes, whereas normal module import
completes before Phase-2 later calls the helper. Their Nano-only factory
namespaces are likewise outside the active Edge helper route.
These are inactive compatibility issues, not an explanation of BONES-only
T2M quality.

The shared retry path cancels SIGALRM after a failure but does not restore
the prior handler on that exception path. This is another shared
robustness issue, not demonstrated silent motion corruption.

The native checkpoint helper loads the joint payload's `model` state
without silently selecting an EMA variant. Its separate generator-DCP
EMA helper is for generator warm-starts; the recovered Nano configuration
has neither generator nor motion warm-start enabled.

## Nano-parity 100k decision-point evaluation

The user requested pausing the broader investigation until this evaluation,
and deprioritizing the old BONES-only failure investigation if 100k showed
a substantial improvement over 85k and surpassed BONES-only.

Evaluation of the requested `step_000100000.pt` completed on node32,
RTX4090 allocation 2212374, Slurm step 2212374.39. Exit status was zero
and the step was absent from Slurm afterward. All 2,374 BONES
repetition/overview cases were covered, with zero missing evaluator captions.

Results:
`/scratch2/whwjdqls99/cosmos/operations/phase2_quant200k_yonsei_20260914/results/edge_nanoparity100k__bones_repetition_overview/summary.json`.

This used FP32 motion I/O, camera/head-aligned generator statistics, the
same fixed manifest and seeded noise, UniPC35/CFG2/shift3, evaluation batch
32 and the separate C45 evaluator normalization. Comparison against the
85k saved signature found no differences in manifest, protocol, mean/std
hashes, evaluator identity, sampler settings or evaluation batch size.
The GT physics/retrieval results were also identical. The checkpoint
identity and required saved step intentionally differ.

| Metric | GT | Nano-parity 85k | Nano-parity 100k |
| --- | --- | --- | --- |
| R@3, percent, higher is better | 93.93 | 13.98 | 15.00 |
| FID, lower is better | 0, self-reference | 0.114300 | 0.104371 |
| Mean per-case floor penetration, cm | 0.353681 | 1.742590 | 2.941368 |
| Frames with every foot above 5 cm, frame-weighted percent | 2.3433 | 20.1689 | 31.6569 |

R@3 improves by only 1.02 percentage points from 85k. FID decreases by
8.69 percent, but remains worse than BONES-only 200k at 0.082723.
The 100k R@3 is only 0.26 points above BONES-only's 14.74 percent.
No uncertainty estimate was computed, so that small difference is not
called a statistically established win. Penetration and the all-feet-high
measure both worsen from 85k. The latter is not an error on every frame
(jumping is legitimate), which is why the identical GT reference is shown.

This does not meet the requested substantial-improvement criterion.
The bundled Nano-parity changes have not yet closed the quality gap;
this result does not prove that longer training cannot help or identify
a specific implementation fault. The full audit remains incomplete and
justified. No training process, weight, configuration or allocation was
changed by this evaluation or documentation update.

## F07. Current Edge and Nano wrappers select different attention backends

Classification: confirmed current execution-path difference; numerical
correctness and causal impact remain unproven. This is not yet a kernel-bug
finding, and the original H200 training kernel has not been observed directly.

A configuration/backend-selection inspection ran on the owned node32
RTX4090 allocation 2212374 and exited zero. It loaded the installed attention
modules through the respective production launch wrappers, without loading
a motion model, executing attention on test tensors, or updating training.
`I4_ATTN_BACKENDS` was unset. Representative BF16/head-dimension-128 packed
shapes were checked for causal/noncausal and requires-grad true/false modes.

| Wrapper | Query/KV heads | Selected backend in all four inspected modes |
| --- | --- | --- |
| Current Yonsei Nano | 32/8 | flash2 |
| Current Yonsei Edge | 16/8 | natten |

Installed versions reported by that compute process:

~~~text
torch            2.10.0+cu128
flash-attn       2.7.4.post1+cu128.torch210
flash-attn-3-nv  1.0.3+cu128.torch210
natten           0.21.6.dev6+cu128.torch210
nvidia-cudnn     9.10.2.21
GPU architecture 89
~~~

This is a wrapper/environment observation, not a profiler capture of every
running trainer or proof that all historical launches had the same environment.

### Why dispatch differs

The current Edge framework's
`cosmos_framework/model/attention/flash2/checks.py:78` explicitly rejects
variable-length inputs:

~~~python
if is_varlen:
    target_fn(
        "Flash Attention v2 (flash2) varlen is banned due to instability. Please choose another backend.",
        exception=ValueError,
    )
    return False
~~~

The current Nano framework comments out that same rejection at lines 79-88
as part of its local hardware-portability modifications. Nano also disables
the FlashAttention-2 package version-range guard; Edge retains it, but the
currently installed 2.7.4.post1 build satisfies Edge's range. The variable-
length rejection is therefore relevant even though FlashAttention 2 is
installed and recognized by both frameworks.

Edge's registry additionally contains cuDNN, but the observed compatible
selection for these packed 4090 inputs is NATTEN, not cuDNN.
The NATTEN sub-backend selection source routes architecture 89 with BF16
and head dimension 128 to `cutlass-fmha`.

### What the source does and does not establish

Both main `attention()` frontends set the default dot-product scale to
`query.shape[-1] ** -0.5`, which is `1/sqrt(128)` here, and forward the
separate Q/KV cumulative sequence lengths. The asymmetric mask remains
two calls: reasoner-only causal attention and full motion attention over
the same sample's reasoner-plus-motion keys. Motion is noncausal;
reasoner Q/KV lengths are equal, so NATTEN's top-left versus FlashAttention's
bottom-right causal convention does not change this active causal mask.

Common shape/varlen checks and the FlashAttention-2 forwarding functions
match after excluding blank/comment-only lines. The main frontend
differences are cuDNN registration and documentation, not an extra scale.
NATTEN's Cosmos forwarding function is byte-identical in the two framework
trees.

The active NATTEN call explicitly requests
`backward_use_pt_reduction=True`. Its module enables KV parallelism and
unrestricted memory usage. Installed `natten/functional.py:311` dispatches
the selected path to `natten/backends/fmha.py`.
That wrapper implements GQA by `torch.repeat_interleave` on K/V heads
before the custom autograd function, retaining the ordinary autograd
reduction through those repeats.

The custom function saves Q/K/V, attention output, logsumexp and the
separate cumulative lengths. It passes the saved causal flag and scale,
plus Q/KV lengths, into `natten._libnatten.fmha_backward`. No Python-layer
detach or additional attention-scale factor was found in this path.
The backward ignores a gradient for the auxiliary logsumexp output;
Phase-2 does not consume that output, so this is not a missing active
loss contribution.

The compiled backward kernel remains a boundary not validated by reading
its Python caller. The earlier trained-query dense-reference probe checked
forward values, not numerical Q/K/V gradient agreement. Nonzero motion
parameter gradients likewise do not prove the gradient values are correct.
No NATTEN-versus-FlashAttention numerical forward/backward comparison has
been run in this pass.

### Coverage added and next discriminating check

The active one-dimensional attention dispatch/check/forward paths were read
through the framework frontend, backend selection, common checks, masks,
environment filters, FlashAttention 2/3 wrappers and NATTEN wrapper, then
through the installed NATTEN standard-attention dispatcher and the complete
284-line CUTLASS FMHA autograd wrapper. Inactive multi-dimensional attention,
other GPU-family kernels and compiled extension internals are not silently
counted as fully audited.

An isolated comparison of NATTEN and direct FlashAttention against a
per-sample FP32 reference is proposed for variable lengths, GQA, the two
active mask modes, and Q/K/V backward gradients. Approval was requested;
the numerical test has not been launched. No safety guard, backend,
framework file, model parameter or training job has been changed.

## Native Edge norm placement, layer mapping and conversion provenance

This source-only pass continued while the isolated attention-gradient
comparison awaited approval. No numerical backend test, checkpoint conversion,
weight rewrite, training restart or new allocation was performed.

### Cross-attention normalization matches the native Edge path

The actual `PackedAttentionMoT` implementation in
`cosmos_framework/model/generator/mot/unified_mot.py:658-710` keeps
reasoner-self-attention keys separate from the keys read by generation
queries. Its relevant branch is:

~~~python
k_und = self.k_norm(k_und)
# ...
k_und_normalized = self.k_norm_und_for_gen(k_und)
_, k_und_for_gen_ = self._apply_rotary_pos_emb(
    q_und_, k_und_normalized,
    get_und_seq(packed_cos), get_und_seq(packed_sin),
    unsqueeze_dim=1,
)
~~~

In the pinned Nemotron configuration `self.k_norm` is `nn.Identity`.
Thus the motion implementation's use of the projected reasoner K, followed
by the frozen cross-path norm and then RoPE, matches the native operation.
Passing an already-rotated dummy Q into the paired helper and discarding its
Q output also occurs in the native code; it does not rotate the consumed K
twice. Native reasoner V is unnormalized and unrotated, matching the custom
motion path's V source.

The native decoder uses pre-attention norm, attention-plus-residual, then
pre-MLP norm and MLP-plus-residual. The custom Edge text/motion layer follows
that ordering. There is no omitted dense-path adaptive gate in these
inspected native decoder branches. MoE load-balancing masks and
autoregressive-memory/context-parallel branches are not active substitutes
for the two-role Phase-2 motion forward.

The Nemotron config wrapper folds upstream 56 alternating attention/MLP
blocks into 28 fused decoder layers only when the original count is 56.
Explicit text-config overrides apply after that fold. The Phase-2 loader
checks the resulting 28-layer/2048-width/16-query-head/8-KV-head geometry.
Motion attaches at indices 3, 7, 11, 15, 19, 23 and 27. Nano's corresponding
selection rule also uses `(i + 1) % stride == 0`, with stride three.
The different depth/capacity is deliberate, not an off-by-one discrepancy
between the advertised and selected Edge layers.

A further initialization detail is not bitwise parity: Native casts motion
blocks to the base BF16 device before its fresh normal initialization;
Edge initializes its new blocks in default FP32 storage before casting
them to BF16. Both request standard deviation 0.02, but seed equality does
not imply equal random draws or an identical rounding path. This does not
establish a quality cause.

### Raw checkpoint indexes include the required reasoner weights

Two previously uninspected downloaded JSON indexes were read:

`/lustre/whwjdqls99/cosmos/weka/Cosmos3-Edge/model.safetensors.index.json`

`/lustre/whwjdqls99/cosmos/weka/Cosmos3-Edge/transformer/diffusion_pytorch_model.safetensors.index.json`

The root index has 698 keys and the transformer index 549. The reasoner
subset is exactly 254 keys: token embedding, final norm, and nine weights
per each of 28 layers. It includes all 28
`layers.N.self_attn.k_norm_und_for_gen.weight` entries.
Every one of those 254 keys is also present in the transformer index and
points to the same transformer shard after accounting for the directory
prefix. This is index/key-path agreement, not a comparison of tensor values.

The current converter deliberately drops root-index cross-path K-norm
entries and takes them from the transformer component index, accommodating
an older export naming defect. Both indexes here supply the correctly
named 28 entries; no absent raw K-norm entry was found.

### Conversion and partial reasoner loading

The 99-line repository converter
`native_phase_training/convert_edge_to_dcp.py` was read in full, together
with its shell launcher and the framework converter. The repository wrapper
changes the processor/VAE lookup to local artifacts. It does not implement
an alternative QKV transformation: it delegates HF loading to
`Cosmos3OmniModel.from_pretrained_dcp`, then saves the model state through
the framework DCP planner.

The complete 779-line framework
`cosmos_framework/inference/model.py` was read. Its Diffusers mapping
renames `to_q/to_k/to_v/to_out` to `q_proj/k_proj/v_proj/o_proj`, and adds
the `language_model.model.` prefix to reasoner keys. The cross-path norm
name itself is retained. The planner rejects duplicate mapped targets and
missing required target keys. Its permitted missing optional modality/vision
heads do not exempt missing reasoner weights.

The safetensors reader checks that indexed names exist in each shard and
copies the requested tensor slice into the target storage. Optional
anonymous-memory staging clones a tensor before the device copy, not a
semantic reshape or learned transformation.

Phase-2's `edge_loader.py:190` separately requires every reasoner target
key in the DCP, including the cross-path norms, before loading. It retains
those norms when removing unused generation submodules. This source path
does not intentionally discard pretrained cross-path norm weights.

### F08. Artifact pins do not cover complete frozen reasoner contents

Classification: reproducibility/integrity coverage gap, not evidence that
the current weights are wrong.

`motion_expert_t2m_edge/config.py:237-244` hashes the root model index,
DCP `.metadata`, vision-weight file and motion statistics, and checks the
framework HEAD. It does not hash the DCP's full reasoner tensor shards.
The tokenizer/processor files must exist, but that function does not include
their contents in its returned hash contract. A framework HEAD check also
does not prove the worktree has no local modifications.

After loading, `edge_loader.py:214-220` checks three sentinel tensors for
finiteness, not equality of all 254 reasoner tensors against their original
HF values. Likewise, the conversion launcher treats an existing nonempty
`model/.metadata` as sufficient to skip conversion; it does not independently
re-prove that an existing DCP came from the currently selected HF revision.

Consequently, successful startup and matching metadata must not be described
as proof of complete frozen-weight identity. No tensor-content mismatch was
observed or asserted in this pass. Exact HF-to-DCP reasoner equality remains
an explicit outstanding check rather than a silently assumed invariant.

### Additional documentation reviewed

The complete 494-line
`native_phase_training/EDGE_EGOCENTRIC_INFERENCE_AUDIT.md` describes Grasp
base-video prompt/inference experiments. Its structured prompts and
shift/steps/guidance recipes belong to Phase-1/video diagnostics; they are
not a hidden Phase-2 motion training recipe. Its one-clip observations do
not establish the cause of BONES-only T2M retrieval failure.

The broad audit is still incomplete. The remaining compiled attention
backward question and frozen-weight identity gap are distinct from the
source-level norm/key-mapping checks completed here.


## Remaining reasoner paths, rotary context and completed file coverage

This pass completed the previously untracked ranges of the shared Edge MoT
implementation and the Native Qwen3-VL implementation, using the captured
source rather than physically re-reading files. It also completed the
tracked ranges of the Edge model, loader, layer and configuration modules,
and the Native Cosmos loader. These are source-review conclusions, not new
numerical tests and not a claim that every repository dependency is covered.

### What actually conditions Phase-2 motion

The Edge wrapper at `motion_expert_t2m_edge/model.py:114` dispatches each
sample to text-only or image-plus-text embedding preparation. It checks
that captions, modes and images have equal counts, rejects an image on
T2M, and requires one on TI2M. The forward at line 174 consumes those
per-sample embeddings and positions; line 234 iterates through the custom
joint layers. It does not call the framework's autoregressive text
generation interface.

The remaining shared framework code makes this distinction explicit:

- `unified_mot.py:771` is cached reasoner attention for autoregressive decoding.
- `unified_mot.py:951` is the native packed understanding/generation forward.
- `unified_mot.py:1558` is the separate reasoner-only hidden-state forward.
- `unified_mot.py:1764` samples new text tokens through the vocabulary head.
- `unified_mot.py:2550` exposes that text-generation route for Nemotron.

The Phase-2 motion expert consumes the intermediate frozen reasoner
projections inside its custom joint layers. It does not generate an extra
reasoning transcript, apply vocabulary softmax, or use the framework's
top-k/top-p, repetition/presence penalties, EOS stopping or incremental
reasoner KV cache. Reading those previously uncovered functions therefore
does not reveal an omitted Phase-2 reasoning step.

Similarly, the Native Qwen `Qwen3VLForConditionalGeneration` wrapper at
`qwen3_vl.py:1221` implements language-model logits/loss and beam-expansion
helpers. The motion expert does not use that language-model loss.
Those functions have now been read, but their generation-specific masking,
cache-position and image-count logic must not be presented as active
BONES-only training behavior.

### Rotary precision: source difference, inactive in the inspected call path

The cached Native Qwen rotary implementation at
`cosmos_framework/model/vfm/vlm/qwen3_vl/qwen3_vl.py:356` computes its
frequency matrix from float32 operands without an explicit autocast-disabled
context. The pinned Edge Nemotron implementation at commit `d4599e2`,
`cosmos_framework/model/generator/reasoner/nemotron_3_dense_vl/nemotron_3_dense_vl.py:119`,
explicitly disables autocast around the same category of frequency/sine/cosine
calculation:

```python
with torch.autocast(device_type=device_type, enabled=False):
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
    freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * self.attention_scaling
    sin = emb.sin() * self.attention_scaling
```

This matters if an external caller surrounds Native motion inference with
autocast, but the inspected Native Phase-2 trainer, joint model, flow and
sampler do not introduce such an enclosing context for the motion forward.
The trainer's explicit BF16 autocast at `train.py:738` belongs to the
separate live WAN VAE encoder and ends before motion forward. Consequently,
this source distinction is not evidence of an active training precision
mismatch in the inspected route. No new rotary numerical test was run.

The already documented different rotary theta values remain real. This
pass does not remove that architecture difference or prove its quality
impact. The Edge snippet above is explicitly from the pinned source;
framework HEAD alone still does not establish a clean live worktree.

### F09: eager position fallback adds avoidable GPU synchronization

At `motion_expert_t2m_edge/model.py:223`:

```python
next_position = int(
    condition.get("next_position_id", int(text_positions.max().item()) + 1)
)
```

Python evaluates the second argument to `dict.get` before calling it.
Therefore `text_positions.max().item()` executes even when
`next_position_id` already exists. At line 203 the positions have been
moved to the model's CUDA device, so retrieving that scalar entails a
device-to-host synchronization. This occurs inside the per-sample packing
loop on every denoiser call, not only when position metadata is missing.

Both the normal text preparation and the image-conditioning preparation
provide `next_position_id`. An explicit key-presence branch could avoid
the unnecessary fallback reduction in those normal cases. No such change
has been applied, and its wall-clock contribution has not been measured.

Classification: a source-proven avoidable operation and potential
performance cost, not a demonstrated cause of floating or weak caption
conditioning. With a valid supplied position, it does not alter that
position, the attention mask, motion target, or optimizer-step count.
It should not be used to explain the quality gap at fixed training steps.

### Checkpoint scope and historical weight-integrity evidence

The now fully reviewed Edge motion-state helpers at
`motion_expert_t2m_edge/model.py:261` exclude `backbone.*` from the motion
checkpoint. Strict loading compares expected/supplied motion keys and
rejects missing or unexpected non-backbone keys. These helpers do not
silently substitute the frozen base with motion-checkpoint contents.

There is additional historical evidence relevant to F08:
`motion_expert_t2m_edge/config.py:155` records that on 2026-09-08 all 254
Yonsei frozen reasoner tensors matched the pinned source safetensors at
DCP precision. That comment must not be conflated with the narrower
current startup guard, and it must not be omitted when discussing the
integrity concern. This pass found the comment; it did not independently
reproduce its historical all-tensor comparison or perform a new one.
F08 remains a coverage/reproducibility limitation, not evidence of observed
corrupt or incorrectly converted frozen weights.

### Coverage closed in this pass

| Captured source | Tracked source review after this pass |
|---|---|
| Edge framework `model/generator/mot/unified_mot.py` | All 2,726 captured lines |
| Native framework `model/vfm/vlm/qwen3_vl/qwen3_vl.py` | All 1,511 captured lines |
| `motion_expert_joint_attention/cosmos_loader.py` | All 552 captured lines |
| `motion_expert_t2m_edge/model.py` | All 291 captured lines |
| `motion_expert_t2m_edge/edge_loader.py` | All 361 captured lines |
| `motion_expert_t2m_edge/layer.py` | All 309 captured lines |
| `motion_expert_t2m_edge/config.py` | All 343 captured lines |

Line counts include captured comments/blank lines. Coverage is review of
these source snapshots, not runtime validation of every branch or proof
of their exact historical H200 versions. No training, allocation,
checkpoint, framework code or Git state was changed.

The numerical NATTEN-versus-FlashAttention backward comparison and the
failed text-key-scale diagnostic remain pending explicit approval.
The complete Nano-versus-Edge audit remains in progress.


## F10: BF16 optimizer storage and attention-scale adaptation

Classification: confirmed precision limitation in the inspected training
implementation, with a newly articulated Nano-versus-Edge interaction
hypothesis. The limitation is shared; its differential quality impact is
not yet established. No optimizer, dtype, model weight or training run was
changed in this pass.

### Trace from trainable norms to the actual optimizer

Edge constructs fresh motion Q/K norms at
`motion_expert_t2m_edge/layer.py:172`, with default unit gains, and casts
the entire motion-block list to BF16 at `model.py:86`. The later
`EDGE_MOTION_IO_DTYPE=float32` branch applies to the motion heads and final
norm only, not these per-block norms. Thus the newer Nano-parity launcher
does not by itself change motion-block norm parameter precision.

Native creates fresh Q/K norms at
`motion_expert_joint_attention/mot_joint_layer.py:202`, casts the blocks
to the generator parameter dtype at line 219, and leaves norm gains at one.
The recovered historical Nano `config.json` explicitly has
`fp32_master=false` and `fsdp=false`. This is consistent with the earlier
checkpoint inspection: 28/28 Edge85k and 48/48 Nano200k motion-block norm
tensors were exactly one in BF16 despite nonzero optimizer moments.
It is not evidence that the working Nano run secretly trained these
parameters through an FP32 master-weight path.

Both relevant trainers use ordinary `torch.optim.AdamW`:
Edge `train.py:502` sets `fused=True`; Native `train.py:807` sets
`fused=not fsdp_sharded`. The installed optimizer source was newly read
from the Yonsei Cosmos environment rather than inferred from an API name.

In `torch/optim/adam.py:177`, the optimizer initializes both momentum
buffers with the same dtype as the parameter:

```python
state["exp_avg"] = torch.zeros_like(
    p, memory_format=torch.preserve_format
)
state["exp_avg_sq"] = torch.zeros_like(
    p, memory_format=torch.preserve_format
)
```

The fused call at `adam.py:845` groups the actual parameter, gradient,
momentum and step tensors by device/dtype. At line 879 it passes these
directly to `torch._fused_adamw_`. No persistent high-precision parameter
copy or residual-update accumulator is supplied. The constructor itself
has an unimplemented higher-precision-copy note at lines 108-111.

### FP32 internal arithmetic does not preserve sub-BF16 updates

The installed CUDA header
`torch/include/ATen/native/cuda/fused_adam_utils.cuh` was also newly
reviewed. Its `adam_math` receives parameter and moment arrays in
`scalar_type`, promotes local arithmetic to `opmath_t`, then stores
the results back to the original scalar type:

```cpp
opmath_t param = static_cast<opmath_t>(r_args[kParamIdx][ii]);
// ... update momentum and compute bias-corrected step ...
param -= step_size * exp_avg / denom;
r_args[kParamIdx][ii] = param;
r_args[kExpAvgIdx][ii] = exp_avg;
r_args[kExpAvgSqIdx][ii] = exp_avg_sq;
```

`ATen/OpMathType.h:25` maps BF16 arithmetic to `float`. This establishes
FP32 *internal update math*, not FP32 *persistent parameter storage*.
The stores at `fused_adam_utils.cuh:81` and lines 85-86 round parameters
and moments back to BF16. The included
`torch/headeronly/util/BFloat16.h` defines the seven-bit mantissa,
ordinary nearest-rounding conversion and its CUDA conversion branch.
There is no stochastic-rounding or error-feedback state in this update path.

The single-tensor and foreach alternatives were also read: they update
the original parameter tensors in place. Merely setting `fused=False`
does not introduce an FP32 master copy and is not a solution to this
storage limitation.

At a norm gain of exactly 1, the adjacent BF16 values are 0.99609375 and
1.0078125. The corresponding nearest-rounding midpoint distances are
0.001953125 downward and 0.00390625 upward. An illustrative Adam update
of magnitude 0.0002, the peak learning rate times a normalized update of
one, is below both thresholds and rounds back to one. These numbers are
an analytic representation example, not a new executed optimizer test.

Actual Adam update magnitude depends on its moments and epsilon; the LR
alone is not an absolute bound on every update. Nevertheless, the source
explains why nonzero gradients/moments can coexist with unchanged unit
gains. Repeating rounded-away parameter updates does not accumulate their
lost residual in an unseen FP32 master. The earlier checkpoint values
are direct evidence that the inspected gains did remain unchanged.

### Why a shared limitation can interact with different reasoner conditioning

For a fresh query norm with unit gain, the motion query is approximately

```text
q = RMSNorm(Wq h),              RMS(q) approximately 1
attention_logit = dot(q, k) / sqrt(128)
```

Multiplying all of `Wq` by a positive scalar approximately cancels under
RMS normalization when epsilon is negligible. The projection can still
learn directions, but simply increasing its overall weight magnitude
does not provide the same global attention-scale adjustment as changing
the norm gain.

The existing matched-checkpoint diagnostic measured larger text-key RMS
for Nano (about 1.75-3.87) than Edge (about 0.945-0.992) in the inspected
layers/cases. Its within-text attention was also less diffuse for Nano.
These are previously collected internal activation diagnostics, not new
R@3/FID measurements or proof of a universal distribution.

Edge reads a separately normalized frozen reasoner-key copy, while Nano
reads its pretrained Qwen-normalized reasoner keys. Their different
pretrained scales therefore interact with the fresh motion query norms.
If query gains cannot accumulate small learning updates, Edge may be less
able to adapt attention sharpness through that parameter channel.
This is a specific hypothesis connecting a shared storage limitation to
an observed backbone difference; it is not proof that it caused floating
or that raising a scale will improve semantics.

Several caveats remain:

- Attention depends on query/key directions, value content, token counts and all preceding layers, not RMS alone.
- The measured activation comparison covers only its recorded cases and checkpoints.
- Nano and Edge also differ in depth, width, frozen backbone and training exposure.
- Scaling text keys alone is not equivalent to scaling motion queries: the latter changes logits against both text and motion keys.
- Successful inference-time text-key scaling would not by itself prove that a norm-precision training change repairs learning.
- The original Edge video generator's learned query-norm scale has not been measured in this pass.

The earlier statement that unit BF16 gains are shared behavior remains
true. It must not be expanded into the stronger, unsupported claim that
their interaction with different frozen reasoner scales is irrelevant.

### Controlled follow-up, not an applied fix

The existing attention-backend forward/backward comparison remains the
first unresolved numerical correctness check. The separate text-key-scale
diagnostic still needs permission to fix its failed guard and retry.

If a norm-precision ablation is authorized later, it should be a separate
controlled fork with fresh motion-block norm parameters and optimizer
state kept in FP32, leaving frozen reasoner weights, data, statistics,
batching, loss and LR unchanged. It must demonstrate actual gain updates,
then compare held-out caption sensitivity and physical/retrieval metrics
with the same GT reference. Changing storage cannot reconstruct updates
already lost in an existing checkpoint's earlier training.

Do not blindly enable Native's `--fp32_master` flag as a cross-backend
fix: `train.py:624` casts all trainable parameter storage, not a separate
master copy, while the custom packed forward retains explicit BF16
buffers. A usable precision change needs explicit dtype-boundary handling;
the historical Nano run did not use this flag.

### Additional source coverage

The complete captured `mot_joint_layer.py` (566 lines), including its
inactive self-test code, was read without executing the test. The newly
captured installed files were fully read: `adam.py` (991 lines),
`adamw.py` (182), `fused_adam_utils.cuh` (206),
`ATen/OpMathType.h` (79), the forwarding `c10/util/BFloat16-inl.h` (7),
and `torch/headeronly/util/BFloat16.h` (481).

These installed headers supply concrete source evidence; no new GPU
kernel experiment or binary-to-source identity check was performed.
The overall source/history/causal audit remains incomplete.


## Data exposure audit: pool mapping, worker state and auto-resume

This pass completed the tracked review of the two shared dataset modules,
the Edge adapter, checkpoint helpers, and the installed index/batch samplers.
It separates nominal task/source probabilities from actual per-window
opportunities. No dataset was instantiated, no sampler was executed and
no training run was modified.

### F11: global index shuffling does not give uniform smaller-pool sampling

The active Phase-2 dataset is `NymeriaJointDataset`, wrapped by
`EdgePhase2Dataset` for BONES caption wording. It is not the older
`ConcatDataset`/`WeightedRandomSampler` path also present in
`motion_expert_joint_attention/dataset.py`. Neither current Phase-2 trainer
calls `build_joint_dataset` or `make_source_sampler`.

At `nymeria_joint_dataset.py:469`, the virtual dataset length is the
maximum live pool size, not the sum:

```python
n = max(len(self._index), len(self._t2m_index))
if self._bones is not None:
    n = max(n, len(self._bones))
return n
```

At lines 627 and 677, a virtual index maps into a chosen source/task pool
by modulo:

```python
b = self._bones[idx % len(self._bones)]
# ... Nymeria branch, including deterministic retry ...
it = pool[(idx + attempt) % n]
```

Let the virtual dataset length be N and a selected pool have n entries.
Before sampler padding, incomplete-batch dropping or load failures,
write N = q*n + r, with 0 <= r < n. In a full permutation of the virtual
indices, pool entries 0 through r-1 each have q+1 opportunities; the
remaining entries each have q. Shuffling changes their order, not these
multiplicities. For example, N=10 and n=4 maps to counts [3,3,2,2], not
uniform counts. This is an analytic example, not a newly executed test.

The first r entries come from deterministic manifest/index order, so this
can produce a content-correlated weighting rather than merely additional
independent noise. Deterministic retries can further redirect opportunities
to the next valid neighbor. Actual severity depends on live pool sizes,
ordering, failures and finite sampling; it has not been measured here.

This is narrower than a complete-prefix-exclusion bug. Because N is at
least as large as every live pool, all pool entries are reachable. It is
also not automatically a BONES-only problem: when BONES is the largest
pool and its routing probability is one, idx modulo len(BONES) is the
identity over a full virtual pass, apart from sampler tail handling.

The same max-pool length and modulo mappings are present in cached
pre-migration commit `f81b1aa`:
`nymeria_joint_dataset.py:460`, line 619 and line 669.
Therefore F11 is a shared sampling limitation, not evidence of an
introduced Edge/Yonsei data regression. Changing it now would change the
baseline training distribution and requires a deliberate ablation.

### Nominal mixture is not an epoch of every source

The router first chooses a task, then sends a T2M item to BONES with
the configured BONES probability. With task weights 0.75 T2M / 0.25 TI2M
and BONES fraction 0.5 within T2M, the nominal source/task probabilities
remain 0.375 BONES T2M, 0.375 Nymeria T2M and 0.25 Nymeria TI2M.
They are sample probabilities, not equal frame counts or equal loss mass.

Even if BONES determines N, a mixed-run virtual epoch does not necessarily
consume every BONES pair: some of its virtual indices route to Nymeria
instead. Conversely, smaller Nymeria pools have repeated opportunities.
Thus dataset epochs, optimizer steps, global sample counts, valid-frame
counts and unique source-window coverage are different quantities.

The earlier equal-global-batch caveat also remains: per-task losses reduce
valid frames inside rank-local task groups. Changing 8x32 to 2x128 does
not preserve every variable-length sample's exact loss coefficient.
Neither nominal mixture arithmetic nor matching data hashes proves equal
lifetime exposure after retries, restarts and changes in world size.

### Installed sampler behavior and the two drop-last decisions

The newly read installed
`torch/utils/data/distributed.py:107` creates a deterministic permutation
from sampler seed plus epoch. At line 134 it assigns strided subsequences
to ranks. Both current trainers construct it without overriding its
`seed=0` or `drop_last=False` defaults:
Edge `train.py:458`, Native `train.py:778`.

Consequently, if N is not divisible by world size, the distributed
sampler can pad with repeated indices to give every rank the same length.
Separately, both DataLoaders set `drop_last=True`, which drops an
incomplete rank-local batch. The installed `BatchSampler` at
`sampler.py:333` implements that second decision. It is incorrect to
describe both levels as dropping their tails.

For world size W and local batch b, the resulting batch count per full
iterator is floor(ceil(N/W)/b), before exceptional termination.
Neither padding nor dropping changes the fundamental smaller-pool modulo
multiplicity into uniform per-source selection. Their exact effects need
the live N, W and b, not only a run-directory name.

### Crop, padding and RNG scope

The shared source paths retain their documented differences:

- Nymeria T2M starts at the metadata caption-window start and uses min(T, available frames); it does not randomly move that window start.
- Nymeria TI2M uses prebuilt non-overlapping aligned subwindows, with 97 valid frames in the compared recipe.
- BONES first loads its metadata pair interval, then random-crops within that interval only when it exceeds T during training; validation uses the center crop.
- Both paths canonicalize the selected frame-zero motion, apply their configured shared mean/std, and center the neutral skeleton.
- `collate_joint` copies each item's existing motion mask rather than treating Nymeria's pre-padded rows as valid motion.

`make_bones` defaults its private crop/dropout RNG seed to 1. The
outer router has a separate RNG instance and uses the dataset seed.
With workers, each incorporates the worker's supplied seed.
The restored historical Nano configuration records four DataLoader
workers. There is no evidence here of all active worker processes using
one cloned RNG stream.

Separate instances do not guarantee different seeds in every possible
configuration: for example, a Native rank whose outer seed equals 1 can
match the BONES child seed. With zero workers, the BONES child seed also
is not automatically offset by rank. These are configuration-specific
correlation/reproducibility considerations, not a newly measured Edge
failure. The approximately 19% combined BONES caption dropout remains an
independence-based expectation for its two dropout gates, not an exact
empirical rate established by this source pass.

### Resume does not restore a precise data-stream cursor

Edge `checkpoint.py:32` captures process-global Python, NumPy, CPU Torch
and CUDA Torch RNG state. It does not serialize the dataset objects'
private `random.Random` instances, persistent workers, prefetched queue
or consumed batch cursor. This also matters with zero workers, because
the datasets use private RNG objects rather than the global Python RNG.

At Edge `train.py:772`, the process RNG is restored, the saved sampler
epoch is selected, and a new iterator begins. There is no advancement to
the previously consumed batch within that epoch. The global index prefix
can therefore be replayed on a mid-epoch DDP resume, while routing/crops
need not reproduce the earlier private worker stream.

Native begins each new iterator with sampler epoch equal to the current
training step (`train.py:2220`); Edge advances an explicit data-epoch
counter. Both are reproducibility differences already noted, now traced
through the installed sampler and checkpoint-state fields. The recovered
Nano configuration/log does not contain a full successfully consumed
window trace with which to reconstruct exact historical exposure.
No new claim of a harmful replay rate is made.

### F12: Edge auto-resume orders checkpoints by mtime, not saved training step

At `motion_expert_t2m_edge/checkpoint.py:18`, the function named
`latest_checkpoint` gathers regular step files and
`recovery_latest.pt`, then chooses:

```python
return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))
```

`train.py:400` uses this function for `--resume auto`. It does not
compare the saved step counters when choosing the candidate. An older
checkpoint copied or touched later can therefore outrank a newer training
checkpoint. This matters for housekeeping/transfer workflows that alter
modification times. Merely reading a checkpoint does not imply its mtime
changes.

The function also checks file existence, not payload completeness, while
selecting. Its normal writer is atomic (`checkpoint.py:117`), which
protects files written through that path; an externally partial file would
instead fail when loaded. No such corrupt file or incorrect auto-resume
was observed in this pass.

Native's `--resume auto` instead resolves the maintained
`<run>/latest.pt` path at `train.py:899`. That is a real operational
difference, but it is not evidence that the evaluated Edge100k checkpoint
was loaded incorrectly: the quantitative evaluation used an explicit
checkpoint path and required its saved step.

Classification: auto-resume selection/reproducibility risk, not a proven
cause of motion floating. Neither selection behavior nor any running
job has been changed.

### Coverage and remaining evidence

The full captured shared datasets (912 and 415 lines), Edge data adapter
(125), Edge checkpoint helper (158) and Native checkpoint adapter (109)
are now included in tracked source coverage. The installed distributed
sampler (158 lines) and sampler module (355) were newly captured and
fully reviewed. Dataset self-check blocks were read, not executed.

Remaining causal questions include the attention-backend backward
comparison and the frozen-key/norm-scale interaction. Their pending
approval status has not changed. The overall audit remains incomplete;
this source pass does not prove exact historical data exposure.


## Active Nano constructor, task routing, heads and decode

This pass closes the remaining captured Native top-level model branches
and newly captures the generator adapter. It separates code that exists
in the repository from code selected by the recovered working run.
Source inspection is not a new runtime test or proof of the original
H200 worktree revision.

### The successful Nano run did not secretly train the generator

The recovered, checksum-verified archive's `config.json` records:

```json
{
  "active_tasks": ["text2motion", "textimg2motion"],
  "textimg_condition": "reasoner",
  "coupling": "joint",
  "gen_lora": false,
  "reasoner_lora": false,
  "gen_full": false,
  "freeze_gen": false,
  "freeze_motion": false,
  "init_gen": null,
  "init_motion": null,
  "motion_layer_stride": 3,
  "motion_mrope": "cosmos3d"
}
```

In [joint_motion_model.py:539](/home/whwjdqls99/cosmos_motion_ft/motion_expert_joint_attention/joint_motion_model.py:539),
the trainability predicate only enables generator parameters when
`gen_lora` or `gen_full` is enabled. `freeze_gen=false` is NOT an instruction
to train the generator. Reasoner LoRA is also absent. The motion
predicate enables the private motion blocks, `heads.*` and
`norm_moe_motion`; the parameter iterator de-duplicates object identities.

The constructor creates fresh motion layers, fresh motion I/O heads and
a fresh final norm. Matching the generator norm's class/epsilon does
not copy its learned gain. With `coupling="joint"`, the bridge dictionary
is empty. The archived null warm-start paths also exclude a hidden
Phase-1 adapter or previously trained motion expert from this recipe.

The archive's 149-tensor resume message is consistent with the declared
motion-only topology, but is not by itself a strict expected-key proof.
The generic subset loader iterates supplied tensors and counts unknown
names/shape mismatches; it does not enumerate every expected-but-absent
target key. Do not equate that log's zero missing count with a complete
historical source/weight contract.

### Why the Nano generator adapter does not supply extra Phase-2 context

The generic task table represents TI2M's image as a present modality.
However, [joint_motion_model.py:929](/home/whwjdqls99/cosmos_motion_ft/motion_expert_joint_attention/joint_motion_model.py:929)
explicitly overrides its carrier:

```python
reasoner_textimg = (
    self.textimg_condition == "reasoner" and mode == "textimg2motion"
)
# ...
has_gen_for_sample = plan.has_gen and not reasoner_textimg
```

T2M has no generator modalities. TI2M with the archived `reasoner`
setting also has none. Therefore both active tasks pack zero generator
rows; their motion queries read reasoner plus motion rows, not an
additional frozen WAN/video-token branch. The reasoner-image embeddings
and Nano DeepStack additions remain active, as documented earlier.

[gen_heads.py:155](/home/whwjdqls99/cosmos_motion_ft/motion_expert_joint_attention/gen_heads.py:155)
only stores references and configuration in its constructor; it creates
no trainable adapter. Its video/image/camera encoders, timestep bias and
decoders are not called in these two active task routes. The
`gen_schedule="legacy"`, camera weight 10 and vision weight 1 in the
archive must not be mistaken for the active motion schedule or extra
Phase-2 reconstruction objectives.

The task defaults in
[task_plan.py:104](/home/whwjdqls99/cosmos_motion_ft/motion_expert_joint_attention/task_plan.py:104)
describe a wider seven-task mixture, not the archived explicit
75% T2M / 25% TI2M mixture. Both active task plans supervise all valid
motion frames and keep only the shape token clean. The experimental
half-weight two-target tasks are not active.

Two newly inspected generator-only details are recorded for scope,
not offered as explanations of this Phase-2 gap:

- The adapter prepends the old H200 `nymeria_world` import directory at module import. That is a portability/import-resolution hazard, not evidence that the working Nano run used different motion data.
- Camera encoding at line 308 detaches to CPU NumPy before padding. It would sever gradients into an upstream differentiable camera input, but active T2M/TI2M never enter this camera branch. It does not detach the Phase-2 motion prediction.

### Motion I/O, masks and decode do not hide another Nano mechanism

[Native motion_heads.py:136](/home/whwjdqls99/cosmos_motion_ft/motion_expert_joint_attention/motion_heads.py:136)
and [Edge motion_heads.py:12](/home/whwjdqls99/cosmos_motion_ft/motion_expert_t2m_edge/motion_heads.py:12)
use the same head topology, with their different residual widths:

- Motion input: linear 283-to-hidden plus a learned modality embedding.
- Shape: flattened 30-by-3 neutral skeleton through Linear, GELU, LayerNorm, Linear, plus modality and shape-type embeddings.
- Time: 256-dimensional cosine/sine features through Linear, SiLU, Linear, added only on noisy motion frames.
- Output: a zero-initialized hidden-to-283 linear head, including zero bias; decode returns FP32.

The Native optional AdaLN constructor flag defaults to false and the
working trainer does not pass an override. There is no active
AdaLN-zero gate missing from Edge.

Native pre-divides normalized sigma by 0.001 in the top-level model,
then multiplies by 0.001 inside the motion head. Edge embeds normalized
sigma directly. This agrees with the earlier actual time-input probe;
the different API spelling is not an extra 1000-fold time scale.

Native packs only valid frames, prepends one shape token per sample,
and at decode removes that token before scattering decoded frames back
to their original valid indices
([joint_motion_model.py:1384](/home/whwjdqls99/cosmos_motion_ft/motion_expert_joint_attention/joint_motion_model.py:1384)).
Padded output rows remain zero. This branch does not apply grounding,
centroid subtraction, translation integration, or contact correction.
Its index-copy path does not explicitly detach the predicted motion.
The later representation decoder/evaluator must still be considered
separately, as in the earlier loss and geometry audits.

The sampling closure captures the same skeleton, pad/noisy masks,
optional reasoner-image inputs and motion schedule for successive
denoising calls. The convenience method's default 50 steps does not
override the archived visualization setting of 35 or the explicitly
configured quantitative sampler.

### Timestep-frequency construction is part of reproducibility

The Native framework constructs its 128 sinusoidal frequencies in FP32
on each call. The Edge framework caches the same formula as
`_timestep_frequencies`, a buffer with `persistent=False`
([Edge modeling_utils.py:43](/home/whwjdqls99/cosmos-framework-edge/cosmos_framework/model/generator/mot/modeling_utils.py:43)).
Thus its values are not supplied by the saved motion state dictionary.

The current Edge model does:

```python
io_dtype = torch.float32 if config.MOTION_IO_DTYPE == "float32" else dtype
self.heads.to(device=self.device, dtype=io_dtype)
self.heads.time_embedder.to(device=self.device, dtype=torch.float32)
```

With the legacy BF16 I/O setting, this also rounds the floating-point
frequency buffer to BF16 before converting it back to FP32. Converting
back does not recover the discarded precision. With Nano-parity FP32
I/O, the initialized FP32 frequencies are preserved. Both paths finally
show an FP32 buffer, so final dtype alone does not establish identical
frequency values.

This is a source-derived construction difference, not a new numerical
measurement or a demonstrated quality cause. It also means that an
unrecorded constructor change to this nonpersistent buffer can change
a checkpoint's forward function without changing saved trainable
weights. Current checkpoint validation includes `motion_io_dtype`
and rejects a mismatched setting; no silent I/O-contract mismatch was
observed in the completed evaluation. Exact historical buffer equality
has not been established, and the new parity run's poor retrieval
cannot simply be assigned to the legacy BF16-buffer rounding path.

### Coverage added and open work

The captured Native `joint_motion_model.py` (1,574 lines),
`motion_heads.py` (312), `task_plan.py` (620), newly captured
`gen_heads.py` (499), and Edge `motion_heads.py` (81) have now been
reviewed in full. The Edge framework `modeling_utils.py` (125) is
covered in full; this pass covers Native's timestep embedder and
connector at lines 330-397, not the entire Native utility file.
Self-check/example blocks were read, not executed.

This closes source gaps and excludes hidden active generator/AdaLN
paths; it does not prove Edge correctness or identify the root cause.
Remaining work includes uncovered helper/history contracts and the
pending isolated attention-backend gradient and key-scale diagnostics.
No training, evaluation implementation, checkpoint, Slurm job or Git
state was modified.


## Sampling, checkpoint reconstruction and visualization artifact audit

This pass reviews the complete Native and Edge sampling/flow files,
Edge fixed-suite implementation and shared skeleton renderer. It covers
inactive legacy and multi-target branches too, but does not attribute
their behavior to the active T2M/TI2M benchmark.

### Active UniPC integration agrees; convenience defaults do not define the run

The active implementations are
[Native flow.py:466](/home/whwjdqls99/cosmos_motion_ft/motion_expert_joint_attention/flow.py:466)
and [Edge flow.py:66](/home/whwjdqls99/cosmos_motion_ft/motion_expert_t2m_edge/flow.py:66).
Both construct the official scheduler with constructor shift 1 and
dynamic shifting disabled, then set the ladder with the requested
shift 3. For each scheduler timestep both do:

```python
model_sigma = timestep.float() / num_train_timesteps
x0_hat = x0_null + guidance * (x0_cond - x0_null)
velocity = (x.float() - x0_hat) / scheduler_sigma
x = scheduler.step(velocity, timestep, x.float(), ...)[0]
```

Model conditioning uses the quantized timestep divided by 1000.
Velocity conversion uses the scheduler's unquantized FP32 sigma,
clamped below by 1e-6. Both operate on FP32 sample state in the
inspected benchmark; both clone supplied initial noise. There is no
extra shift, sign reversal, or conversion of a velocity prediction
as though it were x0 in these active paths. The official scheduler
implementations were already compared earlier.

The Native standalone CLI defaults to T=96, 50 steps and CFG 2.5;
the Edge CLI defaults to T=200, 35 steps and CFG 2. These are real
interface differences, not the settings used by the completed matched
benchmark. Native training visualization explicitly passes its saved
visualization settings, and the quantitative evaluator explicitly uses
35 steps, CFG 2 and native UniPC.

The Native loader restores objective, schedule, shift, sparse depth,
FFN width, motion mRoPE and image conditioning from checkpoint args.
Its old-checkpoint fallbacks include velocity/legacy scheduling,
Euler native solver, legacy mRoPE and generator-side TI2M images.
The recovered successful run records the relevant keys, so those
fallbacks should not be used to describe it. Its image size is also
explicitly 256 rather than the loader's 640 fallback for older
reasoner-image checkpoints.

Both TI2M conditional and null passes retain the identical image,
skeleton and motion masks. Only text is dropped. Native
`_sample_motion_target` returns normalized motion despite an inaccurate
high-level file comment suggesting otherwise; its CLI and the active
visualization caller each unnormalize it once.

A smaller source difference is confined to the duplicated
`native_inference_schedule` helpers: Native converts the NumPy
double-precision shifted ladder to integer timesteps before casting
sigmas to FP32; Edge first materializes FP32 sigmas and then multiplies
and casts to integer timesteps. Near integer boundaries these operation
orders need not agree. The active UniPC samplers do not call these
helpers: they obtain both arrays from the official scheduler. No
numerical boundary case was tested, and this is not an observed active
UniPC timestep mismatch.

### F13: fixed visualization inputs are not bound to their normalization statistics

[train_visualization.py:92](/home/whwjdqls99/cosmos_motion_ft/motion_expert_t2m_edge/train_visualization.py:92)
stores the dataset's already-normalized motion in each fixed NPZ.
The suite contract at line 57 records representation, frame counts,
image size, seed, task/source and split, but no mean/std hashes.
Reload checks the contract and caption hash, not hashes of the
motion, pad mask, neutral skeleton or conditioning image arrays.

Rendering later uses the *current* configuration's mean/std at
[train_visualization.py:341](/home/whwjdqls99/cosmos_motion_ft/motion_expert_t2m_edge/train_visualization.py:341)
for both generated features and the cached GT:

```python
gt_normalized = item["motion"][valid].to(device=device).unsqueeze(0)
gt_joints = decode_joints(gt_normalized * std + mean)
```

If a suite normalized with old statistics is copied into a run using
different statistics but an otherwise matching suite contract,
its cached GT is decoded as:

```text
z_saved = (x_gt - mean_old) / std_old
x_displayed = z_saved * std_new + mean_new
```

The model/checkpoint statistics check does not establish which
statistics originally produced an independently copied fixed NPZ.
Likewise, a valid replacement NPZ with the same prompt is not detected
by the prompt hash alone.

Classification: reproducibility/GT-comparison integrity gap. No such
copied or corrupted suite was observed in this pass. The cached GT
motion is not supplied as denoiser conditioning in T2M/TI2M; only its
valid length, the separate neutral skeleton, caption and optional
image are used. This therefore does not establish a cause of generated
floating or the matched quantitative retrieval gap. The quantitative
benchmark has its own manifest/statistics signatures and is not built
by this fixed-suite renderer.

### Nano checkpoint overlay checks are one-sided

[Native sample.py:1090](/home/whwjdqls99/cosmos_motion_ft/motion_expert_joint_attention/sample.py:1090)
copies matching supplied checkpoint tensors and only prints the number
it skipped. The quantitative wrapper improves this by rejecting an
empty state or supplied keys with missing names/wrong shapes
([phase2_quantitative.py:187](/home/whwjdqls99/cosmos_motion_ft/motion_expert_joint_attention/phase2_quantitative.py:187)).
However, that wrapper still does not compute the expected trainable
key set minus the supplied key set. Its `strict_model` name is stronger
than its actual missing-key check.

A nonempty, shape-compatible but incomplete delta can therefore pass
that particular check, leaving unspecified parameters at initialization.
This is a validation weakness, not evidence that the working Nano
checkpoint is incomplete. Its recovered motion-only configuration,
149-tensor load log and strong evaluated performance are separate
evidence; they do not change what this checker implements.

### F14: latent warm-start validation bug in the quantitative evaluator

At [phase2_quantitative.py:201](/home/whwjdqls99/cosmos_motion_ft/motion_expert_joint_attention/phase2_quantitative.py:201),
the warm-start guard calls:

```python
Path(sample.resolve_legacy_path(value)).is_file()
```

In the fully reviewed Native `sample.py`,
`resolve_legacy_path` is imported locally inside `load_joint_model`
at line 1013, not exported as a module attribute. A non-null warm-start
value therefore reaches an unavailable `sample.resolve_legacy_path`
reference rather than the intended existence check.

This is a source-identified evaluator bug, not a runtime failure
observed in the current benchmark. The evaluated Nano checkpoint has
both `init_gen` and `init_motion` null, so the faulty expression is
short-circuited. It did not produce the reported Nano advantage.
The evaluator has not been silently patched or rerun.

### Rendering preserves height; it is not a hidden grounding correction

The shared renderer copies the joints, negates X for display, and maps
Y-up coordinates to matplotlib axes. Its floor is always world y=0;
its tracking viewport moves in XZ only. Neither the Native active
visualization caller nor Edge's fixed-suite renderer subtracts a
generated centroid or per-frame foot minimum before rendering.

Both trainers compensate their frame stride with
`max(1, round(20 / frame_stride))` output FPS. At the archived stride 2
this is 10 FPS, not an accidental two-times playback-speed difference.
Independent tracking cameras can make positional drift less obvious
than a fixed shared camera; the world grid/root trails remain present.
The renderer requests a vertical view from zero upward, so videos
alone are insufficient to assess penetration.

Edge T2M generates T frames even when a fixed GT caption span is shorter,
then renders only their common prefix; full generated arrays are saved.
TI2M uses its actual aligned valid frame count. Thus a short MP4 can
omit a generated T2M tail. It is not an evaluation of that unseen tail.

These conclusions describe the captured current renderer and callers,
not an independently recovered H200 renderer revision. Quantitative
physics and retrieval use the arrays, not screenshots of these panels.

### Coverage and remaining scope

Full source coverage added: Native `sample.py` (1,230 lines),
`flow.py` (816), shared `render_motion.py` (437), Edge `sample.py`
(162), `flow.py` (128), and `train_visualization.py` (569).
No examples, renderer, model, numerical diagnostic or test was executed.
No training or inference implementation was changed.

The audit remains incomplete. These source gaps are closed, but exact
historical reconstruction and the pending attention-backend
forward/backward and text-key-scale diagnostics remain unresolved.


## Import binding, explicit site selection and documentation reconciliation

This pass closes the captured path helpers, launch wrappers, Native
configuration and several hand-off documents. It follows the selected
Edge path rather than interpreting every exported Nano compatibility
constant as an active Edge setting. No shell environment was sourced
and no import/behavior test was executed.

### Edge's shared data imports do not select a Nano model

[Edge config.py:167](/home/whwjdqls99/cosmos_motion_ft/motion_expert_t2m_edge/config.py:167)
loads Native's data configuration under a private module name, copies
uppercase constants only when Edge has not defined them, and then
explicitly rebinds shared mean/std and Nymeria representation paths.
It does not instantiate the Nano model. Legacy compatibility names
such as `HIDDEN` are not the Edge model's `HIDDEN_SIZE`; the inspected
Edge model/layer/head path uses Edge's loaded geometry.

The wrapper prepends Edge, joint-attention data code, Nymeria helpers,
the repository and the Edge framework in that order, then explicitly
selects the Edge DCP after sourcing the repository-wide Nano-default
environment. The Edge tokenizer is built from `config.EDGE_MODEL_ROOT`
at [edge_loader.py:152](/home/whwjdqls99/cosmos_motion_ft/motion_expert_t2m_edge/edge_loader.py:152).
The inherited repository-wide `COSMOS_TEXT_TOKENIZER_PATH` pointing to
Nano is not used by this Edge loader.

The quantitative entry point inserts its selected backend directory
at the front of `sys.path` before importing backend configuration,
model and flow modules
([phase2_quantitative.py:506](/home/whwjdqls99/cosmos_motion_ft/motion_expert_joint_attention/phase2_quantitative.py:506)).
This matters because its script itself lives in the Native directory.
There are no earlier backend imports in that entry point's reviewed
top-level code.

This is not a guarantee that arbitrary scripts can import both
backends sequentially in one Python process: already-imported
unqualified module names remain in `sys.modules`. The earlier
scratch-hosted Nano probe's import-shadowing failure and authorized
correction are already recorded above; it is not a new training fault.

The shared BONES factory's function defaults still reference the
original PoC statistics. However, the active `NymeriaJointDataset`
passes `config.MOTION_STATS_MEAN/STD` explicitly through `bk.setdefault`
at lines 452-463. Edge's selected camera/head statistics therefore
reach BONES as well as Nymeria. An isolated read of `make_bones` defaults
would incorrectly suggest mixed normalization in this active path.

### F15: site selection happens after some dependent defaults are constructed

[restored_env.sh:6](/home/whwjdqls99/cosmos_motion_ft/restored_env.sh:6)
first derives project roots, framework/env/Python paths, Hugging Face
hub/tokenizer paths and PATH/PYTHONPATH using environment values or
historical Grasp defaults. Its explicit site selector at line 30 then
overwrites the main roots but does not reconstruct every dependent
value built earlier.

For a hypothetical otherwise-unconfigured shell with only
`COSMOS_SITE=yonsei`, source evaluation leaves this inconsistency:

| Value | Result from the captured source |
| --- | --- |
| PROJECT_ROOT, WEKA_ROOT, RUN_ROOT | Yonsei Lustre roots after the selector |
| COSMOS_ENV_ROOT, COSMOS_PYTHON | Can retain the earlier Grasp defaults |
| COSMOS_FRAMEWORK_ROOT | Can retain the earlier Grasp default |
| HUGGINGFACE_HUB_CACHE, Nano snapshot/tokenizer | Can retain earlier derived cache roots |
| PATH/PYTHONPATH | Can retain components computed before the root override |

This table is a symbolic source trace, not output from running the shell.
The Edge wrapper pre-sets the Yonsei project/env and selects its own
framework afterward, mitigating several bare-source defaults. The
inspected contact and Nano-parity Yonsei launchers explicitly set the
project, env, Python, framework, model and DCP paths before invoking
the shared wrapper. This finding does not demonstrate mixed paths in
the evaluated runs.

There is a related split in
[runtime_paths.py:141](/home/whwjdqls99/cosmos_motion_ft/runtime_paths.py:141):
environment exports use `setdefault`, but later Python globals
`REPO_ROOT/WEKA_ROOT/RUN_ROOT/HF_HOME` are forcibly assigned the selected
site defaults. An explicit environment override can therefore disagree
with an imported Python constant. Framework/env globals are not
similarly rebuilt by this final block. In an Edge process the
repository-wide Python `BASE_CHECKPOINT_PATH` default remains Nano,
while Edge's own config reads the Edge-specific environment value.
The inspected Edge loader uses the latter.

Neither selector automatically recognizes the machine. Missing
`COSMOS_SITE` raises an error. Legacy-path translation recognizes the
listed original H200 prefixes and prefers an existing original path;
it is not a general Grasp-to-Yonsei or Yonsei-to-Grasp path translator.
Previously restored corpus/path evidence remains separate from this
generic portability limitation.

Classification: site/override consistency and reproducibility risk,
not an observed floating cause. No selector or launcher was changed.

### F16: two default task-mixture tables disagree

[Native config.py:107](/home/whwjdqls99/cosmos_motion_ft/motion_expert_joint_attention/config.py:107)
and [task_plan.py:104](/home/whwjdqls99/cosmos_motion_ft/motion_expert_joint_attention/task_plan.py:104)
both define default task weights, with different values:

| Task | config.TASK_WEIGHTS | task_plan.TASK_WEIGHTS |
| --- | ---: | ---: |
| text2motion | 0.40 | 0.30 |
| textimg2motion | 0.15 | 0.20 |
| video2motion | 0.15 | 0.15 |
| motimg2video | 0.08 | 0.10 |
| policy | 0.07 | 0.10 |
| forward_dynamics | 0.08 | 0.08 |
| inverse_dynamics | 0.07 | 0.07 |

The default trainer and dataset route from `config.TASK_WEIGHTS`.
The task-plan object's `weight` field and its printing self-check
reflect the other table. Both tables sum to one, so a sum-to-one check
does not expose this disagreement. The dataset docstring names
`task_plan.TASK_WEIGHTS` even though the implementation reads config.

The archived working Nano run records explicit 0.75 T2M / 0.25 TI2M.
The inspected Edge launcher supplies that same two-task dictionary.
Thus this is a real default/documentation disagreement, not evidence
that these runs secretly used seven-task sampling. Likewise, Native
config's 1/10/50 default loss weights are not the archived explicit
1/1/5 contact recipe.

### Handoff documents describe distinct generations and sites

The reviewed Native README and DESIGN_7TASK files are redirects to
`AGENTS_ALL.md`, not independent implementations of the task contract.
Their literal old H200 paths and the H200 SSH examples in Native
`run.sh` are historical text, not current Yonsei operating instructions.
Do not launch unscheduled work by following those old examples.

`PROVENANCE.md` explicitly dates its hand-off to the decommissioning
of the a3ultra server and separates the derived Nymeria/BONES trees
from raw, access-gated captures. It identifies the original shared
283-D statistics, per-source BONES statistics and separate C45
evaluator statistics as different artifacts. This is consistent with
the earlier recovered Drive metadata and corpus evidence; it does not
replace those checks with a path-name inference.

Specific documentation qualifications from this pass:

- `RESTORED_PATHS.md` is primarily a Grasp path reference. Its L40 jobs and status snapshots are historical, not live Yonsei job status.
- Its Nymeria pair-JSONL table calls those artifacts active Phase-2 inputs, but the current active `NymeriaJointDataset` builds T2M/TI2M windows from `manifest_path` and its `t2w_windows` at lines 345-424. BONES still uses its pair JSONLs. The old Nymeria pair files remain relevant to other helpers/PoC paths, not proof of the active manifest population.
- Its camera-RGB Y-up description repeats the producer-basis ambiguity already resolved earlier in this audit from source. Do not infer an extra coordinate conversion from that label.
- The camera/head opt-in shell helper and the Grasp document recommend starting without BONES because BONES has no synchronized camera orientation. That is a documented semantic caveat, not an automatic BONES disable switch. Yonsei's requested BONES50 ablation explicitly enables it.
- The parity launcher conditionally copies the baseline's fixed visualization suite, but also explicitly retains the same camera/head representation/statistics. That intended same-statistics copy is not evidence of the mismatched-statistics scenario in F13.
- The old baseline launcher leaves some optional knobs inherited from the environment; the parity launcher resets more of them explicitly. Run-directory names alone are weaker evidence than saved args/contracts.

The newly captured
[framework-patch README](/home/whwjdqls99/cosmos_motion_ft/external/cosmos_framework_patches/README.md)
lists H200 patches for Phase-1 LoRA action-head trainability,
optional adapter-only DCP saves, logging and a pixel-path experiment.
It does not describe another active Phase-2 motion loss or a hidden
reasoner adapter in the recovered motion-only run. The README alone
does not constitute a line-by-line review of its referenced patch
payloads; those remain a separate source-coverage item.

### Coverage and boundaries

Full review tracked for `runtime_paths.py` (178 lines),
`restored_env.sh` (70), Native `run.sh` (88), Native `config.py`
(279), Edge `run.sh` (72), contact Yonsei launcher (70), parity
Yonsei launcher (113), shared Edge training launcher (84),
`use_camera_head_v1.sh` (37), Native README (21), design redirect
(15), `RESTORED_PATHS.md` (300), `PROVENANCE.md` (216) and the
framework-patch README (43). The two small newly captured files are
the camera/head opt-in shell helper and framework-patch README.

No environment, imports, numerical behavior, training process or
new checkpoint was tested in this pass. The current audit document
is the only modified filesystem artifact. Exact historical source
identity, remaining dependency coverage and causal diagnostics remain
open; no root cause has been established.

## Original BONES/Nymeria producers, statistics and representation provenance

This pass closes source coverage of the six vendored UniEgo preprocessing
files, including their CLI/error paths. It is a source/documentation audit,
not a new conversion, import test, statistics rebuild or numerical experiment.
Existing corpus and finite-pair results are distinguished below from historical
claims in the hand-off documentation.

### What the original BONES producer actually preserves

The proportional producer is
`external/kimodo_uniego_scripts/soma_proportional_to_uniego.py`, not merely
the uniform `soma_to_uniego.py` driver. At lines 65-87 it reads already-computed
global rotation matrices, posed joints, contacts and the actor's neutral joints.
The 77-to-30 map is by skeleton bone name, imported from the uniform driver's
lines 60-66. No new FK, contact detection, grounding or camera alignment is
performed in this BONES conversion.

The essential proportional path is:

```python
grm = np.asarray(d["global_rot_mats"])[:, idx]
pj = np.asarray(d["posed_joints"])[:, idx]
fc = np.asarray(d["foot_contacts"])
nj = np.asarray(d["neutral_joints"])[idx]
# prop_npz_to_uniego:
ud = kimodo_to_uniego(kd, head_idx=SOMA_HEAD_IDX, n_foot=SOMA_N_FOOT)
ud["neutral_joints30"] = kd["neutral_joints30"]
```

The output at lines 99-104 is unnormalized FP32 `features`, passthrough
`foot_contacts` and per-actor `neutral_joints`. The uniform driver also
preserves whatever positions it receives, but does not save actor neutral
joints. Therefore the two drivers are not interchangeable for reproducing
the shape-conditioned training artifact.

`PROVENANCE.md:85` attributes the proportional tree to
`soma_to_uniego.py` generically. The more specific representation document,
lines 407-424, and the proportional producer agree on the distinct driver
and its neutral-joint output. This is a documentation attribution correction,
not evidence that the downloaded data were produced with the wrong script.
The source snapshot is associated with Kimodo hand-off revision `5e3daac`
in `PROVENANCE.md:26`; this reference alone does not prove the exact historical
interpreter, dependencies or working tree used for every converted file.

### Canonical-frame math and the meaning of the global features

`kimodo_to_uniego.py:171-254` infers joint count from the input and uses SOMA
head index 6 when called by either SOMA driver. Its default head index 15 and
211-D explanatory header refer to the separate 22-joint HumanML3D use case.
They are not the active SOMA settings.

The canonical frame has head yaw and head X/Z translation, with canonical
translation Y fixed to zero. The head's forward Z column is projected onto
XZ; if its projected norm is below 0.01, the pelvis forward column is used
(lines 199-212). The converter then stores:

```python
local_T = torch.matmul(tsfm[:, None], M)
delta[0] = cM[0]
if T > 1:
    delta[1:] = torch.matmul(tsfm[:-1], cM[1:])
```

The 283-D layout is 30 local transforms of 9 features, 9 canonical-delta
features, then 4 contacts. Each rotation is a continuous 6D matrix
representation, not Euler degrees. The first delta stores an absolute
canonical transform in the full source clip; later deltas are relative.
The training window helpers replace the window's first delta with identity.
They subtract the manifest ground offset from each local joint Y, not from
the canonical-delta translation.

The inverse at `kimodo_to_uniego.py:262-322` cumulatively composes deltas
and applies the resulting canonical transform to the local joint transforms.
It does not re-estimate yaw from the decoded head. The active differentiable
decoder in `motion_expert_joint_attention/decode_uniego_torch.py:17-51`
likewise uses column-based 6D Gram-Schmidt and cumulative composition.

Neither decoder enforces the original pure-yaw/zero-canonical-Y invariants on
arbitrary generated features, nor does it reconstruct positions by FK on the
conditioned neutral skeleton. Small erroneous delta components can therefore
accumulate; a shape prefix does not itself enforce rigid bone lengths. These
are shared representation properties, not a newly established Edge-only
regression. No new drift magnitude was measured in this pass. Also, exact
equality with the converter's imported historical 6D helper is not established
by the converter's comments alone; its dependency gap is recorded below.

### Nymeria and BONES intentionally have different preprocessing operations

`nymeria_to_uniego.py:132-188` must run SOMA-77 FK using the actor's neutral
joints because its source contains local rotations rather than BONES'
precomputed posed joints. It then uses the same name-based SOMA-30 mapping
and canonical-frame converter.

Default Nymeria conversion retains raw height. The optional `--ground`
branch subtracts a whole-sequence floor offset; it is not the documented
per-window training recipe. The saved `grounded` flag is derived from this
argument, so the CLI assumes its input is the documented raw proportional
tree rather than detecting and undoing arbitrary prior grounding.

Nymeria contacts are redetected at 20 fps using a sliding local-floor estimate:
the 10th percentile of minimum foot height over plus/minus 20 frames, a
0.10 m height threshold and 0.15 m/s speed threshold (lines 102-129).
BONES contacts are carried through unchanged. Nymeria also saves timestamps,
floor metadata and actor information (lines 202-214); these are not silently
replaced with BONES metadata.

These source differences predate the Edge implementation. They can matter
when comparing datasets, but do not explain why the same BONES data produce
a much larger caption-conditioning benefit with Nano in the matched probe.
The earlier all-sequence recanonicalization comparison remains the relevant
numerical evidence for preserved Nymeria body joints and contacts.

### F17: Archived preprocessing success messages are not integrity guarantees

The producer and statistics CLIs have inherited validation gaps:

- Converter writes are direct `np.savez`, not atomic replacements; existing
  output files are skipped solely on existence unless `--overwrite` is set.
- The conversion workers return an error string on exceptions. Their CLI
  summaries count errors but do not exit nonzero merely because some files
  failed. Successfully serializing NaNs also counts as conversion success.
- CLI round-trip checks use the first N sorted source paths, not a random or
  complete corpus sample. Their final PASS/FAIL is printed without a failing
  process exit code. Their maximum-error accumulators lack an explicit
  finiteness check; a NaN operand need not replace a previous finite maximum.
- The round-trip display is therefore not proof of finite features, complete
  conversion, or clean training windows. The historical 200-file random
  report mentioned in the MD is a separate artifact, not the CLI's default
  sample selection.

Examples are the proportional producer's lines 91-107 and 110-126 plus its
CLI summary, the uniform producer's lines 94-109 and 175-189, and the common
converter's lines 359-378 and 481-500. No runtime demonstration was needed
or executed to classify these as source-level validation gaps.

The current corpus inspection already checked NPZ integrity separately.
The earlier numerical scan found 679 BONES source files with nonfinite
features, but the separate interval-level audit found zero nonfinite
intersections across 585,312 training and 18,459 validation pairs.
Do not reinterpret F17 as evidence that active training consumed NaNs.

### What the vendored per-source statistics script does, and does not do

`compute_uniego_stats.py:29-46` reads the supplied clip list, converts each
feature array to float64, replaces only the full clip's first canonical
delta with identity, and accumulates frame count, sums and squared sums.
Lines 118-127 compute population variance and save FP32 mean/std:

```python
mean = ssum / count
var = ssq / count - mean * mean
var = np.clip(var, 0.0, None)
std = np.sqrt(var)
std[std < args.std_floor] = 1.0
```

This is frame-weighted, not clip-weighted. It does not reconstruct the exact
distribution of overlapping/random training windows or their repeated
frame-zero replacements. The default floor is 1e-6; replacing a constant
channel's std with 1 does not disable mean subtraction.

Important limits of this historical helper:

- It does not filter nonfinite arrays. A pre-cleaned input list is required
  to reproduce the documented BONES statistics without NaN contamination.
- It defaults to `--n-joints 22` and does not assert that feature width
  matches that argument. Running it on 283-D SOMA data without the documented
  `--n-joints 30` would overwrite the wrong 9 columns at frame zero.
- Missing or unreadable clips are skipped and counted; they do not by
  themselves fail the whole run. Width inference is from the first existing
  clip, rather than a complete schema/finiteness validation.
- This helper does not ground raw Nymeria windows, mix datasets, select the
  generator's configured statistics or choose the evaluator's statistics.

The representation MD explicitly says the proportional BONES per-source
statistics used 141,541 clean clips and `--n-joints 30` (lines 432-443).
It lists 142,220 total clips and 679 source-NaN clips, agreeing in count with
our independent scan. Count agreement supports the hand-off narrative but
is not a filename-by-filename or byte-level provenance proof.

Those BONES `Mean_uniego.npy`/`Std_uniego.npy` files must not be conflated
with the shared statistics selected by each model's training configuration,
or with C45 evaluation statistics. The previously reviewed active dataset
factory explicitly passes the configured shared mean/std into BONES loading.
No new active statistics mismatch was established here.

### Vendored-source completeness limitation

The common converter imports `humanml3d_to_kimodo` at lines 100-105 for
`matrix_to_cont6d`, `cont6d_to_matrix` and HumanML3D helpers. A direct source
read at the expected sibling path failed with "No such file or directory".
Targeted discovery of `external/` found the six vendored UniEgo files but
not that dependency. The SOMA drivers additionally require the separate
`kimodo.skeleton` package.

Thus the six-file vendor snapshot is not self-contained. This does not prove
a configured Kimodo environment cannot supply the dependency elsewhere, and
no import test was run. It also does not affect the active Phase-2 loader
merely reading already-converted NPZ features. Exact historical helper
coverage and reproducibility require the corresponding external Kimodo
source/environment, not an assumption based on a passing old converter log.

Historical MD instructions to SSH onto an H200/a3ultra host for CPU conversion
are not Yonsei execution instructions. All model/data/archive computation
continues to require an owned compute allocation on Yonsei, never the login
node.

### Coverage and consequence

Full vendored-file review now covers `soma_to_uniego.py` (194 captured lines),
`soma_proportional_to_uniego.py` (190), `kimodo_to_uniego.py` (506),
`nymeria_to_uniego.py` (335), `compute_uniego_stats.py` (144), and
`UNIEGO_REPRESENTATION.md` (447), including their CLI and failure behavior.
The shared layout (71) and differentiable decoder (70) were also connected
to the producer contracts.

This narrows the provenance/reproduction gaps but does not establish the
cause of Edge's weak caption-conditioned motion quality. No training,
checkpoint, environment, dataset or production code was changed; no new
numerical metric or GT metric was produced. The strongest unresolved
Edge-specific diagnostic remains the previously identified attention-backend
forward/backward comparison, which has not been run without approval.

## Archived framework patch payloads and historical attention-guard provenance

This pass reads the actual archived framework patch and all four files in
its `untracked/` payload, not only the patch README. It also inspects one
historical framework source file using read-only Git, as authorized by the
user. No patch was applied to either Cosmos framework checkout, and no
training or numerical diagnostic was run.

### Patch inventory and what it can affect

`external/cosmos_framework_patches/base_commit.txt` records framework base
`82f82293ffd8983651cd51d8191287da3973f534` and bundle generation on
2026-07-23. The 165-line `local_changes.patch` changes exactly six paths:

| Archived patch target | Operative change | Relevance to the successful body-motion Nano run |
|---|---|---|
| Framework README | Historical project description and links | Documentation, not execution |
| `checkpoint/dcp.py` | Optional substring-filtered LoRA/action-head DCP saves | Native camera Phase-1 saving; not the custom Phase-2 PT writer |
| `configs/base/config.py` | Register raw-pixel Nymeria camera experiment | Not the body-motion trainer entrypoint |
| `configs/base/defaults/callbacks.py` | Register TensorBoard callback | Framework trainer logging, not a hidden motion objective |
| `configs/base/defaults/model_config.py` | Add `lora_keep_trainable_modules` option | Camera heads alongside generator LoRA |
| `model/vfm/omni_mot_model.py` | Re-enable selected parameters after LoRA injection | Camera adaptation; no motion-attention kernel or activation edit |

No patch hunk changes the reasoner attention, motion expert, contact loss,
283-D decoder, motion sampler, or text-token preparation. This is a statement
about this captured bundle, not proof that every historical dependency or
uncommitted change anywhere on H200 was captured.

### LoRA trainability and partial-checkpoint semantics

The model patch's lines 150-161 re-enable `requires_grad` for names matching
the comma-separated `lora_keep_trainable_modules` setting after LoRA freezes
base parameters. The corresponding raw-pixel camera config explicitly uses
`action2llm,llm2action,action_modality_embed`. This prevents accidentally
freezing the pretrained 9-D camera-action heads during camera adaptation.
It does not add or initialize a 283-D body-motion expert.

The DCP patch's lines 64-80 deliberately select saved state by substrings:

```python
keep = ["lora_A", "lora_B"] + [
    m for m in os.environ.get(
        "SAVE_TRAINABLE_KEYS", "action2llm,llm2action,action_modality_embed"
    ).split(",") if m
]
return {k: v for k, v in sd.items() if any(s in k for s in keep)}
```

Despite its method name, this is not a general `requires_grad`-based
trainable-state export. It is a recipe-specific filter designed to survive
FSDP naming differences. Other trainable parameter families would need
explicit inclusion if this saver were reused. Saving is gated by
`SAVE_TRAINABLE_ONLY` equal to `1`, `true`, or `True`; the loading
template stays full (patch lines 85-96).

The recovered successful Nano config has `init_gen=null`,
`init_motion=null`, `gen_lora=false`, `reasoner_lora=false`,
`gen_full=false`, and active tasks T2M/TI2M at 0.75/0.25.
The reviewed custom loader constructs/materializes the bare network, loads
base weights and freezes it (`motion_expert_joint_attention/cosmos_loader.py:
119-126`). Its motion pathway is built separately. The reviewed custom
Phase-2 trainer/checkpoint route uses a monolithic PT payload, not this
camera-specific DCP save filter.

Thus the archived patch does not reveal a previously missed Phase-1
warm-start, generator LoRA, reasoner LoRA, or selectively saved body-motion
module in the successful Nano run. The separately documented permissive
PT overlay validation remains a distinct limitation; no actual incomplete
Nano checkpoint has been observed.

### Raw-pixel camera payload: actual settings versus stale descriptions

The payload
`untracked/cosmos_framework/configs/base/experiment/action/posttrain_config/world_camera_nymeria_nano.py`
is a four-task camera/video experiment: forward dynamics, inverse dynamics,
policy and image-to-video. Its "Camera-only Phase 2" header uses an older
phase naming scheme. It is not the successful T2M/TI2M body-motion Phase 2.

The config creates one task stream per camera task, with synchronized
task-per-step selection and disjoint rank sampling. It uses the framework
FSDP/config/trainer path. The companion TOML requests eight replicas,
100,000 iterations and 5,000-iteration checkpoint cadence. These are not
evidence of body-motion Nano optimizer or iteration settings; those must
come from the recovered body-motion run config/log already recorded above.

The payload contains useful documentation mismatches:

- Its opening description says action heads are reinitialized, but operative
  checkpoint settings skip only `net_ema.` and explicitly load/fine-tune
  pretrained camera heads (lines 159-167).
- The README/CLAUDE design notes describe an older Motion-Transformer bridge.
  The archived successful body-motion config and reviewed constructor select
  the joint-attention implementation, not that older proposed bridge.
- The camera config evaluates `NYMERIA_FULL_FT` with
  `bool(os.environ.get(..., ""))`, so even a nonempty value such as `"0"`
  selects full generator fine-tuning. This is an old camera-launcher
  robustness issue, not evidence it affected the body-motion comparison.
- The payload hardcodes the former H200 project path for its external
  camera dataset. It is not an independently portable Yonsei launcher.

No attempt was made to run this superseded raw-pixel pipeline. Its
finite-stream concern was already noted by the patch README and is not
newly established by inspecting this experiment config alone.

### TensorBoard callback is observational, not another loss

The 99-line `untracked/cosmos_framework/callbacks/tensorboard_log.py`
converts loss tensors to detached scalar values, reduces fixed-order
per-modality statistics across ranks, and writes on rank zero. It does not
replace the loss passed to backward or update model weights.

Its `*_active` curves use a nonzero-magnitude heuristic
`abs(s) > 1e-12` to count active ranks, rather than an explicit target mask
(lines 61-66). They are means over selected ranks, not sample/token-weighted
body-motion metrics. A genuinely zero active loss would not count as active.
The `loss_ema` field is a display smoothing operation, not model EMA or an
optimizer operation.

Consequently, these archived camera curves cannot be used to infer a hidden
body-motion loss scale, and the framework note `action_loss_weight=10.0`
in the payload CLAUDE document is not the custom motion loss's
`w_feat/w_joint/w_smooth = 1/1/5`.

### Stale architecture and server instructions are not current evidence

The payload `untracked/CLAUDE.md:26` says there is no Edge variant. That is
a historical statement contradicted by the actual Edge implementation under
audit; it must not override current source. Its generic dual-pathway
description is about the pretrained reasoner/generator, not a complete
description of the separately added body-motion pathway.

Its Hopper/FlashAttention-3, Ada/FlashAttention-2 and Blackwell/NATTEN table
(line 17) is an architecture-level expectation, not a runtime trace from
the successful H200 training. Availability, shape support and backend
guards still decide actual dispatch.

Likewise the payload's SSH instructions for former a3ultra nodes are
historical H200 operations notes, not authorization to use unallocated
Yonsei GPUs. No such commands were executed; Yonsei model/data computation
continues to require an owned compute allocation.

### Historical clarification of F07: the Nano FlashAttention-2 guard

Read-only inspection of this exact historical file succeeded:

```text
82f82293ffd8983651cd51d8191287da3973f534:
cosmos_framework/model/attention/flash2/checks.py
```

Its complete 124-line captured source equals the already captured current
Nano framework file. In particular, the variable-length prohibition is
already commented out in the July hand-off base (lines 79-88):

```python
# if is_varlen:
#     target_fn(
#         "Flash Attention v2 (flash2) varlen is banned due to instability. "
#         "Please choose another backend.",
#         exception=ValueError,
#     )
#     return False
```

This resolves an important historical ambiguity: Nano's permission to use
FlashAttention-2 for packed attention was not introduced only in the
current Yonsei checkout. It is present in the recorded historical base,
and no hunk in the archived patch changes this file.

It still does not prove H200 actually selected FlashAttention-2. The former
Hopper hardware could select FlashAttention-3 with the installed environment;
the historical runtime choice remains unobserved. The earlier actual
Yonsei dispatch inspection separately established current Nano FlashAttention-2
versus Edge NATTEN on the same RTX 4090. Neither historical source agreement
nor dispatch inspection proves numerical forward/backward equivalence.

### Coverage and remaining scope

Full review added: patch base metadata (4 captured lines), actual patch (165),
untracked camera experiment (212), TensorBoard callback (99), camera TOML (43),
and archived CLAUDE document (101). The historical FlashAttention-2 checker
(124) is separately recorded under its full framework revision.

This closes the previously identified framework-patch payload gap and
strengthens historical provenance for F07. It does not establish the entire
historical H200 worktree/dependency identity or an Edge quality root cause.
No framework patches, production code, training runs, checkpoints or data
were changed. No new numerical or GT metrics were produced. The audit
document is the only modified filesystem artifact in this pass.

## Root Nano loader and restored Kimodo representation dependencies

This pass resolves two previously open source-capture gaps: the root
`train_motion_ft.py` helpers imported by the successful Nano implementation,
and the non-vendored HumanML3D/6D helper needed by the original converter.
The original capture inventory did not contain the root file; its complete
current source is now captured and reviewed, along with its local
`motion_decode.py` import. No import, tensor comparison or training test
was executed on the login node or on a compute node.

### Which parts of the legacy root trainer are actually inherited

`motion_expert_joint_attention/cosmos_loader.py:60-68` imports
`SPECIAL_TOKENS`, `build_network`, `build_text_processor`, `freeze_all`,
`load_gen_weights` and `materialize` from the root trainer.
`materialize` also invokes `init_motion_heads`, and loading invokes the
root key mapper.

The root module's main entrypoint is guarded by `if __name__ == "__main__"`.
Its independent dataset, optimizer, training loop and loss are not run merely
because the successful joint-attention trainer imports these helpers.
This matters because the root file contains several superficially plausible
but incorrect explanations for the successful Nano run:

| Legacy root-only setting/path | Why it is not the successful 283-D Phase-2 recipe |
|---|---|
| `MOTION_DIM=369`, root motion I/O heads | Extra heads built on the frozen base; custom joint-attention motion heads are separate and 283-D |
| `TextMotionDataset` and synthetic fallback | Called by root `run()`, not the Nymeria/BONES joint dataset |
| Root velocity prediction / Kimodo FK loss | `forward_loss()` is not among the imported Phase-2 helpers |
| Root head LR multiplier 2 | Root optimizer groups, not the custom Phase-2 optimizer |
| Root `proc.tokenize_text` chat-template call | Root local `tokenize` closure, not `FrozenCosmos.tokenize` raw-token path |
| Root standalone sampling/visualization | Not the recorded joint-attention UniPC T2M/TI2M sampler |

The root build explicitly enables reasoner Q/K normalization, diffusion
Q/K normalization, tied word embeddings and MoT (`train_motion_ft.py:
269-308`). It constructs on meta, materializes in the requested dtype,
initializes the base and extra root heads, then the imported loader applies
pretrained weights. `FrozenCosmos` subsequently freezes all parameters and
sets eval mode before the separate motion pathway is constructed.

The root `motion_decode.py` import is a pure NumPy/Torch function/constants
module; it does not itself load a dataset or apply its legacy loss at import.
Its 369-D normalization uses `sqrt(std**2 + 1e-5)`, and its legacy objective
is weighted block Smooth-L1 plus FK L1, divided by the sum of weights
(`motion_decode.py:283-294`, `369-405`, `439-478`).
Those formulas must not replace the archived 283-D 1/1/5 plus contact
objective when claiming Nano parity.

### Historical source comparison of the active helpers

The recorded pre-migration project revision `f81b1aa` contains
`train_motion_ft.py`. Source comparisons, without importing or executing
either version, found these seven captured function blocks identical:

| Function | Current source lines | Historical source lines |
|---|---:|---:|
| `build_text_processor` | 249-268 | 244-263 |
| `build_network` | 269-310 | 264-305 |
| `materialize` | 311-318 | 306-313 |
| `init_motion_heads` | 319-334 | 314-329 |
| `_diffusers_to_net_key` | 335-373 | 330-368 |
| `load_gen_weights` | 374-419 | 369-414 |
| `freeze_all` | 551-555 | 546-550 |

The module-level location defaults did change: the old snapshot glob was
under `~/.cache/huggingface/hub`; current code derives it from
`HUGGINGFACE_HUB_CACHE`/`HF_HOME`. Both select the lexicographically first
snapshot, not the most recently modified snapshot. Current `QWEN_JSON`
also accepts an environment value; the custom loader can subsequently
replace it with its framework-root config path.

This is function-source agreement with a specific revision, not proof
that every H200 run segment used exactly that working tree, checkpoint
snapshot or installed library set. Different model shapes and initialization
paths also consume different RNG streams; source-level initialization
agreement is not a claim of bitwise cross-architecture initialization.

### F18: Native base loading does not enforce complete pretrained coverage

The root key mapper now explicitly maps the reasoner Q/K/V/output projections
and Q/K norms, after handling generator-specific names:

```python
n = re.sub(r"\.self_attn\.to_q\.", ".self_attn.q_proj.", n)
n = re.sub(r"\.self_attn\.to_k\.", ".self_attn.k_proj.", n)
n = re.sub(r"\.self_attn\.to_v\.", ".self_attn.v_proj.", n)
n = re.sub(r"\.self_attn\.to_out\.", ".self_attn.o_proj.", n)
n = re.sub(r"\.self_attn\.norm_q\.", ".self_attn.q_norm.", n)
n = re.sub(r"\.self_attn\.norm_k\.", ".self_attn.k_norm.", n)
```

These rules, including the comments about an older missing-map bug, are
already present in `f81b1aa`. Thus the comment about 216 originally
random-initialized reasoner-attention tensors must not be reported as a
current newly observed failure. The targeted history search returns the
initial tracked "native cosmos" commit for those additions; it does not
date a potentially earlier untracked fix.

There are nevertheless active loader robustness gaps
(`train_motion_ft.py:374-414`):

- Missing transformer shard paths are silently skipped.
- Tensor keys that do not map to a present, shape-compatible destination are
  counted as skipped, without failing the load.
- No expected-reasoner-key completeness or per-tensor finiteness assertion
  is performed by this function. Destination tensors already have initialized
  values, so absence is not automatically a meta-tensor error.
- The displayed `n_moe_gen_loaded` count checks name presence, not the
  shape predicate used for the actual copy. It is not proof that every
  counted tensor was copied.
- The custom `FrozenCosmos` caller does not inspect the loader's return
  values before freezing the network.
- Snapshot selection is module-level and glob-based; setting a differently
  named snapshot variable does not by itself override this helper's choice.

Therefore a successful `[load]` summary is insufficient evidence of
complete pretrained weights. This limitation exists in both the compared
historical helper and current Nano helper. No new actual missing shard,
random reasoner tensor or wrong snapshot was demonstrated in this pass.

This is distinct from the previously reviewed Edge DCP mapping/required-key
checks and from the custom PT adapter overlay's missing-expected-key gap.
A base-weight completeness check, an adapter completeness check and a
numerical activation comparison establish different things.

### The missing converter helper exists in the restored external snapshot

The repository vendor directory still lacks the sibling
`humanml3d_to_kimodo.py`, but the file was located and read at:

```text
/lustre/whwjdqls99/cosmos/weka/shape_aware_motion_eval_c45_20260715/code/kimodo_open/benchmark/humanml3d_to_kimodo.py
```

Its forward 6D helper concatenates the first two matrix columns
(lines 272-275). This establishes the same column convention as the
vendor converter's feature contract and the active 283-D decoder. There
is no newly found row/column transpose mismatch.

The helper also imports legacy HumanML3D `paramUtil` and quaternion
utilities after adding a hardcoded former H200 path (lines 165-180).
Locating this source therefore narrows, but does not eliminate, the
standalone-reproduction environment gap. It does not establish the exact
file/environment used for every original conversion, and no converter was
run.

The remaining 22-joint HumanML3D code handles chain-reset FK, root
half-angle recovery, carried velocities and 263-D round trips. Those
branches are not used to reinterpret SOMA-30 features during current
Phase-2 loading. Their degrees/half-angle and 263-D discussions must not be
mistaken for the 283-D rotation representation used here.

### Benchmark inverse versus runtime decoder: same convention, different epsilon

The restored benchmark helper's inverse uses:

```python
x = x_raw / x_raw.norm(dim=-1, keepdim=True).clamp_min(1e-9)
z = torch.cross(x, y_raw, dim=-1)
z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-9)
```

The restored runtime helper at
`code/kimodo_open/kimodo/motion_rep/uniego.py:75-82` uses division by
`norm + 1e-6`, as does the active
`motion_expert_joint_attention/decode_uniego_torch.py:17-23`.
Thus comments asserting that the benchmark and runtime helpers are
identical/exact inverses are too strong. Adding epsilon perturbs even
unit-length inputs; clamping only changes denominators below the threshold.

The runtime helper and active differentiable decoder both compose canonical
deltas and apply them to local transforms. Their storage implementation
differs, but their displayed Gram-Schmidt formulas and geometric convention
agree. No fresh numerical/gradient equivalence test was performed.

This is not a Nano-versus-Edge decoder difference: both custom Phase-2
backends use the shared differentiable decoder. It limits what a benchmark
round-trip PASS proves, but does not establish the source of the observed
large semantic or floating-quality gap.

### Do not interchange the three normalization interfaces

The restored `UniegoMotionRep` wrapper additionally supports
`feat_bias` scaling of canonical/contact std and normalizes with
`sqrt(std**2 + eps)` (runtime file lines 210-242). Its default eps is
1e-6. That wrapper is not the custom generator dataset's plain
`(features - mean) / std` path.

It is also not the C45 evaluator's representation wrapper:
the already captured `shape_aware_TMR/st_eval.py:25-50` constructs
`TMRMotionRep`. Its input API decodes unnormalized 283-D motion into raw
joints, then constructs/canonicalizes/normalizes evaluator features using
that evaluator representation (lines 108-132). The header's suggestion
to always use proportional statistics is too specific: generator output
must first be unnormalized with that generator checkpoint's actual
training statistics, which may be shared Nymeria or camera/head-aligned
statistics instead.

The three paths are consequently not interchangeable: legacy 369-D
Kimodo normalization, custom 283-D generator normalization, and C45
evaluator feature normalization each have their own contract. No new
active mean/std mismatch or changed GT score was established here.

### Coverage and remaining scope

Full current-source review added: root `train_motion_ft.py` (1,131
captured lines), root `motion_decode.py` (480), restored external
`benchmark/humanml3d_to_kimodo.py` (767), and restored
`kimodo/motion_rep/uniego.py` (243). The historical root file was captured;
coverage here is explicitly the seven compared helper blocks and constants,
not an assertion that its entire legacy main loop was newly reviewed.

This closes the root-loader capture gap and locates the formerly missing
6D helper. Historical runtime/source identity and causal attention-backend
diagnostics remain open. The audit document is the only modified filesystem
artifact. No model, checkpoint, environment, training job or dataset changed,
and no new numerical or GT metrics were generated.

## Remaining processor/model-helper ranges and inactive API failures

This pass closes the explicitly tracked gaps in Edge's native processor,
Nano's Qwen utility module and Nano's model-helper module. The installed
Transformers processor source was captured once to trace the active
TI2M chat-template call chain. This is source inspection only; the failures
below were not exercised, and no production file or training job was changed.

### F19: Edge video-only token-count helper has an unbound local

In the captured Edge framework
`cosmos_framework/data/generator/processors/cosmos3_edge_processing.py:
525-552`, `_get_num_multimodal_tokens` defines `merge_size` only inside
the image branch, then uses it in the video branch:

```python
if image_sizes is not None:
    # ...
    merge_size = images_kwargs.get("merge_size", None) or self.image_processor.merge_size
# ...
if video_sizes is not None:
    # ...
    num_video_tokens = [(num_patches // merge_size**2) for num_patches in num_video_patches]
```

A video-only call with a nonempty patch-count list reaches an undefined
local variable. When both branches run, the video calculation inherits the
image merge size rather than independently selecting the video processor's
merge size. The latter is not necessarily wrong when the configured values
coincide, but the method does not enforce that invariant.

This does not explain the current BONES-only T2M failure. T2M calls the
direct text-token/embedding path (`edge_loader.py:229-257`). TI2M supplies
one image, not a video. More importantly, the inspected installed
`ProcessorMixin.apply_chat_template` calls the processor's normal
`__call__` directly; it does not call this token-count estimation helper.
The active Edge bridge expands image placeholders using the actual
`image_grid_thw` in `__call__` (lines 462-470), not F19's estimates.

The separate video preprocessing implementation (lines 250-343) groups by
shape, resizes, normalizes, and packs raster patches with channel-last
within each patch. Its existence is not evidence that body-motion TI2M is
feeding video frames through that path. No video API test was run and F19
remains unpatched.

### Actual TI2M processor return type and pixel-budget propagation

The concrete call chain is:

1. `FrozenEdgeReasoner.encode_reasoner_image_text` makes a user message
   containing one RGB image and a caption, with min/max pixel area in the
   image content item (`edge_loader.py:259-337`).
2. The outer `Nemotron3DenseVLProcessor.apply_chat_template` normalizes the
   message structure and pre-resizes the image. It explicitly requests
   `return_dict=True` from its wrapped processor
   (`nemotron3densevl_processor.py:44-120`).
3. `NemotronNanoV3BridgeProcessor.apply_chat_template` extracts image-content
   pixel overrides and forwards `per_image_kwargs` (bridge lines 387-415).
4. Installed `ProcessorMixin.apply_chat_template` renders the prompt and
   calls the bridge with text/images (installed source lines 1675-1711).
   Its `return_dict` branch returns the complete `BatchFeature`.
5. The bridge processes images, creates grid/placeholder tokens, tokenizes
   text, checks special-token counts and returns image data plus token IDs
   (lines 417-523).

Thus the underlying HF default of returning only `input_ids` is not a
missing-dictionary bug in this active wrapper call. The outer wrapper
sets the required flag even though the motion-expert call site does not.

The outer wrapper resets extra `kwargs` to an empty dict at line 65, so it
is not a general forwarding API for arbitrary caller options. However,
the active min/max pixel budget lives in the message content, survives that
reset, and is extracted again by the bridge. The bridge declares
`per_image_kwargs` in its image-kwargs type, allowing the installed
`_merge_kwargs` to forward it rather than drop it as an unknown argument.

The installed regular `_merge_kwargs` copies default dictionaries
(lines 1270-1285). Its special-token count check compares the rendered
text's image/video marker counts with tokenizer IDs (1760-1775).
That check does not independently verify image quality, geometric alignment
or reasoner weights; it is a narrower structural check.

The remaining `add_assistant_tokens_mask` body in the outer wrapper was
also reviewed (lines 122-187). It is for supervising generated assistant
text. The custom body-motion Phase-2 path does not use it as its motion
loss mask. Its presence is not an extra hidden text/reasoner training loss.

### F20: Nano's generic causal-mask registry is not subscriptable

The captured Native framework
`cosmos_framework/model/vfm/vlm/qwen3_vl/utils.py:203-212` defines a
plain `AttentionMaskInterface` class holding only `_global_mapping`.
The former `GeneralInterface` inheritance is commented out. Neither
`__getitem__` nor an equivalent indexing implementation is present.

Both generic mask factories subsequently do:

```python
mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]
```

For a supported eager-mode request that reaches this statement, the
displayed object cannot be indexed (lines 272 and 319). Existing 4-D masks
and unsupported/custom backend identifiers can return early before it,
so this is not a claim that every use of the module fails.

This latent generic-text API problem is not the current custom motion
attention implementation. Nano's reviewed `mot_joint_attention.py` makes
two packed attention calls: causal reasoner-only attention and full
motion-to-reasoner/motion attention. Edge's custom attention module uses
its corresponding packed calls. Neither delegates those masks to
`create_causal_mask` or `create_sliding_window_causal_mask`.

The generic module's chat/system-prompt helper is likewise not the active
Nano T2M raw-token function. No extra system prompt, sliding-window mask or
dense causal-mask factory was discovered in that T2M path.
F20 was not executed or patched.

### Model-helper alternatives versus the active motion timestep

Nano's now fully reviewed `modeling_utils.py` also contains
`FlattenedSinCosPositionEmbedding` and `VideoRopePosition3DEmb`.
Their latent-video spatial grids, FPS modulation and temporal compression
are alternative video-position utilities, not the custom motion expert's
active native mRoPE position construction. Their 10,000-base constants
must not replace the actual reasoner rotary-base values in the architecture
comparison.

The active timestep embedders agree on 256 sinusoidal features, max period
10,000, cosine followed by sine, and a Linear-SiLU-Linear MLP
(Native lines 334-381; Edge model-helper lines 28-109). Native explicitly
casts timestep input to float; Edge's active motion call already supplies
FP32 sigma. No additional timestep scaling or activation-family discrepancy
was found at that boundary.

The previously documented precision difference remains: Edge caches
frequencies as a nonpersistent buffer, while Native reconstructs the FP32
frequency vector. Legacy Edge BF16-I/O initialization can round that buffer
before casting the module back to FP32; the new parity FP32-I/O path avoids
that round trip. This pass does not add a numerical measurement or establish
that issue as causal.

The helper `MLPconnector` in both frameworks is the same two-linear-layer
pattern around a configured activation. It is not evidence of an extra
motion self-attention layer, AdaLN path or secretly wider motion FFN.

### Coverage and limits

Full captured-file coverage now includes Edge
`cosmos3_edge_processing.py` (586 lines), Native Qwen `utils.py` (605),
Native `modeling_utils.py` (397), and Edge
`nemotron3densevl_processor.py` (188). This corrects the earlier outer
processor coverage ending at line 122: its assistant-mask tail is now
explicitly included.

The installed Transformers `processing_utils.py` was captured in full
(1,783 lines), but reviewed coverage is explicitly the constructor/call
boundary, kwargs merge, active chat-template path and token-count check.
Unrelated serialization, Hub upload and other methods are not claimed
fully reviewed by that capture alone.

F19 and F20 are source-level API failures outside the exercised custom
T2M/TI2M paths. They do not prove an explanation for the current metrics.
No new numerical or GT metrics were produced, no tests were run, and the
audit document is the only modified filesystem artifact in this pass.


## Native base-network, mRoPE and migration-source coverage reconciliation

This pass records full review of the previously uncaptured Native base
network and its dense Qwen configuration, and closes the pending mRoPE,
head-camera helper and modality-bridge coverage entries. It also repairs a
failed source capture of the migration document. This is a source audit,
not a numerical equivalence test or a training change.

### F21: a factory assignment is not the materialized configuration

The earlier root-loader section described the factory as enabling tied word
embeddings. That statement followed the assignment in
`train_motion_ft.py:275`, but the now reviewed config wrapper does not
forward that attribute. The earlier characterization should not be used as
evidence that the live embeddings are tied.

The actual current path is:

```python
# train_motion_ft.py:271-276
base_config = Qwen3VLMoTConfig.from_json_file(json_file=QWEN_JSON)
base_config.freeze_und = False
base_config.qk_norm_for_text = True
base_config.qk_norm_for_diffusion = True
base_config.tie_word_embeddings = True
base_config.use_moe = True

# Native unified_mot.py:248-258, simplified to its return
return self._full_config_cls(**self.config_dict)
```

The wrapper stores the JSON separately in `config_dict`. Its
`full_config` property reconstructs from that dictionary, not arbitrary
attributes subsequently assigned to the wrapper. Its `text_config`
property uses the nested JSON plus the explicit `text_config_overrides`
mapping (lines 260-313). The Q/K-normalization flags are explicitly passed
to the layer constructors (1862-1867), so those flags really do take effect.

The selected current
`qwen3_vl/configs/Qwen3-VL-8B-Instruct.json` explicitly contains
`"tie_word_embeddings": false`. The reviewed
`configuration_qwen3_vl.py:259-284` forwards that value to the full HF
config. The text-config constructor also defaults to false
(lines 167-212). Thus the root setter alone does not turn tying on.
Likewise, the wrapper assignments `freeze_und` and `use_moe` are not
a substitute for explicit freezing or architecture-family selection.

This is a concrete configuration/documentation trap, but not a newly
demonstrated motion-quality cause. `FrozenCosmos` explicitly freezes
the loaded network, and the custom Phase-2 forward consumes the input
embedding and transformer submodules, not `lm_head` vocabulary logits.
No weights were retied and no source configuration was changed. This
inspection establishes the current checkout's behavior, not the exact
unarchived H200 working tree.

### Actual Nano factory and reasoner connection

The current JSON independently confirms the dense reasoner's 36 layers,
4096 hidden width, 12288 FFN width, 32 query heads, 8 KV heads, 128 head
dimension, SiLU-family gated MLP, RMS epsilon 1e-6, RoPE theta 5,000,000,
and interleaved mRoPE sections [24, 20, 20]. These are explicit JSON values;
the config class's different fallback defaults are not the selected model.

Native `unified_mot.py:99-131` dispatches the dense Qwen MLP, RMSNorm
and rotary functions. The root factory chooses
`Qwen3VLTextForCausalLM`, not the sparse-MoE wrapper. Its
`PackedAttentionMoT` constructor makes distinct understanding and
generation Q/K/V/output projections and Q/K norms (430-500).
The dense layer creates distinct pre-attention and pre-MLP norms plus two
dense MLP pathways (868-909). There is no newly discovered extra
activation or sparse-router auxiliary loss in that factory path.

The custom training connection is not a call to the stock VFM forward:

1. `FrozenCosmos.__init__` builds, materializes and loads the network,
   freezes it, and keeps `net.language_model.model`
   (`cosmos_loader.py:117-153`).
2. `layer_submodules` exposes the frozen layer's actual projection,
   norm, MLP, rotary-function and attention-dimension objects
   (`cosmos_loader.py:459`).
3. `JointMotionModel` creates its custom layers from those objects,
   embeds each caption or supplied image/text input, and iterates those
   layers with the custom packed masks
   (`joint_motion_model.py:314-366`, 910-925, 1350-1379).
4. Motion rows use the custom motion final norm and 283-D decoder;
   the shape token is removed before frame decoding (1384-1403).

Consequently the stock `Cosmos3VFMNetwork.forward` packing, context
parallel helpers, optional vocabulary CE output and video/action/sound
decode branches are not hidden operations in this Phase-2 body-motion
forward. Their source was nevertheless reviewed in full. Video/action
timestep scaling there does not add another scaling stage to the custom
motion head.

The separate `generate_reasoner_text` implementation
(`unified_mot.py:1203-1839`) uses an autoregressive per-layer KV cache,
vocabulary logits, EOS handling, temperature/top-k/top-p and optional
token penalties. The inspected Phase-2 path does not call it. There is
no extra generated chain of reasoning or persistent AR cache to account
for the working Nano result. The separate TaylorSeer initializer also
does not activate a cache merely by being defined.

### Additional base-API limitations, outside the active Phase-2 path

These were found by reading previously unreviewed code; they have not
been executed or patched and must not be presented as the floating cause.

- The dense and sparse-Qwen `init_moe` methods claim special handling
  for missing understanding Q/K norms when those norms are Identity.
  Their actual else branch raises for every missing counterpart
  (`unified_mot.py:1881-1901`, 2017-2031). Only the Nemotron variant
  implements the advertised Q/K-norm exception (2145-2159). The inspected
  root loader does not call `init_moe`, and its Q/K norms are enabled.
- The autoregressive API accepts `attention_mask`, but text-only prefill
  does not forward it at all (1650-1651). Image prefill passes it to the
  multimodal-input helper, while `reasoner_forward` and its attention
  call have no padding-mask argument (607-677, 1278-1355). Variable-length
  padded generation therefore must not be assumed mask-correct simply
  because this API accepts the argument. This is not the custom packed
  T2M/TI2M attention path used in the comparisons.
- The stock VFM forward computes a guarded
  `vision_token_shapes` value, but then separately dereferences
  `packed_seq.vision.token_shapes` in the packer call
  (`cosmos3_vfm_network.py:1031`, 1053). A call with
  `packed_seq.vision is None` reaches an unguarded attribute access.
  That stock API is not the custom motion-only forward.

### mRoPE helper parity: do not confuse optional video coordinates with motion

The Native helper `unified_3dmrope_utils.py` and Edge helper
`data/generator/sequence_packing/mrope.py` were both reviewed in full.
Edge adds optional explicit temporal coordinates and an actual temporal
compression factor. Those options are not supplied by the active motion
position builder.

With `fps=None`, spatial reset enabled, a T-by-1-by-1 grid and zero
start-frame offset, both helpers construct the same integer coordinates.
For a positive-length motion segment whose supplied temporal offset is L:

```text
shape token:     (max(0, L-1), L, L)
motion frame k:  (L+k, 0, 0)
```

Call sites are `motion_expert_t2m_edge/model.py:17-39` and
`motion_expert_joint_attention/joint_motion_model.py:409-453`.
The Native helper's base-FPS values do not rescale this branch because
FPS modulation is disabled. The optional generator temporal margin is
not silently added by either motion helper.

This does not erase the previously documented TI2M call-site difference:
Native supplies the reasoner token count, whereas Edge can derive its
offset from maximum reasoner position plus one. For T2M's ordinary
monotonic text positions these agree; image grids can make them differ.
The helper formulas themselves are not a new T2M mismatch.

Edge's motion builder also calls the grid helper unconditionally,
whereas Native guards a zero-frame segment when constructing positions.
The helpers' final maximum operation does not support an empty grid.
Actual inspected training windows are positive-length; this is not
evidence that current batches are losing frames.

### Runtime Phase-3 head-camera alignment is not offline recanonicalization

The recovered successful Nano contact-run config explicitly records
T2M/TI2M, weights 0.75/0.25, `coupling="joint"`,
`textimg_condition="reasoner"`, and `motion_mrope="cosmos3d"`.
The head-camera-alignment fields are absent from that archived config;
they should not be reported as explicitly stored false/null values.

In the current implementation, the alignment option defaults off and is
rejected for active T2M/TI2M tasks. It is restricted to Phase-3 V2M/M2V
with `bridge_local` (`train.py:495-503`).
`JointMotionModel` loads calibration/statistics only when the option
is enabled (296-312), and constructs bridge modules only for
`bridge_local` (368-374). The inspected reconstruction of the archived
joint-coupling recipe therefore has no hidden bridge layers or head-camera
calibration loss.

The separately reviewed `head_camera_alignment.py` decodes head world
transforms, forms relative SE(3), and conjugates into the calibrated
camera frame with the lever-arm term. Its loss uses normalized
SmoothL1 translation error and a rotation chordal error, not a hidden
Euler-angle body-motion loss. The oracle actor-calibration loader
explicitly requires its test/oracle/leakage metadata; the ordinary
global calibration loader checks geometry but does not enforce those
oracle metadata fields. Neither calibration loader is called by the
disabled Phase-2 option.

`modality_bridge.py` defines extra gated attention with zero-initialized
scalar gates and local cross-modality masks. Same-modality blocks remain
full; its mask should not be described as eliminating every possible
clean/noisy dependency. Its source does not imply that these extra layers
were part of the working Nano Phase-2 model.

Offline recanonicalized Nymeria files and their camera/head-aligned
statistics are a different producer/data choice, already covered by the
data audit. Disabling the runtime Phase-3 feature does not undo that data
choice.

### Migration capture correction and precise provenance boundary

The earlier cached capture attributed to
`migration/SERVER_MIGRATION.md` contained a literal truncation marker
inside the C45-statistics row and then text from the neighboring site
document. Its recorded 421-line length was therefore not valid
single-file coverage. A recovery read, justified by that capture failure,
obtained the complete 350-line source. The previous mixed capture is
retained in the session for provenance, but is not authoritative source
text for this file. This is a capture/extraction correction, not evidence
of an unexpected on-disk edit.

The complete migration contract explicitly states:

- Native Phase 1 and the joint-attention trainer were validated on the
  H200 a3ultra machines (59-61).
- Drive and GCS both archive derived Nymeria and proportional BONES
  motion/283-D UniEgo; the optional uniform-motion upload was still
  active on the document's July status date (68-83).
- Shared generator 283-D arrays, proportional source 283-D arrays,
  and C45 evaluator 190-D arrays are distinct, with separate hashes
  (90-119). The bundle's `artifacts/generator` arrays are proportional
  source stats, not a substitute for either shared training stats or
  evaluator stats.
- The Phase-2 restore contract includes the exact BONES pair/index files
  and fourth-overview-caption policy (150-173).
- The historical 57-GB raw proportional cache was not archived because it
  can be regenerated from the verified motion tree (257-261).
- Selected run metadata archives retain configs/logs/numeric results, not
  checkpoint shards or media (220-223). The separately inspected Nano
  archive contained no Python or shell source.
- Raw Nymeria VRS coverage is explicitly incomplete/access-gated
  (343-349). A verified operational derived-data restore must not be
  called a complete raw-from-scratch backup.

The July cloud verification record is historical evidence, not a new
September remote check. Combined with the previously completed Yonsei
hash/corpus/window checks, it supports the recorded data lineage.
It still does not reconstruct every untracked H200 code/environment
change or prove equality of the old compiled attention backend.

### Coverage matrix after this pass

| Requirement | Authoritative evidence reviewed | Remaining gap |
|---|---|---|
| Data loading and normalization | Custom datasets, pairs/cache/captions/collation; producer/statistics code; existing corpus and finite-window reports; complete migration contract | Reconcile older multi-file captures with exact source boundaries; do not equate captured with reviewed |
| Architecture and activations | Custom Nano/Edge layers and heads; both base MoT implementations; Native VFM network and dense config; optimizer source and existing activation probes | Historical H200 runtime identity and numerical backend equivalence remain unproved |
| Reasoner conditioning | Active embedding/image helpers, custom packed masks, mRoPE, layer loop, existing caption-shuffle comparison | Weaker Edge caption response is measured; its causal mechanism is not established |
| MD / Drive / H200 provenance | Migration, provenance and upload records; restored data checks; recovered Nano run metadata | Metadata archive is not the exact original source/environment snapshot |
| Previously unseen differences | F21 config forwarding and the explicitly inactive base-API limitations above | None is shown to explain the observed large quality gap |
| Training unchanged | Source reads and this audit-document append only in this pass | No new numerical tests, patches, restarts or job cancellations |

Newly recorded full-file coverage comprises Native
`unified_mot.py` (2,202 captured lines),
`cosmos3_vfm_network.py` (1,177),
`configuration_qwen3_vl.py` (288), and its selected JSON (63);
Native mRoPE helper (195), Edge mRoPE helper (227),
`head_camera_alignment.py` (278), `modality_bridge.py` (141),
and the recovered migration document (350).

The attention forward/backward comparison and the repair of the failed
text-key-scale diagnostic still require explicit approval. No such test
was run here, no new generated/GT metrics were produced, and no causal
root-cause claim or whole-goal completion claim is made.


## Shared-loss, cache and recanonicalization-producer review

This pass closes additional line-coverage entries using the already
captured source, rather than reading the same physical files again.
The source comparison below is not a new training or numerical test.

### F22: recanonicalization reuse does not establish the requested calibration

`build_camera_head_aligned_uniego.py:70-94` validates an existing
output's NPZ member list, feature shape, timestamps, non-feature members
and feature finiteness. It does not compare the output Head rotation to
the requested calibration/camera target, verify body-transform
preservation, or check a content/calibration identity saved when that
output was built.

The reuse path accepts those limited checks and returns before running
the converter:

```python
# build_camera_head_aligned_uniego.py:123-126
if output_path.is_file() and not overwrite:
    existing = _validate_existing(source_path, camera_path, output_path)
    record.update(status="validated_existing", seconds=time.time() - started, **existing)
    return record
```

The final manifest nevertheless records the calibration supplied to the
current invocation (290-295). Reusing one output root after changing
calibration or camera rotations can therefore label old features with a
new calibration digest. Even finite original features with matching
metadata would satisfy these reuse checks; "validated_existing" is not
a semantic recanonicalization proof. The reuse branch also does not
perform the fresh-conversion source/camera FPS equality check at 142-143.

A second coverage limitation exists at 173-187 and 319-325:
a missing split file returns `status="missing"` without train/test
records, and the final full-build check only rejects a nonempty
`missing` list inside those records. A missing split file, or a
dictionary lacking the expected split lists, can therefore avoid the
intended full-production coverage failure.

These are real producer/provenance weaknesses, not evidence that the
current Yonsei corpus is stale. The previously completed independent
full-corpus comparison checked actual Head targets and preserved body
quantities; that evidence is stronger than the builder's reuse status.
No corpus was rebuilt, manifest rewritten, calibration changed, or
producer code patched in this pass.

### The fresh converter preserves world positions, not feature-channel values

Full review of `camera_head_recanonicalization.py` confirms that the
fresh conversion path is materially stronger than F22's reuse path:

- `decode_uniego` rejects empty/non-finite input, reconstructs every
  joint and accumulates canonical deltas in float64 (153-197).
- The calibration loader requires a train-split declaration and validates
  its SO(3) rotation (98-108); the declaration itself is not proof of
  which fitting population produced that file.
- Camera rotations undergo the explicit Aria-to-Kimodo world-basis change
  and calibrated local-axis conversion (288-307).
- Only Head joint 6's world rotation is replaced. The re-encoder receives
  the unchanged decoded world positions and copied non-Head world
  rotations; contacts are explicitly copied back (310-346).
- The new canonical frame is yaw-only with zero canonical Y translation.
  Nearly vertical Head forward axes fall back to the root forward axis
  (200-225). No full camera pitch/roll is inserted into the canonical
  trajectory by this encoder.
- Re-encoding changes local positions/rotations and canonical deltas
  together. "Preserved body" does not mean those raw feature columns must
  remain byte-identical.

This helper normalizes nondegenerate 6D axes without additive epsilon
(111-128), rejecting norms at or below 1e-12. The training decoder uses
the already documented norm-plus-epsilon formula. Their column convention
agrees, but the NumPy whole-sequence producer and differentiable training
decoder are not literally identical numerical implementations. The
existing measured reconstruction tolerances, not the word "lossless" in
a docstring, bound the demonstrated preservation.

The fresh CLI path checks exact source/camera timestamps, conditionally
checks FPS, saves each NPZ through a temporary file, and returns a failing
exit when conversion records contain errors. These safeguards distinguish
it from the earlier original-converter CLI limitations; they do not
repair the existing-output reuse gap.

### F23: the candidate-statistics utility does not reject non-finite candidates

`audit_motion_stats.py:148-153` correctly checks the old mean/std
shape, finiteness and positive std. However, the subsequently loaded
motion features and resulting candidate arrays have no equivalent
finite/positive guard before being saved.

```python
# audit_motion_stats.py:90-98
features = all_features[start:end].astype(np.float32)
features = canonicalize_frame0(ground_features(features, offset))
features_64 = features.astype(np.float64)
zmax = float(np.abs((features_64 - old_mean) / old_std).max())
old_zmax.append(zmax)
if args.max_old_z > 0 and zmax > args.max_old_z:
    continue
sum_1 += features_64.sum(axis=0)
sum_2 += np.square(features_64).sum(axis=0)
```

A NaN z-score does not satisfy `zmax > threshold`, so the optional
outlier guard is not a finite-data guard. NaNs can propagate to the
candidate mean/std; the later `std < const_eps` replacement does not
remove NaNs. The utility also accepts a nonpositive `--const-eps`,
which can leave constant-channel std at zero. These are source-level
failure conditions, not newly observed failures in the saved statistics.

The mean and std are saved separately with `np.save` before the atomic
summary write (162-166, 206-211). A versioned directory and refusal to
overwrite the currently selected stats directory are useful safeguards,
but they are not atomic publication of a complete mean/std pair. Reusing
another output directory or interrupting between saves must not be
assumed checkpoint-compatible.

The audited Nymeria corpus was finite, and the selected statistics have
already been checked separately. No new bad array or actual changed
training normalization was found here. The utility was not run or fixed.

### What population the head-aligned statistics actually summarize

The statistics utility selects usable captioned windows from the requested
sequence split, applies the exact floor-drop entries and per-sequence
floor correction, and accumulates each retained window's full span
(`audit_motion_stats.py:29-119`). It uses frame-weighted first/second
moments, not an equal-sequence average. Overlapping windows and repeated
caption windows count their frames again; it does not form a union of
unique source frames.

It also does not reproduce the entire mixed training sampler. In the
runtime Nymeria T2M path, `_load_motion` takes the first
`min(output_T, available_span)` frames and pads shorter samples
(`nymeria_joint_dataset.py:525-556`). TI2M uses the separate aligned
window index. The stats utility does not apply that T cap, the TI2M
sampling distribution, or a BONES mixture. Its population is a legitimate
declared normalization choice, not proof that each channel has exactly
zero mean/unit variance under every training mixture.

The recanonicalization document explicitly records
`--max-old-z 0` for the matching stats: the old-z outlier exclusion is
disabled in that documented build. It also initially recommends
`--bones_frac 0` because BONES lacks camera-aligned Head semantics.
The user-requested 50% BONES-within-T2M run intentionally goes beyond that
initial recommendation; it is not a hidden dataset omission. This leaves
a semantic population difference to evaluate, not a demonstrated
geometric corruption of BONES.

### Shared contact loss and historical identity

`contact_aware_losses` is a function in
`motion_expert_joint_attention/motion_losses.py`, not a missing module
named `contact_aware_losses.py`. Edge imports that helper at
`motion_expert_t2m_edge/losses.py:9`; Native imports it at
`motion_expert_joint_attention/train.py:77`.

The complete current helper equals the already captured
`f81b1aa:motion_expert_joint_attention/motion_losses.py` source after
trimming trailing whitespace. This establishes source identity at that
reference, not the entire historical trainer/runtime identity.

The fully reviewed helper confirms the following loss contracts:

- BCE uses raw unnormalized contact predictions, logits
  `2 * (contact_prediction - 0.5)` at the actual default/launch setting,
  and positive-class weighting derived from the selected contact mean.
- Physical masks use GT contacts, so predicting all contacts off cannot
  disable physical supervision.
- Horizontal velocity is in m/s, uses adjacent valid frames, and is
  gated by the preceding frame's contact flag. It does not require
  both endpoints to have contact.
- Foot-height reconstruction uses raw local-pose Y channels, not decoded
  world Y. The previously demonstrated global-Y/tilt blind spot remains
  shared by Nano and Edge; this is not a newly found Edge-only loss.
- Masked means normalize over the selected elements after expansion.
  Matching this helper does not erase the already documented outer
  pooled/per-task reduction and precision differences.

The H200 contact launcher is now also fully range-recorded: eight ranks,
batch 32 per rank, the shared-stat SHA checks, T2M/TI2M 0.75/0.25,
BONES fraction 0.5 within T2M, and 1/1/5 plus 0.05/1/10 losses are explicit.
Its H200 paths, partition and GPU-memory preflight are historical launch
requirements, not a launcher to execute unchanged on Yonsei. No launch
or preflight was run.

### The existing attention self-test does not validate the actual GPU backend

The full `mot_joint_attention.py` source, including its test tail,
was reviewed. Production uses two per-sample varlen calls: causal
reasoner-only Q/K/V and full motion/generator queries over all keys/values.
The offset builder asserts matching list lengths and
`und + full == total`, then builds per-sample cumulative offsets.

The `__main__` self-test replaces the module's real attention function
with a Python CPU reference (206-270). Its concrete test tensors use
one sample, two query heads, two KV heads, and head dimension four
(282-305). It compares forward outputs against a dense mask and tests
forbidden/allowed information flow by perturbing values.

Accordingly, even a historical PASS of that test would not prove:

- the real NATTEN or FlashAttention kernel is correct;
- grouped-query head expansion is exercised by the concrete test case;
- multiple packed samples cannot leak into one another;
- BF16 numerics or the backward pass match the reference.

The reference helper contains GQA support, but the actual test case uses
equal Q/KV head counts. Support code is not the same as exercised coverage.
No self-test or new backend comparison was executed here. This strengthens
the case for the still-unapproved actual-backend forward/backward diagnostic;
it does not demonstrate a kernel bug.

### Cache/caption boundaries and source-capture accounting

The raw motion cache returns writable float32 copies of each requested
slice and neutral skeleton, whether read from a read-only NPY mapping
or the NPZ fallback (`motion_array_cache.py:121-132`). It does not
normalize, ground, recanonicalize or sample windows. Per-process
PID reset and bounded LRU references prevent forcibly closing mappings
still used by a caller (77-118).

Its identity uses relative source path, size and nanosecond mtime, not
source-content hashing. Matching metadata is a freshness heuristic, not
a proof that bytes cannot have changed while preserving size/mtime.
The prior independent cache/content comparison is stronger evidence for
the inspected corpus. Publication is staged/atomic; runtime cache errors
fall back to NPZ. The documented free-space check is filesystem capacity,
not Yonsei user quota.

The caption policy's full review confirms that ambiguous/role-noun rows
are retained, not silently dropped. Only a recognized primary subject
is rewritten; action wording and later coreference are not globally
rewritten. Subjectless/plural/unrecognized phrasing can therefore remain
a dataset cue. This matches the earlier recorded policy distinction and
does not by itself explain the measured caption-response gap.

The two camera helper files were also fully range-recorded. Their
camera-only four-task dataset and standalone CLI are not an extra task
mixture inside the custom body-motion dataset. The existing retry-handler
restoration limitation remains; this pass did not alter it.

Finally, the original 57-entry source inventory contains a failed
`cat` result for `motion_expert/dataset.py`, previously labeled as
three captured lines. Those lines are an error message, not Python
source, and must not be counted as file coverage. The active paired
loader is the separately captured
`motion_expert_joint_attention/dataset.py`. The recovered migration
capture is clean; the remaining stored entries have no visible embedded
file/truncation markers. That narrow check is not proof of complete review
or current/historical runtime identity.

Full coverage entries added here: shared motion losses (87 captured
lines), motion-array cache (237), joint-attention primitive (386),
caption policy (221), camera RGB helper/dataset (235), camera-action
helper (155), recanonicalization builder (331), pure conversion helper
(382), candidate-statistics utility (217), recanonicalization MD (416),
image-cache MD (91), and H200 contact launcher (94).
The complete recanonicalization comparator and remaining initial
evaluation/provenance captures still need their range-coverage
reconciliation; they are not declared fully reviewed by this paragraph.

Only this audit document was modified. No new metrics, model/data
computations, tests, source fixes, environment changes, or Slurm actions
were performed. The root cause remains unproved.


## Recanonicalization-comparator evidence scope and archive checklist reconciliation

The complete 1,069-line
`compare_camera_head_aligned_uniego.py` has now been reviewed, including
its resumable-shard path, fresh computations, aggregation, plotting and
CLI failure handling. This pass inspects the checker itself; it does not
rerun the dataset comparison.

### F24: a reused metric shard is not bound to its current inputs

The reuse branch checks that all declared metric names exist and reads a
stored JSON summary:

```python
# compare_camera_head_aligned_uniego.py:305-312
if shard_path.is_file() and not overwrite:
    with np.load(shard_path, allow_pickle=False) as shard:
        if not all(metric in shard.files for metric in ALL_METRICS):
            raise ValueError("existing shard is incomplete")
        summary_text = str(np.asarray(shard["sequence_summary_json"]).item())
    summary = json.loads(summary_text)
    record.update(status="validated_existing", seconds=time.time() - started, **summary)
    return record
```

The shard path is derived from the output directory and relative sequence
name. This branch does not bind the shard to old/new motion bytes, camera
bytes, calibration/lever values, floor offsets, selected physical windows,
or metric implementation. It does not recompute feature preservation or
check metric array lengths/finiteness against the current inputs.

Consequently a new report can combine current path/selection metadata
with metrics from an earlier invocation if the output directory is reused
after changing those inputs. The documentation's instruction to overwrite
shards when source/schema changes is important; `validated_existing`
does not automate that invalidation.

This limitation does not invalidate the recorded fresh Yonsei comparison:
its cached report explicitly has `record_status_counts={"computed":728}`,
not reused shards. It was created on 2026-09-14 at 05:59:02 UTC and records
the Yonsei old/new roots and scratch camera root. That is the run already
cited earlier in this audit, not a new run in this pass.

### What the fresh comparison actually checks

For newly computed records, the checker loads the actual old/new NPZs and
camera sidecar, checks NPZ member sets and timestamps, and rejects changed
non-feature members (314-342). It then decodes both feature arrays and
compares every frame's joint positions, foot positions/heights, bone
lengths, non-Head rotations and corrected Head target (344-443).
Computed metric arrays must all be finite before shard publication
(505-506). Contacts inside the feature array are compared separately
(513-518), not assumed equal just because other metadata matches.

The recorded result covers 642 train, 71 test and 15 other sequences with
no metadata/contact failures. These explicit report values support the
preservation conclusion; an exit status alone would not.

There are two important limits on the word "independent":

- This is a separate check of the saved output arrays, not merely a
  repetition of the builder's success flag.
- It imports the same NumPy `decode_uniego` and related rotation
  helpers as the producer. It is not an independently implemented
  decoder check and does not exercise the differentiable PyTorch
  training decoder or its GPU precision/backward path.

The existing decoder/source/runtime diagnostics address different parts
of that boundary. The NumPy comparator by itself must not be treated as
proof that all implementations have identical numerical behavior.

Both camera-origin proxies use the original decoded Head position,
combined with old versus corrected Head rotations and the same calibrated
lever (351-361). This deliberately isolates the rotation/lever effect.
A material change in the new Head position would be omitted from that
proxy but would appear in the separate decoded-position preservation
metric. That metric was small in the recorded run; the camera-origin
error increase is therefore not evidence that the physical feet moved.

The old and new physical-window calculations also deliberately use the
same old GT contact masks (484-490). A corrupted new contact channel
would not necessarily change those paired physical metrics, but the
separate contact-equality flag would expose it. The recorded flags were
all preserved.

### Successful report generation is not a complete quality/coverage gate

The CLI fails on missing old/camera inputs and on sequence-computation
exceptions (913-919, 963-966). However:

- It enumerates the corrected tree, rather than requiring every original
  auxiliary sequence to have a corrected counterpart.
- It records expected split counts but does not assert their equality
  with the successfully aggregated train/test counts.
- Contact-equality failures are collected in
  `metadata_or_contact_failures` and printed; they are not a final
  nonzero-exit condition.
- Numeric preservation thresholds are summarized, not enforced as
  hard pass/fail tolerances.

A quantitative comparator can legitimately report imperfect data.
The practical consequence is narrower: "command exited zero" is not
equivalent to "all preservation/coverage checks passed." Here the
recorded coverage, failure list and numeric values were inspected
separately, so the preservation conclusion is not based only on exit zero.
No new missing sequence, contact mutation or preservation failure was
discovered by this source review.

### Correction: physical aggregates are window-weighted

The earlier paragraph saying means/extrema cover every frame was too broad
when applied to physical-window metrics. That wording has been corrected.

Frame and transition metrics contribute their complete per-frame or
per-transition arrays to the streaming accumulator. Physical quantities
are first reduced to one number per caption window
(`_physical_window_metrics`, 135-216), and those numbers are then
averaged. Thus a long window and a short window have equal weight in
those reported physical means. Repeated/overlapping caption windows
remain repeated observations; this is not a union of unique source frames.

Windows with no contacts contribute zero to the contact-height/floating/
penetration entries, and the contact-skate denominator has an epsilon.
Such zeros must not be interpreted as demonstrated correct foot contact
in that window. The all-feet-above threshold is descriptive because
legitimate jumps can be airborne.

The comparator's physical-window selector mirrors usable captions,
floor-drop entries and calibrated offsets (631-690), but does not
reproduce the runtime T cap, aligned TI2M sampling distribution or BONES
mixture. It is a source-preservation comparison, not the exact distribution
of training batches or the generated-model evaluation cohort.

`_StreamingMetric` sums all stored values in float64, while shards
store metric arrays as float32 (526-533, 544-614). "Exact" in those report
field names means all stored observations rather than sampled quantiles;
it does not mean unlimited numerical precision. Quantiles sample every
stride-th value separately within each sequence, and physical quantiles
therefore sample windows, not frames.

The CDF plot drops values beyond a high-percentile display limit and
rescales the displayed subset to reach one (780-785). It is consequently
a truncated/renormalized visual CDF, not a display of the complete tail.
The full-population numeric mean/extrema are not recomputed from that
clipped plot. Preservation histogram floors and plot limits likewise do
not clamp the numeric report.

### Comparison with the restored Kimodo skating implementation

The complete restored `kimodo/metrics/foot_skate.py` was reviewed.
Its per-sequence height-based toe speed, contact-weighted four-foot speed,
and below-floor-at-both-endpoints skating ratio agree with the comparator's
respective per-window formulas when joints, contact masks, FPS and valid
lengths agree. The comparator slices valid windows directly; Kimodo
applies length masks.

These are 3-D speed metrics. They are not identical to the training
helper's squared horizontal planted-foot velocity loss. A shared metric
name or loss weight must not erase that distinction.

Both height-gated and predicted-contact-gated skating metrics can report
zero when their gating set is empty. They should be read with the
separately reported contact/floating/penetration measures, not taken as
standalone proof of grounded motion. The source-only comparison did not
produce new model or GT scores.

### Upload MD records: consistent active data, one historical status discrepancy

`UPLOAD_MANIFEST.md` and `UPLOAD_PATH_MAP.md` were reviewed in full.
They agree that active derived Nymeria and proportional BONES/UniEgo,
critical split/floor/pair files, evaluator state and selected Phase-2
checkpoints were archived. They also distinguish Git-tracked small
source/statistics from Drive data and GCS runtime/source bundles.

The path map explicitly identifies
`cosmos_ckpts/ja_phase2_t2m_ti2m_contact_unipc35/ckpt_step200000.pt`
as the contact-loss Phase-2 checkpoint. The original Phase-1 compact EMA
file is not a full resumable checkpoint, and raw Nymeria VRS remains an
incomplete/access-gated archive boundary.

There is a coarse-date status discrepancy for the optional uniform
BONES tree: `UPLOAD_MANIFEST.md:19` says verified, whereas the July
migration contract says its upload was still active. These human-readable
status snapshots do not by themselves settle the exact completion time.
Both agree on the active proportional BONES tree. The discrepancy is not
evidence that current proportional training data is missing. No cloud
listing, authentication change or download was attempted here.

New full-file range coverage: recanonicalization comparator (1,069 lines),
restored Kimodo foot-skate metrics (255), upload checklist (96), and upload
path map (59). Remaining initial evaluator/provenance captures still need
their explicit coverage reconciliation. The attention forward/backward
comparison and failed text-key-scale diagnostic repair remain unapproved.

Only the audit document was changed. Existing metric values were not
modified, no tests or model/data computations were run, and training jobs
were left untouched. The implementation-quality root cause remains open.


## Evaluator boundaries, archived launchers and smoke-evidence coverage

This pass closes additional full-source coverage using the existing source
captures. It does not run another evaluation, numerical probe or training
step. Findings below distinguish the active matched quantitative path from
older standalone APIs and historical launchers.

### F25: the generic C45 motion wrapper drops lengths during resampling

The restored
[st_eval.py](/lustre/whwjdqls99/cosmos/weka/shape_aware_motion_eval_c45_20260715/code/cosmos_motion_ft/shape_aware_TMR/st_eval.py:109)
resamples the entire supplied tensor when source FPS differs from evaluator
FPS and then sets `lengths = None`. Its subsequent default length becomes
the full resampled tensor extent for every sample. Consequently, calling
this API directly with differently sized, already-padded sequences at
20 FPS can make padding valid to the 30 FPS encoder. This is a
source-identified API defect, not an observed failure of the completed
85k/100k evaluations.

The active route deliberately bypasses that wrapper:
[phase2_quantitative.py:161](/home/whwjdqls99/cosmos_motion_ft/motion_expert_joint_attention/phase2_quantitative.py:161)
calls
[_tmr_embed_joints](/home/whwjdqls99/cosmos_motion_ft/motion_expert_joint_attention/eval_phase2_shape_tmr.py:161).
That helper handles each valid sequence before padding:

```python
posed = joints[index, :length].detach().cpu()
posed = resample_joints_time(posed, source_fps, embedder.fps)
# Construct normalized/canonicalized evaluator features per sequence.
# Only afterward pad those features and build the encoder mask.
```

It calls `embed_motion_features` with the explicit mask, not
`embed_motion`. GT and generated motion share this helper. Therefore this
particular padding/resampling defect is excluded as the explanation for
the completed matched quantitative comparison. The restored
[inline evaluator](/lustre/whwjdqls99/cosmos/weka/shape_aware_motion_eval_c45_20260715/code/cosmos_motion_ft/shape_aware_TMR/st_inline_eval.py:109)
also resamples individual cases before feature padding.

The generator/evaluator normalization boundary remains separate:
[_decode_generated](/home/whwjdqls99/cosmos_motion_ft/motion_expert_joint_attention/eval_phase2_shape_tmr.py:212)
first applies generator mean/std to normalized 283-D outputs and decodes
raw joints. The per-case C45 helper then constructs its own canonicalized
features using the evaluator representation and evaluator statistics.
The active Edge loader validates its checkpoint contract before
[loading the selected generator statistics](/home/whwjdqls99/cosmos_motion_ft/motion_expert_joint_attention/phase2_quantitative.py:174).
This pass does not replace either set of statistics.

### Text features are an evaluator input, not replacement reasoner conditioning

The restored
[LLM2VecCache](/lustre/whwjdqls99/cosmos/weka/shape_aware_motion_eval_c45_20260715/code/cosmos_motion_ft/shape_aware_TMR/st_text_cache.py:22)
stores one pooled vector per caption and returns a tensor with shape
`[B, 1, D]`. Thus the evaluator's all-true one-token text mask is
appropriate for this cache; these are not ragged language-token sequences
whose padding mask has been discarded. This LLM2Vec path is separate from
the Cosmos reasoner tokens conditioning the generator.

Missing captions raise `KeyError`; the loader does not silently substitute
the null caption or encode text online. The null caption is found by its
string key, not by assuming row zero. Duplicate cache strings would select
the last matching row, and the loader does not independently enforce
4096 columns or finite feature values. Those are validation limitations,
not evidence that the selected benchmark cache is malformed.

The older
[BONES case builder](/home/whwjdqls99/cosmos_motion_ft/motion_expert_joint_attention/prepare_shape_tmr_eval.py:43)
retains raw metadata text. The active
[quantitative preparation caller](/home/whwjdqls99/cosmos_motion_ft/motion_expert_joint_attention/phase2_quantitative.py:79)
explicitly applies Kimodo `sanitize_text` to those BONES cases before
writing the immutable manifests and evaluator-caption list. The restored
inline evaluator independently sanitizes its metadata captions too.
The previously completed 100k report recorded zero missing evaluator
captions. No new cache construction or caption rewriting was performed.

### Retrieval and FID protocol traced to the restored implementation

The restored
[retrieval implementation](/lustre/whwjdqls99/cosmos/weka/shape_aware_motion_eval_c45_20260715/code/kimodo_open/kimodo/metrics/tmr.py:328)
uses normalized C45 motion/text embeddings supplied by the embedder. It
does not perform generator-feature normalization itself.

Its apparently different similarity thresholds have a specific meaning:
`get_score_matrix_unit` maps cosine similarity to `cosine / 2 + 0.5`.
With `threshold=0.99`, `contrastive_metrics` compares these text scores
against `2 * threshold - 1 = 0.98`. Hence the effective text-cosine
threshold is strictly greater than 0.96. Each query accepts the best motion
among its directly similar-caption positives; this is not a transitive
caption-clustering algorithm or global removal of duplicate candidates.
Exact score ties use their average rank. All compared models use this
same convention.

The separate plain-retrieval helper counts strictly greater scores and
therefore gives optimistic ranks to ties. Plain and protocol retrieval
must not be silently interchanged. The inline evaluator has collapse and
text-clustering diagnostics, but those flags are not a reason to redefine
the current quantitative protocol or aim for a desired score.

`TMR/FID/gen_gt` compares generated and GT motion-embedding distributions;
`gen_text` compares motion and text distributions and is a different
quantity. FID uses empirical means/covariances and a matrix square root,
with diagonal regularization only on its singular/complex fallback.
It is not an average per-frame joint error. The
[six-suite summary](/home/whwjdqls99/cosmos_motion_ft/motion_expert_joint_attention/eval_phase2_shape_tmr.py:747)
explicitly computes case-weighted means of separately evaluated suites,
not retrieval or FID over a newly merged candidate pool. Accordingly,
the bundle README's six-suite historical scores are not directly
comparable to one repetition/overview cohort.

An additional dormant API defect exists in
[all_contrastive_metrics](/lustre/whwjdqls99/cosmos/weka/shape_aware_motion_eval_c45_20260715/code/kimodo_open/kimodo/metrics/tmr.py:395):
it passes `return_cols=True` to `contrastive_metrics`, whose actual
signature does not accept that keyword, and expects tuple outputs that
the function does not return. The active path uses
`compute_tmr_retrieval_metrics`, not this broken convenience helper.
It cannot explain a successfully completed run through the active path.

### Older evaluator entry point and text-cache builder have narrower guarantees

The standalone
[evaluator main](/home/whwjdqls99/cosmos_motion_ft/motion_expert_joint_attention/eval_phase2_shape_tmr.py:808)
loads module-level `SHARED_MEAN_PATH/SHARED_STD_PATH` before loading its
Native model. Its introductory description alone does not guarantee that
arbitrary checkpoint statistics have been selected. Correct environment
selection is required. This is distinct from the active mixed-model
quantitative loader and the previously checked 85k/100k configuration.

The standalone evaluator's
[shard reuse signature](/home/whwjdqls99/cosmos_motion_ft/motion_expert_joint_attention/eval_phase2_shape_tmr.py:859)
records checkpoint/text-cache/evaluator-checkpoint identities, case and
neutral hashes, protocol hash, rank topology, sampling steps and variants.
It omits actual generator-stat and evaluator-stat identities, `--fps`,
batch size and implementation identity. Reusing its output directory after
changing one of those inputs can therefore accept stale shards while
writing a report with the new CLI labels. Its final coverage, uniqueness
and finiteness checks do not detect that semantic mismatch. No claim is
made that the completed active quantitative evaluation reused such shards.

The
[text-cache builder](/home/whwjdqls99/cosmos_motion_ft/motion_expert_joint_attention/precompute_nymeria_tmr_text.py:83)
copies existing benchmark vectors and only encodes missing captions.
Its parity gate checks cosine similarity on a small caption prefix;
maximum absolute difference is logged but not bounded. Cosine alone
does not prove equal vector magnitudes or correctness on all new captions.
There is also no explicit finite check before `parity_min < threshold`,
so a NaN parity statistic would not trigger that comparison. Output reuse
checks provenance metadata, not every stored caption/feature value.
These are limitations of the guard, not newly observed failures of the
actual cache. Downstream finite embedding checks are a separate defense.

The generic metric base merely accumulates values and concatenates/stacks
them; it does not supply an additional validity or provenance check.
The inline evaluator temporarily changes module training flags, and
restores the text encoder according to the motion encoder's former
training state rather than independently preserving both flags. Its
C45 inference use starts in evaluation mode; this limitation does not
establish a train-mode leak in the current quantitative path.

### What the historical scripts and Edge MD files do and do not prove

The complete
[H200 C45 launcher](/home/whwjdqls99/cosmos_motion_ft/motion_expert_joint_attention/sbatch_phase2_shape_tmr_eval.sh:14)
hard-codes H200 paths, an eight-GPU partition request and a default Nano run
without the later contact-loss suffix. Its existence alone does not prove
that it evaluated the successful contact-trained checkpoint, and it must
not be launched unchanged on Yonsei.

By contrast, the captured
[Yonsei Nano visualization launcher](/scratch2/whwjdqls99/cosmos/operations/nano_phase2_viz_20260910/run.sh:33)
explicitly selects the contact-trained 200k checkpoint, clears inherited
Nymeria/statistics/cache and distributed-rank overrides, requests native
x0/shift-3/UniPC-35/CFG-2, and invokes `--viz_only`. It is not a hidden
Nano training restart. This source evidence complements, but does not
replace, the previously recovered actual checkpoint/archive metadata.

The captured
[legacy Edge sampler comparison](/scratch2/whwjdqls99/cosmos/operations/phase2_legacy_sampler_comparison_20260912_job2213496/compare.py:41)
hard-links the selected checkpoint inode before sampling, loads one model,
and reuses the existing ten fixed cases across its four step/CFG settings.
It instantiates no optimizer. This is a within-Edge sampler comparison,
not an Edge-versus-Nano architecture equivalence test or a full-cohort
quality evaluation. Source code describing completion is not fresh
evidence that any job is currently live or finished.

The complete
[Edge README](/home/whwjdqls99/cosmos_motion_ft/motion_expert_t2m_edge/README.md:25)
describes the historical Grasp Nymeria-only, batch-128 schema-v3 default.
It separately permits BONES as an explicit ablation. That text should not
be treated as the exact settings of the later Yonsei BONES50/global-256
parity run. Its older allocation/status statements are historical,
not current Slurm observations.

The complete
[smoke record](/home/whwjdqls99/cosmos_motion_ft/motion_expert_t2m_edge/SMOKE_RESULTS.md:174)
explicitly says that finite short forward/backward, reload and sampled
outputs establish wiring, not learned quality. Neither those results nor
their CPU mask tests establish numerical correctness of the real packed
GQA attention backward kernel. The previously identified actual
Edge-versus-Nano backend difference remains a higher-value causal lead
than the inactive evaluator defects above.

New full-file coverage in this reconciliation comprises the bundle README,
common evaluation helper, restored C45 wrapper, inline evaluator and text
cache, restored metric base and retrieval implementation, full Native C45
evaluation/preparation/text-cache code and its historical launcher, both
Edge MD files, and the two captured visualization/sampler launch artifacts.
Existing JSON captures are not newly claimed as reviewed runtime state.
The initial missing `motion_expert/dataset.py` capture remains an error,
not a reviewed source file or the active paired dataset.

Only this audit document was edited. Training, source code, checkpoints,
environments and allocations were untouched. No numerical tests, cache
rebuilds, new evaluations or post-edit validation were performed.
The implementation-quality root cause remains unresolved.


## Attention coverage extension: installed NATTEN configuration and CUDA boundary

This continuation expands the earlier explicitly partial attention-source
coverage. It uses cached framework sources and newly captured installed
NATTEN Python modules. No tensor/model tests, imports of the installed
GPU package, environment changes or training actions were performed.

### Framework tails: common code, not a newly different motion attention path

The remaining portions of both framework attention frontends, backend
selectors, common checks and NATTEN adapters were reviewed. Corresponding
nonblank/noncomment source is identical after accounting for one
metadata-function path in a docstring. The already established top-level
backend-registration/selection differences remain; this finding does not
claim that the entire attention implementations or actual kernels match.

The newly covered functions implement multi-dimensional attention,
variable-sized spatial attention, spatio-temporal attention and attention
output merging. The Phase-2 custom motion/reasoner path uses the standard
packed one-dimensional attention entry point, not a three-dimensional
spatial neighborhood kernel. Using 3D mRoPE positions is not equivalent to
calling a 3D neighborhood-attention operator.

One detail changes how a future backend comparison must be constructed:
the
[multi-dimensional full-attention shortcut](/home/whwjdqls99/cosmos-framework-edge/cosmos_framework/model/attention/frontend.py:453)
flattens Q/K/V and calls standard attention without forwarding the
requested backend or backend-specific arguments. This is explicitly
logged by that code. A test that merely passes a backend name to this
multi-dimensional shortcut could therefore compare the same automatically
selected backend twice. The pending numerical comparison must instead
use the active standard attention entry point with independently checked
backend selection.

Other framework-tail limitations are not current quality explanations.
The multi-dimensional NATTEN selector asserts that its FMHA-to-FNA mapping
contains the selected result even when the lower selector can return None.
An unsupported use case can consequently assert rather than cleanly
return False. Variable-sized spatial attention requires a newer NATTEN
version than the observed installed version, but this restriction does not
apply to the standard packed FMHA already selected in the Phase-2 probe.
Attention merging has separate LSE/backpropagation constraints; the custom
Phase-2 two-call attention pattern does not merge different attentions
over the same query with that API.

### Installed CUTLASS configuration: sequence-local sizing is preserved

The actual installed
[CUTLASS configuration package](/lustre/whwjdqls99/cosmos/envs/cosmos/lib/python3.13/site-packages/natten/backends/configs/cutlass/__init__.py:202)
was read in full. It is a package directory, not a missing cutlass.py
module. Both default forward and backward one-dimensional tile shapes
are 64 by 64. The larger candidate tiles are architecture/dtype gated;
their presence in an enumeration is not proof that the current run uses
them.

The previously inspected FMHA autograd wrapper passes
max_seqlen_KV into
[check_cutlass_fmha_backward_config](/lustre/whwjdqls99/cosmos/envs/cosmos/lib/python3.13/site-packages/natten/backends/configs/cutlass/__init__.py:653).
Its split-selection helper replaces the packed tensor extent with that
per-sequence maximum. Therefore the final default split count is not
computed as though every packed sample's keys formed one huge sequence.
The split limit also depends on the memory preference, heads, batch
dimension and deterministic/KV-parallelism settings. These controls select
work partitioning; their source does not establish numerical equivalence
to FlashAttention.

Some edge-case input validation remains permissive. Explicit integer KV
split counts are accepted without a positivity check at this layer. The
generic input-size helper used for default split selection requires sizes
greater than one, despite the varlen metadata validator permitting a
maximum length of one. Current Edge text conditioning contains more than
one token, and motion windows also exceed this boundary. These limitations
should be considered when constructing tiny synthetic probes, not treated
as demonstrated defects on the actual training inputs.

The
[capability checks](/lustre/whwjdqls99/cosmos/envs/cosmos/lib/python3.13/site-packages/natten/backends/configs/checks.py:403)
confirm that the CUTLASS path permits BF16, divisible-by-eight head sizes
and grouped heads through the wrapper's head repetition. These are
compatibility predicates, not accuracy tests. Some alternative
Hopper/Blackwell/Flex predicates test query.requires_grad alone for
backward restrictions, rather than OR-ing all Q/K/V gradient flags.
Current trainable motion queries require gradients; this observation does
not establish an incorrect selection for the active motion branch.

### Cumulative-length validation does not itself prove sample boundaries

The full installed
[varlen checker](/lustre/whwjdqls99/cosmos/envs/cosmos/lib/python3.13/site-packages/natten/utils/checks.py:568)
validates packed batch dimension, tensor devices/dtypes/ranks, matching
Q/KV boundary-array sizes, and positive maxima bounded by total lengths.
It does not inspect whether the cumulative arrays begin at zero, are
monotonic, end at the actual packed totals, or have adjacent differences
bounded by their stated maxima. Thus passing these checks alone is not
proof that sample-local boundaries are semantically correct.

The optional
[length-to-boundary helper](/lustre/whwjdqls99/cosmos/envs/cosmos/lib/python3.13/site-packages/natten/utils/varlen.py:33)
prepends zero and builds int32 cumulative sums, but likewise does not
validate positive individual lengths or equality of their sum to the
packed tensor extent. It obtains maxima with device-to-host item calls
and disallows this convenience path inside torch.compile. The active
Cosmos call supplies precomputed cumulative arrays and maxima, so this is
not a newly discovered repeated item() call on that path.

The custom packer's earlier source review remains separate evidence of
how its boundaries are built. No malformed active boundary tensor was
observed in this pass. A numerical probe must include several unequal
sequence lengths, not just one packed sample, to cover the boundary
semantics that the input checker leaves to its caller.

### Newly inspected Python-to-CUDA layer

The installed
[libnatten import shim](/lustre/whwjdqls99/cosmos/envs/cosmos/lib/python3.13/site-packages/natten/_libnatten/__init__.py:26)
loads the compiled library and Python operator wrappers. If that import
fails, its fallback functions raise ImportError; they are not a silent
identity, zero-output or CPU attention implementation.

The actual
[FMHA operator wrapper](/lustre/whwjdqls99/cosmos/envs/cosmos/lib/python3.13/site-packages/natten/_libnatten/torch_wrappers.py:440)
makes Q/K/V contiguous, allocates output in the query dtype and LSE in
FP32, and zero-initializes both for varlen calls before invoking the C++
function. The
[backward operator](/lustre/whwjdqls99/cosmos/envs/cosmos/lib/python3.13/site-packages/natten/_libnatten/torch_wrappers.py:516)
also zero-initializes dQ/dK/dV for varlen, forwards the saved Q/K/V/output,
incoming output gradient, LSE, scale, causal flag and both cumulative
arrays/maxima, and returns all three gradients.

There is a second split-count guard immediately before the C++ backward:
it clamps num_kv_splits to the number of KV tiles computed from
max_seqlen_KV for packed inputs. A source comment attributes this guard to
incorrect split arguments after some torch.compile recompilations. The
comment is not evidence that our current training is compiled, that it
has hit this case, or that the existing clamp is mathematically wrong.
The pending comparison should record the actual split settings and can
separate default splitting from a one-split case if approved.

The PyTorch-delta-reduction option is forwarded to the C++ call. This
Python wrapper does not itself expose the arithmetic implementing that
option; the earlier claim that the option is enabled must not be
upgraded into proof of a particular compiled accumulation algorithm.
The exported compute_delta helper is marked test-only and is not the
ordinary Python function used for this backward path.

Operator aliases select either the directly wrapped functions or their
registered torch.ops equivalents according to the package setting.
Both aliases target these wrappers. The fake implementations only return
shape-compatible empty tensors for fake/meta dispatch; their existence
does not mean actual CUDA inference returns uninitialized fake outputs.

### Coverage and remaining causal work

This pass adds the remaining ranges of ten framework attention files,
three complete installed configuration/checking modules, the libnatten
import shim and fallback module, and the varlen parameter generator.
The 1,732-line torch_wrappers file has been reviewed through line 740 and
from line 1501 onward, including its active FMHA functions and export
aliases. Its intervening FNA/reference/permutation machinery is not
newly claimed as reviewed. The generic neighborhood functions after the
previously covered standard attention function in natten.functional also
remain a distinct coverage item.

The highest-value remaining check is still numerical comparison of the
actual packed GQA forward and backward, including causal reasoner-only
attention and noncausal motion-to-reasoner-plus-motion attention, with
several ragged samples and the real dtype/scale. Source-level allocation,
metadata forwarding and compatibility checks do not establish the
compiled operator's numeric correctness. That GPU comparison and the
failed text-key-scale diagnostic repair remain unapproved and unrun.

This section was appended without reapplying or reading back the earlier
evaluator-boundary patch, whose tool confirmation was missing. Its earlier
write state is not resolved by this append. Training code, data, model
weights, environments and allocations were left untouched. No post-edit
verification was performed, and the root cause remains unresolved.



## Dependency pins, wheel provenance and NATTEN context controls

The complete current
[Edge dependency manifest](/home/whwjdqls99/cosmos-framework-edge/pyproject.toml:170)
and
[Native dependency manifest](/home/whwjdqls99/cosmos-framework/pyproject.toml:170)
were captured and reviewed. Their cu128 and cu128-train dependency blocks
are textually identical. In particular, both pin Torch 2.10.0+cu128,
FlashAttention-2 2.7.4.post1+cu128.torch210, FlashAttention-3-NV
1.0.3+cu128.torch210, NATTEN 0.21.6.dev6+cu128.torch210 and cuDNN
9.10.2.21. These agree with the versions reported by the earlier scheduled
backend-selection inspection. No new package imports or runtime probes
were executed in this pass.

The NATTEN package records were extracted from each framework lockfile,
without claiming full review of those entire lockfiles. The selected
cu128 record is identical in both. It names this wheel:

```text
natten-0.21.6.dev6+cu128.torch210-cp313-cp313-linux_x86_64.whl
locked SHA256:
164b4cd16d36f747ddd19c6f0cb1cd0a275e758e01f2402d4e957be5d7818a0d
```

Both resolve it through the Cosmos dependency index and the v1.5.0
release asset. Thus these checkouts do not require different NATTEN
builds for the selected CUDA 12.8 environment. This supports the existing
distinction: Edge and Nano select different backends from the shared
installed environment, rather than these manifests requesting different
cu128 attention-package versions.

This is not proof that every installed byte matches the lockfile.
The installed
[WHEEL metadata](/lustre/whwjdqls99/cosmos/envs/cosmos/lib/python3.13/site-packages/natten-0.21.6.dev6+cu128.torch210.dist-info/WHEEL)
identifies a non-pure CPython-3.13/Linux-x86_64 wheel. Its distribution
metadata reports the full local-build version; the package's
[version module](/lustre/whwjdqls99/cosmos/envs/cosmos/lib/python3.13/site-packages/natten/version.py:24)
reports only the base development version. These two strings are not
evidence of a mismatch: the latter omits the CUDA/Torch local version
suffix.

The installed RECORD lists the compiled library as 358,508,056 bytes and
records its URL-safe-base64 SHA256 as
`yPqvA8X0r6PmBiN5lsFDDkuoFw5pvcdMO15Ob9zcUpE`.
That is an expected installed-file digest recorded by the wheel, not a
digest recomputed during this audit pass and not the compressed wheel's
SHA256. No binary hashing, wheel download or rebuild was performed.
The inspected RECORD has no C++/CUDA source-file entries. This does not
prove that the build source is unavailable elsewhere.

No direct_url.json was present at the selected distribution metadata
path. The inspected version/distribution metadata does not identify an
upstream C++ Git commit. An index-installed distribution need not have a
direct-URL record, so that absence alone is not corruption evidence.

The official dependency repository describes a wheel build/publication
workflow. Its v1.5.0 release identifies publishing-repository commit
87e86d1df3efe2ff12c4892c0ed29206a2129f89. That publishing commit is not,
by itself, an attestation of the NATTEN C++ revision embedded in a wheel.
The release asset listing was not successfully expanded, and source
directory/API requests failed in the web tool. Therefore this lookup
does not establish the wheel's exact source revision or a match to the
historical H200 runtime.
[Dependency repository](https://github.com/nvidia-cosmos/cosmos-dependencies),
[v1.5.0 release](https://github.com/nvidia-cosmos/cosmos-dependencies/releases/tag/v1.5.0).

### Other manifest differences are real but not the observed cu128 selection

Both framework manifests use the same project name and version 1.2.2;
those identifiers alone cannot distinguish their source trees. Their
contents do differ outside the shared cu128 groups. Edge requires a
minimum Diffusers version and adds newer CUDA-13/Torch-2.13 groups;
Native retains an editable diffusers-cosmos3 training dependency and
different optional-group conflicts. These are not evidence that the
running CUDA-12.8 process switched to Torch 2.13, a GB300 wheel or a
different UniPC implementation. Earlier source/runtime-path and sampler
comparisons remain the relevant evidence for those claims.

The repository-wide Python lower bound does not imply that every
accelerated dependency wheel supports all such Python versions. The
selected locked NATTEN artifact is specifically CPython 3.13, consistent
with the observed Yonsei interpreter. No inference about the original
H200 interpreter is made from the current checkout's manifest.

### Newly found inactive Flex-control bug

The full installed
[NATTEN context module](/lustre/whwjdqls99/cosmos/envs/cosmos/lib/python3.13/site-packages/natten/context.py:161)
has a concrete control-flow defect:

```python
if not mode:
    NattenContext.flex_compile_allowed = False
    NattenContext.flex_compile_backprop_allowed = False
# No return here.
...
NattenContext.flex_compile_allowed = True
```

Consequently, allow_flex_compile(False), including the
disable_flex_compile wrapper, turns forward compilation permission back
on before returning. Also, when forward permission is already on,
passing backprop=False does not clear an already-enabled backprop flag:
the function only updates that flag in its backprop=True branch.
Thus disable_flex_compile_backprop does not reliably undo prior enabling.

This defect is not established as active in Phase-2. The reviewed custom
Edge attention/model/layer/trainer and Cosmos NATTEN adapter contain no
calls to these Flex-control helpers, and the observed attention backend
is CUTLASS FMHA rather than Flex. Permission to use a backend is also not
the same as selecting that backend. The module initializes both Flex
permissions to False. No control flag or package code was changed.

The package
[environment parser](/lustre/whwjdqls99/cosmos/envs/cosmos/lib/python3.13/site-packages/natten/utils/environment.py:31)
accepts only literal 0 and 1 for boolean overrides; other nonempty strings
fall back to the default. A proposed test of NATTEN_DISABLE_TORCH_OPS must
therefore use 1, not true. Its default registers Torch operators on
Torch 2.8 and newer. This pass establishes the source default, not the
current value in a training process's environment.

The separate NATTEN deterministic flag, PyTorch deterministic flag,
KV-parallelism flag and memory preference are distinct controls.
Describing a run as deterministic from one flag alone would be
insufficient. The pending numerical comparison must record actual
settings rather than assuming that one convenience setter controls every
kernel path.

### Outcome and scope

This pass strengthens package/build provenance and identifies an inactive
third-party control bug. It does not establish a new active-path cause of
floating, weak text sensitivity or poor retrieval. No package upgrade or
downgrade is justified solely by the development-version label or the
unrelated Flex warnings.

The compiled forward/backward comparison and the text-key-scale probe
repair remain unapproved. Only this audit-document appendix was written.
Training, data, model state, environments and allocations were untouched;
no tests or post-edit validation were performed. The previous unconfirmed
evaluator-document patch remains unresolved, and was not reapplied here.


## Remaining NATTEN wrapper coverage and causal limits

This extension closes the previously recorded Python-source gaps in the installed
NATTEN `functional.py` (now lines 1-1152) and
`_libnatten/torch_wrappers.py` (now lines 1-1732). The final
`functional.py` capture had been truncated in the tool output; its remaining
lines were recovered from the existing cached source in bounded chunks before
being counted as reviewed. No physical source reread or numerical test was
performed. This is coverage of those captured files, not a claim that every
transitive dependency or compiled kernel has been audited.

### What the remaining functions actually do

- `functional.py:929` and `:1128`: `na2d` and `na3d` forward their inputs,
  neighborhood geometry, causal flags, scale and backend controls to
  `neighborhood_attention_generic`. They add no motion-specific coordinate,
  grounding or reasoner-conditioning transformation.
- `functional.py:404-444`: the neighborhood-to-full-attention shortcut routes
  standard-attention backend options through `attention_kwargs`, not the outer
  neighborhood `backend` argument. A future forced-backend experiment must
  exercise the actual standard packed one-dimensional attention call directly,
  rather than assuming a full-window neighborhood call selects the same kernel.
- `functional.py:524-559`: the optional neighborhood additional-context path
  computes another full attention and merges outputs using log-sum-exp values.
  This is not the custom Phase-2 coupling: motion queries attend to concatenated
  reasoner and motion keys/values, while reasoner queries have their separate
  causal attention call. Three-dimensional mRoPE coordinates do not turn that
  operation into three-dimensional neighborhood attention.
- `torch_wrappers.py:796-1303`: the Hopper, CUTLASS and reference FNA factories
  register dimensional wrappers, make inputs contiguous, allocate outputs and
  gradients, and pass geometry, scale and scheduling arguments to compiled
  functions. The reference FNA operators are also compiled-library calls; their
  name does not make them an independent Python or high-precision oracle.
- `torch_wrappers.py:1311-1480`: the token permutation helpers implement padded
  neighborhood tiling and its inverse, including dilation groups folded into
  the batch dimension. They are not an extra permutation in the active standard
  packed FMHA path.
- Some inactive FNA wrappers allocate with `torch.empty`; that is not evidence
  that active packed FMHA leaves outputs unwritten. Its previously inspected
  wrapper explicitly zero-initializes packed outputs and gradients. Fake
  operator implementations describe shapes, not executed CUDA arithmetic.

These additional lines expose no new demonstrated cause of floating, low
retrieval or weak caption sensitivity. Inactive-library defects must not be
presented as an explanation of the actual training run.

### Remaining discriminating evidence

The earlier owned-compute-node inspection selected NATTEN CUTLASS FMHA for Edge
and FlashAttention2 for Nano in the Yonsei environment. That observation does
not establish which backend the historical H200 training used. Python argument
flow, scaling and grouping checks do not prove numerical equivalence of the
compiled forward or backward kernels.

An isolated comparison, if explicitly approved, should use identical inputs
within each actual head configuration (Edge Q16/KV8 and Nano Q32/KV8, head
dimension 128), BF16 inputs, ragged sample boundaries, the reasoner's causal
case and the motion-query noncausal case. It should compare both outputs and
Q/K/V gradients with an independent FP32 mathematical reference, record the
backend actually executed and preserve the real scale and split settings.
Outputs from different head configurations must not be compared as if their
shapes were identical. No production optimizer step, restart or backend change
is needed for this diagnostic.

Numerical-test permission remains requested but not granted. The earlier
text-key-scale diagnostic also still needs approval to repair its
inference-tensor `_version` access failure; it produced no scale-comparison
result. Neither limitation is a confirmed training bug. The root cause remains
unresolved and the full audit is not complete.


## Processor loading coverage and audit evidence boundary

The previously unseen ranges of the installed
[processing_utils.py](/lustre/whwjdqls99/cosmos/envs/cosmos/lib/python3.13/site-packages/transformers/processing_utils.py)
have now been reviewed from the captured source: lines 1-478, 652-1204 and
1396-1527. Together with the earlier ranges, the coverage record spans lines
1-1782. No processor, tokenizer, environment or training configuration was
modified.

### Newly inspected behavior

- Lines 100-456 define typed argument schemas and chat-template options. These
  declarations are not an additional motion-feature normalization or transform.
- Lines 652-888 serialize processor configuration, tokenizer/image components
  and chat templates. They are not part of the motion loss or decoded-coordinate
  computation.
- Lines 1067-1125 load legacy chat-template JSON before the raw default Jinja
  branch; combining legacy JSON with additional named templates raises an
  error. Missing processor configuration can fall back to an empty dictionary
  for backward compatibility. A successful processor load alone therefore
  does not prove all optional configuration files were present. No actual
  missing-template or wrong-template failure was established for this run.
- Lines 1177-1203 and 1510-1520 let explicit kwargs override configuration, then
  keep constructor-recognized arguments. Unsupported options become
  `unused_kwargs`, returned only when requested. This is a possible ignored-
  option hazard, not evidence that any actual run option was discarded.
  Component loaders receive kwargs separately, so this filtering cannot be
  treated as proof that tokenizer or image-loader options are all lost.
- Lines 1421-1455 select fast/slow implementations only when a component's class
  declaration is a tuple. In that branch, tokenizer selection defaults to fast;
  unset image `use_fast` selects the slow branch with a warning. Actual behavior
  still depends on the declared classes and caller options. No actual
  Edge/Nano image-processor mismatch was demonstrated, and an image-only
  discrepancy would not directly explain the image-free T2M failure.
- Lines 1457-1507 resolve registered component classes or raise, delegate
  decoding to the tokenizer, and collect component input names. They do not
  silently replace missing classes with empty conditioning.

The constructor-filtering behavior is explicit:

```python
processor_dict.update(kwargs)
unused_kwargs, valid_kwargs = cls.validate_init_kwargs(
    processor_config=processor_dict, valid_kwargs=accepted_args_and_kwargs
)
processor = cls(*args, **valid_kwargs)
```

This inspection adds source-coverage and loading-contract evidence, not a new
demonstrated cause of the quality gap.

### What the audit proves, and what it does not

| Requirement | Established evidence | Remaining limit |
| --- | --- | --- |
| Data loading and representation lineage | Recorded custom loader/producer/statistics review, selected-window finiteness results, Drive archive evidence and recanonicalization comparison | These checks do not reconstruct every historical H200 worktree or runtime state. |
| Architecture, activations and reasoner conditioning | Recorded custom/base source review and an existing matched caption-conditioning comparison | A measured conditioning-response gap is not proof of which implementation difference caused it. |
| Attention implementation | Python packing, grouping, scale and backend-selection paths reviewed, including the previously omitted wrappers | Actual compiled forward/backward numerical equivalence remains untested. |
| Evaluation interpretation | Completed 85k/100k evaluation comparison does not meet the requested substantial-improvement stopping criterion | That observation neither identifies a root cause nor substitutes for matched-exposure comparisons. |
| Complete source coverage | The remaining line-range gaps in the captured Python coverage record have been closed | A coverage record is not proof that every transitive source file or compiled instruction was captured; three historical entries use separate evidence rather than the standard source-cache keys. |
| Documentation | New findings appended without overwriting the existing audit | No post-edit readback was performed; an older evaluator-append acknowledgment remains unresolved. |

The earlier inspected Yonsei environment chose CUTLASS FMHA for Edge and
FlashAttention2 for Nano. This is a concrete difference to discriminate, not a
confirmed faulty kernel and not proof of the historical H200 backend. The
highest-value next action is the isolated packed-attention forward/backward
comparison described above, on an owned compute-node GPU, with training
untouched.

Explicit permission for that numerical diagnostic has not been received.
Repair of the earlier failed text-key-scale diagnostic also remains
unapproved. Further repetition of already reviewed source or expansion into
unused library branches would not establish the missing causal evidence.
The audit remains incomplete; no production fix is justified by these new
processor findings.

