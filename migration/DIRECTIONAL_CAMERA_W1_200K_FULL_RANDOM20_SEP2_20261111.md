# sep-2 directional local camera W1 EMA200k final evaluation queue

User authorizes evaluation immediately after this trainer completes200k, using
GPUs3/5, batched sampling, full test and frozen random20, reporting in Final
Variations. Also explicitly requires exact100k preservation for current
bidirectional/no-camera/W1 and Naeive-adaptation/W1.

## Queue and inference

Operation: `/mnt/local/jinhyung.kim/cosmos_storage/operations/eval_directional_camera_wv1_EMA200k_full_random20_20261111`.
Controller `manage.py` is detached/SIGHUP-safe, waits for exact final200k raw/EMA/
optimizer/scheduler/two-rank RNG/cursor, final validation completion and training
exit. It preserves/checksums the selected EMA200k checkpoint locally and uses
that run's own frozen source/specialist/data contracts. No training is stopped,
resumed or mutated. GPU5 starts when empty; GPU3 joins only when empty because
another researcher's VLLM is currently resident. Foreign processes are untouched.

Random20 runs first: exactly1420 frozen physical windows,20 per each of71 held-out
recordings; SHA47288f230f6b21835f30d5874641b24b0129ab669f99e2f0d03cb1dcf5ced2df.
Then full test:11677 unique physical windows, one deterministic valid caption per
window, exact parent eligibility. The same random20 output NPZs are hardlinked
into full-test chunks with original provenance retained and audited. No subset
reselection, caption substitution, GT-camera conditioning, or duplicate sampling.

Three inference conditions: M2V CFG1.1; V2M and video+text→motion CFG1.5, using
whole-seven-bridge-disabled reference, native UniPC35, shifts video10/motion3,
T97/20FPS/256px, supplied neutral shape. Generator captions empty; explicit text
only enters the frozen motion expert in both CFG branches. Cameras start from
private independent noise with conditioned camera readout/no separate camera CFG.
Sampling batches32M2V/64motion; decode4; logged OOM backoff. Four virtual shards
(355 each random20;2920/2919/2919/2919 full), up to two physical GPU workers.
Fresh native VAE encoding for previously unstaged windows; verified original
native1420 input encodings reused. No old incompatible latent cache substitution.
Only node-local data/checkpoints/metrics. Storage guard before every chunk.

## Quantitative protocol

Global FVD/FID/retrieval recomputed separately per population, never shard means.
Native I3D400 FVD uses8520 descriptors for random20,70062 for full. PSNR/SSIM/
LPIPS-Alex/DreamSim exclude observed frame0. Motion: current Body23 physical/
reconstruction, UPE/LPE/sJPE, matched91f9d8 TMR. GT physical/skating/penetration/
retrieval are measured on identical windows and preprocessing, not assumed zero/
100%. Per-case scores/features/embeddings retained. Recording bootstrap5000,
seed20261110, weights full recordings by their window counts (random20 is equal20).
Successful model/nativeCFG1 parity and full provenance/prefix/camera checks are
required by each GPU worker before scoring. Five CPU cohort/shard/bootstrap/
resource/supplement tests and all source syntax checks passed.

## Six missing evaluator text vectors

The original2.03GB frozen LLM2Vec cache is missing exactly six full-test captions.
Random20 is fully covered. Full-test generation still proceeds for all windows;
only affected motion-scoring chunks are marked pending, without fabricated
embeddings/metrics. The controller releases GPUs while waiting, then replays just
incomplete scoring when `TEXT_SUPPLEMENT_READY.json` arrives. Existing predictions
are never regenerated. `MISSING_TEXT_CAPTIONS.json` is authoritative.

The user has been asked for an expanded exact cache or authorized encoder bundle.
The original encoder base denied official Hugging Face access; no restricted
weights were downloaded. No NAS bulk transfer or private credential disclosure.
For a supplied expanded cache, `adopt_text_cache.py --cache PATH` extracts only
six vectors after exact known-caption parity; all original vectors stay unchanged.
`driver/text_supplement.py` overlays only missing rows. Do not substitute captions,
zero vectors, another text encoder, or silently drop six windows.

## Exact100k protection

Bidirectional: native rotation exemption for `step_000100000.pt` verified in its
frozen trainer; existing independent-copy watcher PID2802225/start40618130 alive.
Root: cosmos_storage/preserved_milestones/phase3_bidir_nocamera_wv1_100k_20261110.

NaeiveW1: native `checkpoint.save_iter=5000`, wall-clock retention explicitly
refuses periodic milestones, so100k persists. New independent DCP-copy watcher
uses scripts/preserve_native_motion_milestone.py; root preserved_milestones/
naeive_wv1_100k_20261111. Validate-only on the actual35k checkpoint passed all
model/raw+EMA, optimizer, scheduler and two-rank RNG metadata/shard bounds, trainer/
scheduler step checks. Future100k copy is fsynced, every file SHA256 verified,
with contract/config/frozen sources retained. No trainer restart was needed.
`100K_PROTECTION_VERIFIED.json` records both live process identities and policies.

## Reporting and resumption

Notion destination: https://app.notion.com/p/binne/Final-Variations-412d2a44f44b80a2936aca27f96ac016
Queue entry is published; no EMA200k metrics are yet claimed. Connected completion
publisher cell101 polls each cohort's COMPLETE/AUDIT and appends model/GT tables
and CSV independently, with unique markers and local publication receipts. Read
NOTION_PUBLISHER_HEARTBEAT; re-arm the connected publisher if its tool session has
expired. Durable local inference/preservation controllers do not depend on it.

Read status.json, MANAGER_STARTED.json (identity), MANAGER_ERROR, per-shard ERROR,
SAMPLING_COMPLETE vs COMPLETE, and cohort AUDIT before action. A sampling-complete
full shard may still await text scoring; do not call it quantitatively complete.
The initial controller attempt before CPU readiness was rejected; historical
RESOLVED_INITIAL_SETUP_GUARD.json preserves that event. Current controller is
armed and waiting for final200k. No duplicate manager may hold manager.lock.
