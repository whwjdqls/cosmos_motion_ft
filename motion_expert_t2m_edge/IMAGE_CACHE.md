# Optional Phase-2 image conditioning cache

The motion-array cache does not cache TI2M video frames or frozen image encoding.
These two additional caches reuse exact outputs of the existing implementations.
Both are disabled by default, including on Grasp.

Enable them in the launching shell before `train_contact_legacy_yonsei.sh`:

```bash
export COSMOS_REASONER_IMAGE_CACHE_ROOT=/scratch2/whwjdqls99/cosmos/cache/phase2_images_v1
export COSMOS_EDGE_CONDITION_CACHE_ROOT=/scratch2/whwjdqls99/cosmos/cache/phase2_images_v1
export COSMOS_EDGE_CONDITION_CACHE_ID=edge-d4599e2e43fbd06168e9884205b9b66c3902d8f6-pinned-weights-processor-v1
export COSMOS_EDGE_CONDITION_PREFETCH_THREADS=4
export COSMOS_EDGE_CONDITION_ASYNC_WRITES=1
```

On Grasp, use a writable cache path on that server instead of the Yonsei path.
Never share a cache ID across different frozen weights or processor versions.
The current ID refers to the artifact-validated, frozen Edge reasoner and visual
encoder. Change the ID if changing those artifacts or their preprocessing.

The frame key includes source path, size, modification time, frame index, FPS and
resize size. The conditioning key includes exact image bytes, caption, dtype,
image size and the explicit frozen-model ID. Caption dropout therefore has a
separate cache entry. Neither cache changes motion data, mean/std, task mixture,
losses, gradients through trainable parameters, or optimizer/LR state.

Entries are filled lazily, so a cold cache still performs the original work and
adds a one-time write. Speedup depends on reuse, and is not immediate for unseen
windows. Atomic no-replace publication allows multiple DataLoader workers and
trainers. Read/write errors fall back to the original path. Writes pause when
filesystem space falls below 20 GiB; this is not a user-quota check. Override
with `COSMOS_CONDITION_CACHE_MIN_FREE_GIB` if needed. Monitor user quota separately.

Caches contain no checkpoints or authoritative data and can be discarded when
unused. Yonsei scratch's retention policy still applies; regeneration is safe.
Keep frozen encoders in evaluation mode, as in the existing trainer. Caching
trainable or stochastic encoder outputs is not supported.

## Avoid blocking the GPU on shared-filesystem cache I/O

`COSMOS_EDGE_CONDITION_PREFETCH_THREADS` defaults to zero. Setting it to four
submits CPU-only cache reads for the batch concurrently; GPU transfers and all
encoder calls remain on the training thread in their original order.

`COSMOS_EDGE_CONDITION_ASYNC_WRITES=1` queues detached CPU conditioning outputs
for atomic background publication. The queue holds at most 64 entries; if full,
only the optional cache fill is skipped, never a training example. Outstanding
fills may be discarded at shutdown and regenerated later. Neither option changes
the cache format/identity, so existing entries remain usable on either server.

## Shared launch behavior on Grasp and Yonsei

The shared Edge entrypoint, `motion_expert_t2m_edge/run.sh`, now defaults to four
conditioning-cache reader threads and asynchronous cache writes whenever
`COSMOS_EDGE_CONDITION_CACHE_ROOT` is configured. This applies to both servers;
there is no hostname check or Yonsei path in this fast-I/O default. Explicit
`COSMOS_EDGE_CONDITION_PREFETCH_THREADS=0` and
`COSMOS_EDGE_CONDITION_ASYNC_WRITES=0` retain the synchronous comparison path.
The Python cache library remains opt-in when used outside this entrypoint.

On Grasp, configure a writable cache location on that server before invoking the
usual Edge launcher. Prefer node-local SSD or the site's designated scratch;
shared scratch can still incur network filesystem latency. Do not copy Yonsei's
`/scratch2/whwjdqls99` path into a Grasp launch. For example, after setting
`COSMOS_PHASE2_CACHE_DIR` to the selected Grasp cache directory:

```bash
: "${COSMOS_PHASE2_CACHE_DIR:?Set this to a writable cache directory on this server}"
export COSMOS_REASONER_IMAGE_CACHE_ROOT="$COSMOS_PHASE2_CACHE_DIR"
export COSMOS_EDGE_CONDITION_CACHE_ROOT="$COSMOS_PHASE2_CACHE_DIR"
# Use this ID only for the same pinned Edge weights and processor as the Yonsei run.
export COSMOS_EDGE_CONDITION_CACHE_ID=edge-d4599e2e43fbd06168e9884205b9b66c3902d8f6-pinned-weights-processor-v1
# Invoke the existing Grasp Edge launcher, which calls run.sh, with unchanged training flags.
```

Use a different immutable namespace if the frozen weights or processor differ.
Cache contents are disposable, site-local performance artifacts, not model
checkpoints. Neither this setup nor the I/O defaults change batch size, losses,
normalization, or the Nymeria/BONES sampling mixture.

A Yonsei RTX 4090 comparison using the real training loop measured warm-cache
throughput of 1.26 -> 1.42 steps/s and cold-cache throughput of 0.78 -> 0.89
steps/s with this I/O change. The conditioning outputs were bit-exact for a real
128-example batch and did not change the CPU or CUDA RNG state. Production's
first measured 120-step interval improved from approximately 1.00 steps/s before
the restart to 1.16 steps/s after it, including a recovery checkpoint save.
These are short-window Yonsei measurements, not a guarantee of identical Grasp
throughput; GPU model, cache hit rate, and storage affect the result.
