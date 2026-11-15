# Winning generated-caption random20 handoff

The completed caption/guidance study selects Q40 generated captions with
**video CFG 1.5 / text CFG 0.5**, supplied shape, directional W1/no-camera
EMA200k and native UniPC35. All 1,420 frozen windows across 71 recordings
are included. These are the actual conditioning strings used by the selected
native full-set evaluation, not Q77, fresh PRO-B1 recaptions or first71 only.

Prepared archive:

```text
/scratch2/minsu1206/cosmos/operations/random20_best_caption_handoff_20261115/random20_best_q40_v1p5_t0p5_20261115.zip
SHA256 f7154cef9a976c0e92e1bdbc98a2d4496bbe171ad68a95d3510642dd22c3fbd1
```

The extracted sibling directory contains the exact VLM instruction
`Q40_prompt.txt`, `captions_by_id.json`, ordered caption JSON/CSV, case bounds
and native/VLM seeds, portable recipe, generated/GT/control metrics, source
provenance and checksums. There is no target GT caption, motion, RGB,
checkpoint, credential or raw model reasoning in the bundle. The instruction's
three unrelated training style examples remain intact.

All parent directories are traversable and the new handoff is group-readable
for `ys_ms`, including `whwjdqls99`. The user's follow-up delegates Drive upload
to that account; **Drive upload remains an operator step until its receipt
exists**. No claim of completed upload or public sharing is made here.

From the `whwjdqls99` login:

```bash
bash /scratch2/minsu1206/cosmos/operations/random20_best_caption_handoff_20261115/upload_with_whw.sh
```

The wrapper defaults to
`gggdrive:cosmos_data/vlm_random20_best_q40_v1p5_t0p5_20261115`; pass another
rclone destination as its first argument if needed. It uses that account's
existing login, without copying credentials. `rclone check --download` verifies
the remote file contents after upload. Completion is recorded at:

```text
/scratch2/minsu1206/cosmos/operations/random20_best_caption_handoff_20261115/drive_receipt/DRIVE_UPLOAD_VERIFIED.json
```

Original caption source (four native shards 0–3):

```text
/scratch2/minsu1206/cosmos/operations/vlm_qwen38_caption_20261113/full_native_shards/captions/random20/Q40_qwen38_same_prompt_full32/
```

Exact instruction source:

```text
/scratch2/minsu1206/cosmos/operations/vlm_qwen38_caption_20261113/prompts/Q40_qwen38_same_prompt_full32.txt
```

The source exporter, verifier and unchanged independent video/text CFG adapters
are in [`nymeria_baselines/vlm_caption_handoff/`](../nymeria_baselines/vlm_caption_handoff/).
Model checkpoint/source dependencies remain in the
[EMA200k model folder](https://drive.google.com/drive/folders/1f8KgvTkn-Ez_aZ3gSaS5Ty9C1PrBvJbp).
The compact EMA SHA256 is
`78dd00a6df0d95a96dfefcd0a68ad376e6dff8968bb48f2b13d4be18551ffddc`.

Native full1420 all-ten comparison (distance cm; skate cm/s; R@3 percent):

| Input | MPJPE ↓ | PA ↓ | Root ADE ↓ | UPE ↓ | LPE ↓ | sJPE ↓ | R@3 ↑ | FID ↓ | Skate ↓ | Penetration ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Video |16.125472|8.265577|14.145068|14.409327|18.795031|0.379951|83.806818|0.024899|12.444675|0.909551|
| Q40 text CFG0.5 |14.627242|7.921580|12.465740|12.691532|17.638346|0.366483|86.126894|0.034835|12.126269|0.734787|
| GT text CFG1 |13.073182|7.207300|11.330708|11.661530|15.269084|0.381771|94.199811|0.022276|12.904979|0.708652|
| GT text CFG0.5 |13.505623|7.414496|11.634485|11.986614|15.868525|0.367114|91.571970|0.022582|12.473162|0.752449|
| Actual GT motion |0|0.000021|0|0|0|0|91.666667|0|14.417053|1.281043|

GT retrieval and physical metrics are measured, not set to ideal values.
The source-bound bundle has full precision. Repeated benchmark development and
metric tradeoffs are disclosed in the
[full report](https://app.notion.com/p/415d2a44f44b80b2b687dd273fa22f36).
