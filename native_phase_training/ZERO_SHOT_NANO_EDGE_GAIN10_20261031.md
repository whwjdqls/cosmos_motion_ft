# Zero-shot Nano / Edge with camera translation gain10

Requested report: https://app.notion.com/p/binne/409d2a44f44b80899b2bec1fc10ba9bc

Operation: `/scratch2/whwjdqls99/cosmos/operations/edge_nano_camera_gain10_20261031`.
Driver: `native_phase_training/compare_edge_nano_gain10.py`.
A6000 request2651408: one48GB GPU,8CPUs,128GB host RAM,3days, suma/gigabyte partitions,
excluding cs-gpu-01,node37 and previously faulty node29. Tmux
`zs_nano_edge_a6000_20261031` holds the srun and automatically starts `run.sh`
after allocation and the preparation READY gate. No unrelated jobs are cancelled.

## Controlled inputs

Reuse the exact six fixed71 windows0,7,14,28,42,63, initial images and seed0
from the completed `edge_zeroshot_camera_cfg_20261031` study. Copy each96x9
backward-framewise camera action array; multiply only the first three translation
channels by10. Keep all six rotation channels bit-identical. Preserve the input
files and record both original/scaled hashes; no absolute-position scaling,
rotation amplification or trajectory fitting.

Generate zero-shot Edge and Nano at cameraCFG1,1.5,2:36 new clips. Same official
UniPC30/shift10/T97/20fps/256, empty text, no diffusion or text-KV cache. Reuse
the completed two-task Phase1 EMA200k outputs at original camera gain1 and the
same three CFG values. Prior zero-shot Edge gain1 metrics stay in the new table
as a direct gain1/gain10 comparison. The prior Notion report is retained.

Camera CFG is distinct from trajectory scaling. The isolated existing hook
removes only camera tokens in the null branch. It retains image/text/noisy video
and timestep; scale1 has no null forward. Check first-case scale1 native output
parity bit-for-bit for each backbone and log30/0 or30/30 branch calls.

Use official HF checkpoints and official YAML/CLI. Disable unused audio modules
and Nano's understanding vision tower, matching the previous official five-way
study. Generation image conditioning uses Wan. LoRA disabled; do not allocate an
EMA copy of the public model. This is inference only; no training/checkpoint edits.

## Text + first-image comparison

Add Nano to all10 prior examples: six held-out captions plus left/right prompts
on images0/14. Same image/prompt/seed0, official UniPC35/textCFG6/shift10 and
native diffusion-cache default. No camera input, so translation scaling does not
apply. Reuse verified Edge zero-shot/originalPhase1/two-taskPhase1 clips. Four
columns: Nano zero-shot, Edge zero-shot, original200k, two-task200k.

## Metrics and artifacts

PSNR/SSIM/LPIPS-Alex on generated frames1–96, equal window weighting, MP4-decoded
RGB exactly as the preceding report. Measure GT identity too (PSNR100cap/SSIM1/
LPIPS0). Text metrics use six original captions only; alternate left/right futures
have no matching GT. Save per-case and per-frame camera scores. This six-window,
one-seed diagnostic is not a full71 benchmark or a prompt-accuracy metric.

Six camera videos: GT plus rows Nano x10 / Edge x10 / two-task x1 and columns
CFG1/1.5/2. Ten text+image videos: four model columns. Prompts/settings visible;
observed first frame green, generated frames red, GT blue. Decode all97 frames
and verify border colors; save contact sheets and gallery/index.html.

Trust per-arm VERIFIED markers, guidance audits, METRICS_COMPLETE and
MEDIA_COMPLETE. Final COMPLETE requires Notion table readback and remote media
hash verification. Queued allocation and prepared code do not mean inference ran.

## Queue / automation receipt

At08:23KST2651408 is pending on Priority; Slurm estimated12:28KST (not a
guarantee). The partition default initially reserved515GB/all-node RAM;
MinMemoryNode was corrected to128GB on the same pending job without cancelling
or resetting its submission. READY is installed after syntax checks. The tmux
srun will prepare and audit copied actions/reused artifacts, run both official
models sequentially, compute metrics, render16 videos, then upload and verify
the new Notion section. Failures retain logs/artifacts and update the status;
no model workload has run while the request is pending. The allocation remains
open after the workflow; do not cancel it without an applicable user request.
