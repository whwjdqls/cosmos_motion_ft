# Bidirectional camera prediction without joint training

Controlled variant of the pretrained-motion camera/video-weight1 run. Tasks
are50% video→motion and50% motion+image→video, with joint probability zero.
Camera tokens are predicted in native generator attention, with no camera loss
or direct camera-motion bridge edges. Video velocity loss weight1 applies to
M2V; the original motion physical losses are unchanged. Seven117469198 bridge
parameters train, while the selected pretrained Phase2 motion specialist and
Phase1 LoRA+video-I/O EMA200k remain frozen. This is not scratch-motion training.

Retains original bidirectional attention, clean motion timestep zero,
captions/dropout, noise clocks/private camera RNG, cohort/shape/cache, seed0,
global32=2x16, AdamW2e-4/betas0.9,0.99/no decay/clip1,1000-step warmup then
cosine to zero at fresh200k, BF16 model/FP32 motion I/O and bridges, TF32 off,
checkpoint retention and W&B streaming.

Joint remains a supported diagnostic mode with sampling probability0. The
unchanged disposable acceptance checks still exercise its native camera and
caption invariants; those disposable weights never initialize production.
The actual per-draw training sampler never selects joint. A dataset subclass
restores this policy in every spawned DataLoader worker and rejects any joint
row, preventing worker initialization from restoring the parent40/40/20 mix.

Run `run_foundary.sh` with the original weight-one camera run's contract as
`--reference-contract`. Only task probabilities and experiment identity may
differ. Contracts hash all unchanged parent packages and the new adapter.
Actual two-GPU/microbatch16 acceptance and fresh4/reload2 probes precede fresh
production; old joint-training checkpoints cannot be resumed as this run.
Use this package's `sample.py` for strict checkpoint recipe validation.

User requested replacing the scratch-motion run on GPUs5,6. Preserve its full
stop checkpoint and all other active training/evaluation jobs. Live operation
receipts supersede launch notes. Inputs and outputs remain node-local.
