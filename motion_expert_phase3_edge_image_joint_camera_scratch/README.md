# Camera prediction with a motion expert trained from scratch

Matches `motion_expert_phase3_edge_image_joint_camera`, with two approved
changes: no learned Phase2 motion weights, and motion parameters train together
with the same seven bridges. The Phase1 Cosmos reasoner/generator, LoRA and
video/camera projections remain pretrained and frozen. Motion layers, heads,
normalization and time/shape embeddings use the original Phase2 constructor's
fresh initialization, including its zero output projection.

The `--phase2` input is a small tensor-free artifact containing the reference
camera run's architecture/normalization contract and source-contract SHA256.
A trained Phase2 checkpoint or any tensor inside that artifact is rejected.
No learned Phase2 weights are loaded during model construction.

Unchanged:112503 windows; T97/20fps/256; V2M40%, M2V40%, initial-image joint20%;
bidirectional bridges and clean motion E(0); global32=2x16; AdamW2e-4,
betas0.9/0.99, weight decay0, clip1 across all trainable parameters;
warmup1000/cosine-to-zero at200k; native physical motion +10 video velocity
loss; same captions/shapes/noise clocks and RNG. Every task retains96 noisy-GT
camera tokens in native generator attention. Camera readout is detached and
has no loss. Same video-motion bridge scope; no direct camera-motion edges.

Checkpoint files include complete `motion_state`, `bridges`, optimizer,
schedule, per-rank RNG and data cursor. The native two-GPU preflight verifies
fresh initialization, no Phase2 weights, actual motion updates, frozen Cosmos,
all camera audits and complete-state save/reload/continue. Fresh4/reload2 probes
precede fresh production; disposable weights never initialize production.
All data/outputs are local NVMe. W&B scalar logging is required.

Use `run_foundary.sh` with the same flags as the camera run, the metadata-only
`--phase2`, the camera run's saved `--reference-contract`, idle `--gpus` and a
new local `--out`. `sample.py` restores both trained motion and bridge weights
before using the unchanged sampler; inference cameras start from pure noise.
