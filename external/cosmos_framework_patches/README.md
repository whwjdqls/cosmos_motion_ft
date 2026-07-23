# Local patches to NVIDIA cosmos-framework (REQUIRED for native Phase-1 training)

The `/home/jungbin_cho/cosmos-framework` checkout used for every native run was
`github.com/NVIDIA/cosmos-framework` @ `82f8229` **plus the uncommitted changes captured
here**. Without them, the Phase-1 launchers in `native_phase_training/` do not reproduce.

## What the patches do

1. **`lora_keep_trainable_modules`** (`configs/base/defaults/model_config.py` +
   `model/vfm/omni_mot_model.py`): new config knob; after LoRA injection freezes all
   non-LoRA params, re-enables `requires_grad` on modules whose names match the
   comma-separated substrings. Every Phase-1 experiment sets it to
   `action2llm,llm2action,action_modality_embed` — **without this, the camera action
   heads silently do not train.**
2. **`SAVE_TRAINABLE_ONLY` / `trainable_state_dict`** (`checkpoint/dcp.py`): env-gated
   LoRA-adapter-only DCP saving (filters state-dict keys by substring, NOT
   named_parameters FQNs — FSDP mangles the deep LoRA names). Off by default; full-state
   saving unchanged.
3. **TensorBoardLog callback** (`callbacks/tensorboard_log.py` new file +
   registration in `configs/base/defaults/callbacks.py`): per-run TB event files
   (`run_latent_train.py` sets its `log_dir`).
4. **Pixel-path experiment registration** (`configs/base/config.py` import +
   `configs/base/experiment/action/posttrain_config/world_camera_nymeria_nano.py` new
   file + `examples/toml/sft_config/world_camera_nymeria_repro.toml`): the older
   raw-pixel Nymeria camera experiment (superseded by the cached-latent path in
   `native_phase_training/`, but kept for reference; note it still carries the
   finite-stream livelock documented in `native_phase_training/AUDIT.md`).
5. README banner describing the fork (cosmetic).

## How to reapply on a fresh machine

```bash
git clone https://github.com/NVIDIA/cosmos-framework
cd cosmos-framework
git checkout 82f8229            # see base_commit.txt
git apply /path/to/external/cosmos_framework_patches/local_changes.patch
cp -r /path/to/external/cosmos_framework_patches/untracked/* .
```

If checking out a newer upstream, hunks 1–2 are the ones to port carefully
(`model_config.py` / `omni_mot_model.py` / `dcp.py`); upstream may have moved the LoRA
injection or DCP save sites.
