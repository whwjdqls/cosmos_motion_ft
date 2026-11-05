# EgoEgo with camera trajectory and supplied SOMA shape

This integration uses the original EgoEgo transformer and diffusion structure
from `lijiaman/egoego_release`, pinned at
`4dae8cb67b453f0a0042809bf37aafad5604615b`. `upstream.py` executes its model
definitions without the unused SMPL/AMASS visualization imports. The upstream
checkout is unmodified; its MIT license is included as `LICENSE.EGOEGO`.

The observed condition is the measured camera trajectory, used in the native
head-conditioning slots. The anatomical head remains part of the motion target.
A separate clean shape token supplies the full centered SOMA-30 neutral skeleton
(90 values). The motion output has 23 joints: 69 global position values followed
by 138 global row-convention rotation-6D values per frame. Every clip has 97
frames at 20 FPS. Camera-only initial yaw/horizontal translation removal and
floor calibration define the coordinate frame; target motion does not define
the camera anchor.

The Nymeria split matches UniEgoMotion: 101,576 train, 10,927 validation and
11,911 test annotation windows. Statistics are fit on training windows only.
The shared cache and upstream paths are currently Yonsei-specific constants;
the command-line interfaces accept the prepared statistics and output paths.

## Training recipe

- Fresh weights, two GPUs, global batch64 = 2x32, accumulation1.
- AdamW LR3e-5, betas0.9/0.999, weight decay0.01, FP32, gradient clipping1.
- 350 epochs, 1,588 updates/epoch, 555,800 total updates.
- StepLR: multiply LR by0.1 at epoch300; EMA0.999.
- Native cosine diffusion with1,000 steps and uniform-timestep clean-motion MSE.
- Validation every epoch; rolling checkpoint every epoch and archive every10.

The schedule, LR and batch follow the supplied-shape UEM recipe. The target
representation and camera/shape conditioning above are EgoEgo-specific adapters.
No RGB or text is supplied. Sampling uses only camera, shape and random noise.

## Commands and validation

Run all model/data commands inside an owned GPU allocation. Install/activate the
existing compatible UEM environment and add this checkout to `PYTHONPATH`.

```bash
op=/scratch2/minsu1206/cosmos/operations/egoego_shape_20261104
python -m nymeria_baselines.egoego_shape.data --output "$op/data"
torchrun --standalone --nproc_per_node=2 \
  -m nymeria_baselines.egoego_shape.verify \
  --stats "$op/data/stats.npz" --out "$op/acceptance"
torchrun --standalone --nproc_per_node=2 \
  -m nymeria_baselines.egoego_shape.train \
  --stats "$op/data/stats.npz" --out "$op/training" --resume auto
```

The existing run passed nine independent geometry checks across all splits,
native no-shape network bitwise parity, finite two-rank backward/update checks,
identical updated replicas, nonzero camera/shape conditioning effects, and a
finite native1,000-step sample. A fresh4-update/reload2-update check restored
optimizer, EMA and scheduler state and verified further parameter changes.
Receipts are `acceptance/ACCEPTED.json` and `RESUME_ACCEPTED.json` in the operation.
These checks do not establish final model quality; training is ongoing.

`train.py` fingerprints every Python file in this package. Preserve those files
while resuming an existing run; an altered source/recipe contract is rejected.
Launch scripts and completed evaluation notes are in `../minsu_operations/`
and `../MINSU_BASELINES_20261105.md`.
