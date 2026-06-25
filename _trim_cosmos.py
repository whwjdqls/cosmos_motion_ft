#!/usr/bin/env python
"""Trim COSMOS.md in place to the sections relevant to adding a human-motion modality
to the Cosmos3 generator, prepending a finetuning-assessment header. Backs up the
original to COSMOS_full.md.bak first."""
import os, shutil

SRC = "/home/jungbin_cho/cosmos_motion_ft/COSMOS.md"
BAK = "/home/jungbin_cho/cosmos_motion_ft/COSMOS_full.md.bak"

NOTES = r"""## Finetuning notes (this project) — added by analysis

This COSMOS.md has been **trimmed** to the sections relevant to our project: adding a **3D human-motion
modality to the Cosmos3-Nano GENERATOR** (text -> motion, reasoner frozen). Full paper: arXiv:2606.02800
(complete original preserved at `COSMOS_full.md.bak`).

### Is our finetuning strategy okay? (assessment vs the paper)
**Verdict:** the *structure* matches Cosmos's design; some *hyperparameters* diverge from Cosmos's own
action-modality recipe (see **Sec 4.2.5 Robot Policy Post-Training**, the direct analog).

**Matches Cosmos:** freeze the reasoner + train only the generation (`_moe_gen`) pathway; add the new
modality via freshly-initialized projection heads (= robot policy's freshly-init action encoder/decoder/
embeddings); rectified-flow with velocity target `v = eps - x0` and x0 reconstruction; one token per frame
of the normalized continuous vector; bf16, grad-clip 1.0, no EMA. Our per-block smooth-L1 + FK loss is a
sensible motion-specific upgrade.

**Representation difference (important):** Cosmos action = **delta / relative SE(3) pseudo-actions**
(per-frame pose *deltas*, normalized to [-1,1]); ours = **absolute kinematic motion** (369-d kimodo rep:
joint rot6d + positions + velocities + foot contacts, z-scored). So Sec 4.2.5 is a **structural** analog only
(how to attach a continuous per-frame modality + the freeze/LR recipe) — keep kimodo's kinematic rep + FK
loss; do **not** adopt Cosmos's delta-action rep or plain MSE.

**Divergences to consider for a future run (NOT applied to the live runs):**
1. **LR ~10x low** — Cosmos robot policy uses base **2e-4 + 5x on the freshly-init heads (~1e-3)**; we use 2e-5, 1x.
2. **No 5x multiplier** on the new motion heads.
3. **Timestep**: we use Uniform(0,1); Cosmos uses **logit-normal + rectified-flow shift** (biases toward higher noise).
4. **Inference**: Cosmos action uses **shift~5, guidance=1**; our sampler default is shift=1, cfg=2.5 (now both
   selectable in `sample_motion.py` via `--shift` / `--cfg`).
5. **LoRA is off-recipe** — Cosmos full-finetunes the generator; projector/LoRA-only is reported to degrade.
6. Constant LR vs Cosmos warmup+decay; no 10% CFG text-dropout.

**Glossary.** *logit-normal*: `t = sigmoid(u)`, `u ~ Normal(mu, sigma^2)` -> concentrates training on mid-range
noise. *rectified-flow shift*: `sigma = s*tbar/(1 + (s-1)*tbar)`, `tbar = 1-t`, `s >= 1` (s=1 identity; s>1 ->
higher noise; Cosmos uses 3/5/10 by resolution, ~5 for action).

---
*Below: the trimmed Cosmos 3 paper — only the sections relevant to the generator + action/motion modality.*

"""

# KEEP blocks (1-indexed inclusive line ranges from the header map):
#  - 335-1001 : Sec 2 Model Architecture (encoders incl. 2.1.3 Action, token arrangement, MoT, pos-emb, variants)
#  - 1619-1722: Sec 3.2.3 Action data + normalization
#  - 1790-2119: Sec 4.2 Generator Training (pre/mid/post-training incl. 4.2.5 Robot Policy)
#  - 4780-4854: Sec 6.3.1 Generation Guide (action sampling: steps, shift, guidance)
KEEP = [(335, 1001), (1619, 1722), (1790, 2119), (4780, 4854)]
SEP = "\n\n---\n*[... non-relevant sections omitted — see COSMOS_full.md.bak ...]*\n\n"

with open(SRC) as f:
    lines = f.readlines()
print(f"original lines: {len(lines)}")

if not os.path.exists(BAK):
    shutil.copyfile(SRC, BAK)
    print(f"backed up -> {BAK}")

out = [NOTES, lines[0]]  # title line
for k, (a, b) in enumerate(KEEP):
    if k > 0:
        out.append(SEP)
    out.append("".join(lines[a-1:b]))   # inclusive
new = "".join(out)
with open(SRC, "w") as f:
    f.write(new)
print(f"trimmed COSMOS.md written: {len(new.splitlines())} lines "
      f"(kept {sum(b-a+1 for a,b in KEEP)} content lines + notes)")
