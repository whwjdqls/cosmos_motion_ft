# Nymeria proportional — foot skating analysis

Foot skating = mean 3D speed (cm/s) of a foot joint during frames it's in floor
contact. Contact = `foot_detect_from_pos_and_vel` (vel < 0.15 m/s AND height <
0.10 m), the same gate `KimodoMotionRep` uses. Computed on the 4 foot joints
(LeftFoot, LeftToeBase, RightFoot, RightToeBase). Identical method to the
BONES-SEED proportional check.

Scripts: `foot_skating_proportional.py` (full set, sharded), `_skate_breakdown.py`
(locomotion/height split), `_skate_77vs30.py` (collapse check), `_skate_source.py`
(SOMA source, env `soma`). fps verified correct end-to-end (`compute_vel_xyz`
scales by fps → m/s; skate speed = ‖Δpos‖·fps·100 → cm/s; all NPZs are 20 fps).

## Headline numbers (cm/s)

| dataset / method | mean | median | p95 |
|---|---|---|---|
| **Nymeria proportional** (all 732 seq, our kimodo data) | **4.05** | 3.35 | 9.82 |
| Nymeria — **SOMA source** (`layer.pose`, 60-seq sample) | **4.07** | 3.36 | 9.98 |
| Nymeria — **RAW SMPL fit** (`smplx` forward, 60-seq sample) | **4.06** | 3.39 | 9.80 |
| Nymeria — our 77-joint FK (120-seq sample) | 3.81 | 3.16 | 9.20 |
| Nymeria — our 30-joint FK (same sample) | 3.81 | 3.16 | 9.20 |
| BONES-SEED proportional (800-seq sample, same method) | 1.21 | 0.40 | 5.53 |

Chain: **raw SMPL (4.06) → SOMA (4.07) → kimodo (4.05)** — identical at every step.
The jitter originates in the raw SMPL fit (legs inferred from head+wrists); neither
SMPL→SOMA nor SOMA→kimodo adds any. Scripts: `_skate_smpl.py`, `_skate_source.py`.

## Conclusions

1. **The skating is inherent to the SMPL/SOMA source, NOT our conversion.**
   The SOMA source (`SOMALayer.pose`, the true LBS human joints) skates 4.07 cm/s
   — essentially identical to our converted data's 4.05. Our kimodo-77 FK
   reproduces the source joints to ~2e-6 m, so by construction it can't add
   skating, and the direct source measurement confirms it.

2. **The 77→30 joint collapse does NOT touch the feet.** 77-joint and 30-joint FK
   give bit-identical skating (3.81 / 3.81). The leg chain (Leg→Shin→Foot→Toe) is
   intact in SOMASkeleton30; the collapse only affects spine/hands.

3. **Why ~3× higher than BONES-SEED** (4.05 vs 1.21): Nymeria is real-world Aria
   capture whose SMPL fit observes only head + wrists — **legs are inferred**, so
   "planted" feet never sit perfectly still (median ~3 cm/s ≈ 1.6 mm/frame jitter).
   BONES-SEED is clean BVH mocap (planted feet ~0.4 cm/s).

4. **It is NOT caused by sitting / floating sequences** (a hypothesis we tested):
   - Locomotion split: stationary (92% of contacts) skates 3.66, *moving* skates
     5.71 — walking is the higher one, not sitting.
   - Root-height split: sitting (<0.7 m) is the *lowest* at 3.66; standing 3.87;
     elevated 4.21.
   - Floating frames don't inflate it — when feet are >10 cm up they fail the
     contact gate and are excluded entirely.
   - A small tail (3.8% of contacts >10 cm/s = feet mid-swing briefly mis-flagged
     as contact) contributes ~25% of the total.

## Contact detection vs grounding (multi-floor caveat)

Contact uses an absolute height gate (foot < 0.10 m), and the data is grounded so
the **whole-sequence** lowest foot = 0. In multi-floor captures (e.g. kirk on an
upper floor, feet ~2.9 m) or segments where the SMPL fit sits feet a bit high
(e.g. dawn walking, feet ~0.14 m above the global floor), feet never enter the
0.10 m band → **no contact detected** even while clearly walking. Fix for
per-segment / per-window metrics: ground to the LOCAL floor (subtract the
window's own min foot) before contact detection (`skating_pool(..., ground=True)`
in `viz_with_text.py`). After this, dawn's "walks" segment reports 4.2 cm/s (220
contacts), kirk's upper-floor segment 9.1 cm/s (92 contacts).

Implication: the **aggregate 4.05 cm/s** is computed on globally-grounded data, so
it only counts contacts near the ground floor and under-samples upper-floor /
raised-segment frames. Per-segment values are consistent (~4 cm/s), so the
conclusion is unchanged, but a fully complete aggregate would ground per-window.

## Reducing it

**kimodo's built-in postprocess (`post_process_motion`, foot-contact IK lock) does
NOT meaningfully help here.** Tested on 8 sequences (`_test_postprocess.py`, after
`pip install MotionCorrection/`): skating before 6.48 → after 6.11 cm/s = **~6%
mean / 8% median**, and inconsistent — it improves sequences with dense contacts
(e.g. 6.04→4.61) but *worsens* ones with sparse/noisy contacts (e.g. 2.41→9.56,
where its root correction injects motion). It's designed to clean up already-good
*generated* motion given reliable contacts, not to fix pervasive SMPL-fit leg
jitter with noisy auto-detected contacts.

To actually cut it would need a dedicated foot-IK with robust contact labels
(detect planted spans, hard-lock the foot to a single point across each span, then
IK the leg). Out of scope; flagged. Given the jitter is inherent to the SMPL
source and identical across SMPL/SOMA/kimodo, the cleaner long-term fix is upstream
(better leg observation in the SMPL fit).
