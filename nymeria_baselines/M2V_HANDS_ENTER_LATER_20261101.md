# Hands absent initially, entering later: matched M2V comparison

Completed qualitative follow-up on the same Notion page, cohortB:
https://app.notion.com/p/40bd2a44f44b80c5a349ca6df079d9be .
The original hand-visible cohortA is preserved, including all33 verified files.

Operation:
`/mnt/local/jinhyung.kim/cosmos_storage/operations/eval_m2v_hands_enter_later_20261101T213810Z`.
Pointer: `cosmos_storage/operations/m2v_hands_enter_later_current_ops.txt`.

## Selection and frozen protocol

Shortlist48 distinct test recordings from the eligible aligned test cohort;
visually inspect uncropped initial frames, including borders, and later GT RGB.
Lock eight cases and two seeds before new predictions. The camera wearer's
hands are absent at frame0 and visibly enter later. Background peers' hands are
not the wearer. Later review frames below establish visibility; they are not
an exact first-onset annotation. All inputs remain the original full images.

| Case | Source ID | Confirmed later visible GT frames |
| --- | --- | --- |
| Shelf/toy reach | `001941_52ee6d1ec2810f82` | 72 |
| Kitchen-island object | `001845_636d87a6fd2d12da` | 48,72 |
| Blanket pull | `002245_4b2ab863ca871cfe` | 72 |
| Laptop reach | `003180_804a7f937522903f` | 48,72 |
| Laundry closet | `009343_4cdf4ade84eab6d8` | 48,72,96 |
| Badminton hand/racket | `011569_8d81da9655b69bd0` | 24,48 |
| Hand gesture | `009239_59f32b93c3fd6241` | 72 |
| TV remote | `000803_4b186214f1ed44ef` | 24,48 |

`selected_cases.json`, `cases.json`, `INPUTS_VERIFIED.json`, initial-frame review
and GT reference sheets retain exact recording/start/seed and source hashes.
Native first-frame PNGs are uncropped and match their corresponding GT RGB.

Use the identical five specialists/checkpoints and inference recipe documented
in `M2V_HANDS_CFG2_20261101.md`: original two-task200k, equal-thirds joint200k,
hidden FiLM200k, trajectory FiLM180k and original matching Phase1 EMA200k.
All frozen source/normalization identities and strict bridge loads are retained.
No checkpoint transfers or new model training are needed for this follow-up.

T97/20fps/256; first RGB frame only; empty captions; UniPC35/shift10. Phase3
uses the user-approved whole-bridge-disabled velocity guidance2; Phase1 uses
camera-token omission at guidance2. All methods share the realized initial
Gaussian, with16 bitwise Phase1 audits and matching Phase3 per-output hashes.
Motion and neutral shape are supplied to Phase3; matching raw9-D RGB camera
actions are supplied to Phase1. Only full GT motion, not future GT video, is
provided to M2V. Specialists stay frozen and TF32 stays disabled.

## Execution and verified artifacts

The user previously authorized any unoccupied GPU. After fresh ownership/
NVML checks, independent workers use sep-2 GPU2 for Phase1, GPU4 for hidden
and original, GPU5 for trajectory and joint. Per-GPU locks and occupied-device
refusal remain enabled. Existing trainers on0/3 and1/7 and scoring on6 are
preserved. All evaluation workers finish and release2/4/5.

All80 new clips pass acceptance, finite-output and97-frame checks. Export16
six-panel videos (GT plus all five methods), eight GT initial/later visibility
sheets, eight fixed-crop hand details and a protocol/provenance ZIP. Both seeds
are retained; no result-based replacement, best-seed choice or recentering.

Eight new native Notion toggles CaseB01–B08 contain both comparison videos and
detail/GT stills. All33 new files are downloaded and SHA256-matched to local
exports. Final readback verifies16 total toggles,32 videos,32 images and two
ZIPs across cohortsA/B, with every original media reference preserved.

`MEDIA_COMPLETE.json`, `publication.json=published_verified`,
`remote_verification.json` and final `COMPLETE.json` establish completion.
This selected qualitative cohort supports inspection of missing/premature/late
hands, hand placement, camera motion and scene consistency; different task
exposures and checkpoint ages remain relevant to interpretation. FiLM camera
modulation is inactive for M2V, while its learned modality bridges remain active.
