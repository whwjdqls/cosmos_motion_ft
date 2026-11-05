# Minsu baseline operation scripts

These are source copies of the minsu launch/evaluation glue. The live originals
remain in `/home/minsu1206/cosmos_launch` and the corresponding Scratch2 operation
directories. They invoke the shared implementation at
`/home/whwjdqls99/cosmos_motion_ft`, except for the explicitly authorized EgoEgo
integration in the minsu checkout. They do not contain weights, data, tokens,
environment directories or runtime logs.

- `egoego_train_20261104.sh`: validated two-GPU smoke/resume and production
  launcher for `nymeria_baselines.egoego_shape`; currently bound to job2670587.
- `egoallo_c2m_fulltest_20261105.py`: native supplied-shape EgoAllo sampling
  across the exact frozen11,911-case cohort, with fixed per-case seeds and
  deterministic replay, coordinate and FK checks. Two processes split the cases.
- `egoallo_c2m_handoff_20261105_v2.sh`: wait for final200k training and trainer
  exit, sample inside existing2667149, then invoke the shared full-test scorer.
  Keep the Python sampling driver beside this shell launcher. GPU-idle checks
  run inside the allocated step, after the final checkpoint appears.
- `egoallo_c2m_report_20261105.py`: validate completed score/cohort/checkpoint
  identities and render a Notion-ready report. It performs no remote write;
  publication used the connected Notion tools and remote readback.
- `score_requested_common16.py`: additive REVIV common16 UPE/LPE/sJPE scoring
  over existing predictions, using the original operation's `protocol.py` and
  `score.py`. `REVIV_OPERATION` overrides the default existing Scratch2 root.
  The original predictions and scores remain unchanged.
- `request_gpu_pairs_20261105.sh`: queue one two-GPU A100 or PRO6000 allocation
  per invocation,16 CPUs/128GiB/three days; retain it idle after allocation.
  The family-specific QoS follows the partition's allowed QoS. Existing per-pair
  locks prevent duplicate queue workers. Requests were explicitly authorized.

These scripts intentionally record the launch-time paths/job identities. Review
and update a new copy when using another allocation; do not edit a shell file
that an active worker is reading. GPU work must use the owning user's allocation.
GPU indices reported by Slurm and host-wide NVML can differ: inspect GPU UUIDs
inside the specific `srun` allocation when verifying ownership or availability.

See `../MINSU_BASELINES_20261105.md` for completed metric scopes and artifacts.
