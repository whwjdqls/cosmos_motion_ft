# Yonsei Scratch2 access-time refresh, 2026-11-08

Scope: `/scratch2/whwjdqls99`, including hidden entries. The full refresh ran
on owned CPU allocation2717529, `dell_cpu` / `cnode03`. It did not traverse
symlinks, cross filesystem boundaries, delete files, or change file contents.
Audit files are kept outside Scratch2 at
`/home/whwjdqls99/.codex/maintenance/scratch2_refresh_20261108`.

## Preserve training cache identities

Phase3 records each latent's size and modification time in its cache index;
`motion_expert_phase3_edge_joint/data.py` checks those identities when reading.
Use access-time refreshes (`touch -a` semantics), preserving modification times.
The audited walker uses directory file descriptors, no-follow opens, and Linux
`utimensat` with `UTIME_NOW` for access time and `UTIME_OMIT` for modification
time. A real filesystem/symlink probe verifies those semantics before the walk.

An initial ordinary timestamp pass was stopped before it reached the Cosmos
folder after this identity dependency was identified. It had already refreshed
entries in earlier projects/environment directories. The entire root was then
processed with the access-only method. No Cosmos training-cache identities
were rewritten to bypass validation.

All108,760 unique latent paths in the current Phase3 cache index (112,503 rows)
were subsequently checked: size and modification time match every saved
identity, with zero mismatches. The check is read-only; its result is
`CACHE_IDENTITY_AUDIT.json` in the audit directory.

## Permission handling and audit

The initial access-only walk reports permission errors separately. Writable
foreign-owned directories outside Cosmos can receive an ordinary timestamp
refresh; this changes directory timestamps only. No ownership or permissions
are changed. Files inside readable directories are processed even when the
parent directory's timestamps cannot be changed.

The final receipt combines `COMPLETE.json`, `DIRECTORY_REPAIR.json` and the
cache audit. `FINAL.json` is the authoritative combined result; raw walk error
counts include any directory permission errors subsequently repaired.

## Git integration checks

The application/evaluation changes and GitHub master were merged without
changing the upstream model implementation. The only merge conflict was in
`AGENTS_ALL.md`; all local Yonsei and incoming server notes were retained.
All46 focused CPU tests passed using the standard repository launch environment,
and52 changed Python files passed syntax checks. Existing evaluation evidence
files were kept byte-for-byte, including an upstream archived file's trailing
blank line, to preserve its recorded hashes.

## Completed result

Finished November8 at17:14KST; the full pass took24minutes29seconds:

| Entry type | Refreshed |
|---|---:|
| Regular files | 2,429,180 |
| Symlinks, without following targets | 26,755 |
| Directories | 313,260 |
| Other filesystem entries | 34 |

All regular files succeeded. Eighteen directories owned by another account
could not have their timestamps changed; their readable contents were still
processed. These are cached-package, W&B/OpenPI, LeRobot and Robosuite parent
directories outside the Cosmos tree. Exact paths and ownership IDs are in
`FINAL.json` / `DIRECTORY_REPAIR.json`. One additional writable foreign-owned
directory was refreshed with ordinary directory-only timestamp semantics;
file modification times were preserved throughout the full pass.

The CPU allocation was released after the combined receipt was written. No
recurring maintenance schedule was installed. The original pass logs, corrected
access-only walk, kernel/symlink probe, cache audit and test outputs remain in
the home audit directory for review.
