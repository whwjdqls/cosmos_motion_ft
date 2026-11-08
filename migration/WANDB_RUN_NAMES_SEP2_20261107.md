# Active sep-2 W&B display names

On2026-11-07 the user requested following the current sep-1 convention.
Reference runs: p3-d2d167e7b78773091498 (fresh directional/video1),
p3-244621e199b8191a46c6 (continued directional/video10), and
p3-54fb3e983641efb7c839 (continued bidirectional/video10).

Use:

```text
[node] P3 MMDiT {directionality} | V2M{percent}/M2V{percent}/J{percent} | videoW{weight} | EMA{decay} start{step} ({fresh|continued}) | NoText | LocalAttn BF16 | T{frames} GBS{batch}
```

Fill every field from the actual recipe/checkpoint. The sep-2 joint runs use
40/40/20, rather than copying sep-1's50/50/0. Late EMA origins are actual
optimizer steps, not an approximate50k label or invented averaged history.

- p3-d47749076808b4bfecc2:
  `[sep-2] P3 MMDiT bidirectional | V2M40/M2V40/J20 | videoW10 | EMA0.999 start50017 (continued) | NoText | LocalAttn BF16 | T97 GBS32`
- p3-79f9ba1f501245ec7b18:
  `[sep-2] P3 MMDiT directional | V2M40/M2V40/J20 | videoW10 | EMA0.999 start50819 (continued) | NoText | LocalAttn BF16 | T97 GBS32`
- p3-aa6c658167d6bc2eec5f:
  `[sep-2] P3 MMDiT directional | V2M40/M2V40/J20 | videoW1 | EMA0.999 start0 (fresh) | NoText | LocalAttn BF16 | T97 GBS32`

Display names are metadata; keep run IDs, source directories, groups, tags,
scientific config and histories. No trainer/logger restart is needed to rename
an active run. Public API readback must verify the requested names and that
all other metadata/config remains unchanged.

Local audit: `/mnt/local/jinhyung.kim/cosmos_storage/operations/wandb_run_names_20261107T000012Z`.
Canonical map: `/mnt/local/jinhyung.kim/cosmos_storage/operations/WANDB_DISPLAY_NAMES.json`.
Each run also has `wandb_display_name.txt` and
`launch/wandb_state/display_name.json`.

The currently frozen logger takes its name from `--name` and does not read
these override files itself. When restarting a logger, explicitly pass the
saved display name instead of the long directory basename. Do not edit an
immutable training source snapshot or restart a trainer merely to rename it.
