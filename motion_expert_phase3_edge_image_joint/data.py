"""Reuse the immutable train cache and sampler; remove all Phase3 captions."""
from motion_expert_phase3_edge_joint.data import Phase3Dataset as OriginalDataset
from .policy import captions_for_modes


class Phase3Dataset(OriginalDataset):
    def __setstate__(self, state):
        # Spawned DataLoader workers need the same process-local I/O adapter
        # and task policy as the trainer; no cache/index bytes are rewritten.
        self.__dict__.update(state)
        from .train import configure_local_paths
        from .policy import configure
        configure_local_paths()
        configure()

    def __getitem__(self, key):
        row = super().__getitem__(key)
        return {**row, "caption": captions_for_modes([row["caption"]], [row["mode"]])[0]}
