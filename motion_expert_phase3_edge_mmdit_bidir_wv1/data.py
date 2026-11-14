"""Immutable training cache with the same policy in spawned workers."""
from motion_expert_phase3_edge_joint.data import Phase3Dataset as BaseDataset


class Phase3Dataset(BaseDataset):
    def __setstate__(self, state):
        self.__dict__.update(state)
        from motion_expert_phase3_edge_image_joint.train import configure_local_paths
        from .policy import configure
        configure_local_paths()
        configure()
