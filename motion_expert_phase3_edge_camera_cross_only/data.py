from motion_expert_phase3_edge_camera_directional_nojoint.data import Phase3Dataset as BaseDataset, collate
from .policy import configure


class Phase3Dataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        configure()

    def __setstate__(self, state):
        super().__setstate__(state)
        configure()
