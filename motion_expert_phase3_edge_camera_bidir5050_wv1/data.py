"""Apply the two-task policy in every spawned worker, preserving data order."""
from motion_expert_phase3_edge_image_joint_camera.data import Phase3Dataset as CameraDataset, collate
from .policy import configure

class Phase3Dataset(CameraDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        configure()

    def __setstate__(self, state):
        super().__setstate__(state)
        configure()

    def __getitem__(self, key):
        row = super().__getitem__(key)
        if row['mode'] not in {'video2motion', 'motimg2video'}:
            raise RuntimeError('Disabled joint task entered the training loader')
        return row
