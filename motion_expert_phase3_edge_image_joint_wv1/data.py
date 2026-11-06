from motion_expert_phase3_edge_image_joint.data import Phase3Dataset as BaseDataset


class Phase3Dataset(BaseDataset):
    def __setstate__(self, state):
        super().__setstate__(state)
        from .policy import configure
        configure()
