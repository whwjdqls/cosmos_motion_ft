"""Original immutable-cache sampling, with the weight-one M2V caption policy."""
import random
from motion_expert_phase3_edge_joint.data import Phase3Dataset as BaseDataset
from motion_expert_phase3_edge_joint.settings import RECIPE, sample_task
from .policy import captions_for_modes, configure


class Phase3Dataset(BaseDataset):
    def __setstate__(self, state):
        self.__dict__.update(state)
        from motion_expert_phase3_edge_image_joint.train import configure_local_paths
        configure_local_paths()
        configure()

    def __getitem__(self, key):
        row = super().__getitem__(key)
        draw, index = key if isinstance(key, tuple) else (int(key), int(key))
        rng = random.Random(RECIPE["seed"] + 1000003 * (draw + 1))
        assert sample_task(rng) == row["mode"]
        caption = "" if row["mode"] == "video2motion" or rng.random() < .1 else self.rows[index]["caption"]
        return {**row, "caption": captions_for_modes([caption], [row["mode"]])[0]}
