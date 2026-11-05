"""Shape-input UEM reconstruction, equally mixing RGB, trajectory and both."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from nymeria_baselines.shape_conditioned.common import contract, save_contract
from nymeria_baselines.uem_visual_cache import require_compute, atomic_json, sha

MODES = ("rgb_only", "traj_only", "rgb_traj")


def components(upstream):
    from nymeria_baselines.shape_conditioned.uem import components as shape_components
    ShapeDataset, ShapeModule, _ = shape_components(upstream)
    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader
    from nymeria_baselines.train_uem_variants import VisualDataset

    class MixedDataset(VisualDataset):
        def __init__(self, cache, features, split):
            super().__init__(cache, features, split, "recon")
            self.base = ShapeDataset(str(cache), split)

        def __getitem__(self, index):
            item = super().__getitem__(index)
            mode = int(torch.randint(3, ()).item()) if self.split == "train" else index % 3
            # Keep identical dictionary keys for default_collate. Explicit
            # all-one masks select native mask tokens, never zero observations.
            item["y"]["traj_mask"].fill_(int(mode == 0))
            item["y"]["img_mask"].fill_(int(mode == 1))
            item["misc"]["conditioning_mode"] = mode
            return item

    class Module(ShapeModule):
        def training_step(self, batch, batch_idx, mode="train"):
            loss = super().training_step(batch, batch_idx, mode)
            for i, name in enumerate(MODES):
                fraction = (batch["misc"]["conditioning_mode"] == i).float().mean()
                self.log(f"{mode}/condition_fraction/{name}", fraction,
                         on_step=True, on_epoch=True, sync_dist=True,
                         batch_size=len(batch["misc"]["conditioning_mode"]))
            return loss

    class Data(pl.LightningDataModule):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
        def setup(self, stage=None):
            self.train = MixedDataset(self.cfg.DATA.DATA_DIR, self.cfg.DATA.VISUAL_CACHE, "train")
            self.val = MixedDataset(self.cfg.DATA.DATA_DIR, self.cfg.DATA.VISUAL_CACHE, "val")
        def loader(self, dataset, shuffle):
            return DataLoader(dataset, batch_size=self.cfg.DATA.BATCH_SIZE, shuffle=shuffle,
                              num_workers=4, persistent_workers=True, pin_memory=True)
        def train_dataloader(self):
            return self.loader(self.train, True)
        def val_dataloader(self):
            return self.loader(self.val, False)
    return MixedDataset, Module, Data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "cache", "features", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--initialize-from", type=Path,
                   help="Optional model-only warm start; fresh optimizer/schedule")
    p.add_argument("--stop-after", type=int, default=-1)
    a = p.parse_args()
    require_compute()
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.strategies import DDPStrategy
    from pytorch_lightning.plugins.environments import TorchElasticEnvironment
    Dataset, Model, Data = components(a.upstream)
    from config.defaults import get_cfg_defaults
    from module.ema import EMA

    world, local = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if world != 2 or local not in (1, 2):
        raise RuntimeError("Requires exactly two GPU ranks; each rank stays inside its owned allocation")
    torch.set_num_threads(4)
    pl.seed_everything(62, workers=True)
    ready = __import__("json").loads((a.features / "READY.json").read_text())
    assert ready["cache_index_sha256"] == sha(a.cache / "index.json")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(a.upstream / "UniEgoMotion/config/nymeria_soma.yaml"))
    cfg.DATA.DATA_DIR = str(a.cache)
    cfg.DATA.REPRE_TYPE = "soma_body220_shape"
    cfg.DATA.COND_BETAS = False
    cfg.DATA.COND_IMG_FEAT = True
    cfg.DATA.COND_TRAJ = True
    cfg.DATA.IMG_FEAT_TYPE = "dinov2"
    cfg.DATA.VISUAL_CACHE = str(a.features)
    cfg.DATA.VISUAL_CACHE_FINGERPRINT = ready["fingerprint"]
    cfg.DATA.UEM_VARIANT = "recon"
    cfg.DATA.BATCH_SIZE = 32
    cfg.MODEL.FINETUNE_TYPE = "recon"
    cfg.TRAIN.NUM_GPUS = 2
    cfg.TRAIN.EXP_PATH = str(a.output)
    cfg.TRAIN.USE_CKPT_LR = True
    resume = a.output / "last.ckpt"
    cfg.MODEL.CKPT_PATH = str(resume) if resume.exists() else None
    cfg.freeze()
    value = contract(a.cache, a.upstream, "UEM supplied shape / mixed reconstruction modalities", 64, "350 epochs")
    value.update(conditioning={name: 1 / 3 for name in MODES},
                 shape="Always clean centered SOMA30 neutral joints [30,3], meters; identical to Phase3",
                 state_dim=220, shape_dim=90, shape_prediction=False, shape_loss=False,
                 lr=3e-5, weight_decay=.01, seed=62, ema=.999, grad_clip=1.,
                 per_gpu_batch=32, accumulation=1, global_batch=64,
                 task="Reconstruction of all97 frames; no forecasting/generation task", frames=97, fps=20,
                 masking="Exactly one mode per example, native explicit full-stream masks; no extra stochastic dropout",
                 validation="Deterministic window index modulo3 selects conditioning mode",
                 visual_cache=str(a.features), visual_identity=ready,
                 trainer_sha256=sha(Path(__file__)), visual_dataset_sha256=sha(Path(__file__).with_name("train_uem_variants.py")),
                 initialization="fresh random weights" if a.initialize_from is None else "model-only warm start; fresh optimizer/schedule",
                 initialize_checkpoint=str(a.initialize_from) if a.initialize_from else None,
                 initialize_checkpoint_sha256=sha(a.initialize_from) if a.initialize_from else None)
    if int(os.environ.get("RANK", "0")) == 0:
        save_contract(a.output, value)
    model = Model(cfg)
    if a.initialize_from is not None and not resume.exists():
        payload = torch.load(a.initialize_from, map_location="cpu", weights_only=False)
        old = payload["hyper_parameters"]["DATA"]
        assert old["REPRE_TYPE"] == "soma_body220_shape" and old["COND_IMG_FEAT"] and old["COND_TRAJ"]
        model.load_state_dict(payload["state_dict"], strict=True)
        del payload

    class Progress(Callback):
        def on_fit_start(self, trainer, module):
            import socket
            import torch.distributed as dist
            properties = torch.cuda.get_device_properties(module.device)
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")
            device_id = str(getattr(properties, "uuid", None) or visible[torch.cuda.current_device()])
            identity = dict(host=socket.gethostname(), device=device_id,
                            job=os.environ.get("SLURM_JOB_ID"), rank=dist.get_rank())
            ranks = [None] * dist.get_world_size()
            dist.all_gather_object(ranks, identity)
            assert len(ranks) == 2 and len({r["host"] for r in ranks}) == 1, ranks
            assert len({r["device"] for r in ranks}) == 2, ranks
            if trainer.is_global_zero:
                atomic_json(a.output / "DDP_LAYOUT.json", dict(ranks=ranks, same_physical_node=True,
                            global_batch=64, per_gpu_batch=32, accumulation=1))

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            if trainer.is_global_zero and (trainer.global_step <= 5 or trainer.global_step % 50 == 0):
                record = dict(global_step=trainer.global_step, epoch=trainer.current_epoch,
                              loss=float(outputs["loss"].detach()), world_size=trainer.world_size,
                              global_batch=64, per_gpu_batch=32)
                atomic_json(a.output / "progress.json", record)
                print(record, flush=True)

    trainer = pl.Trainer(accelerator="gpu", devices=local, num_nodes=world // local,
        strategy=DDPStrategy(find_unused_parameters=True), plugins=[TorchElasticEnvironment()],
        default_root_dir=str(a.output), accumulate_grad_batches=1,
        logger=TensorBoardLogger(str(a.output), name="", version="", default_hp_metric=False),
        callbacks=[ModelCheckpoint(dirpath=str(a.output), every_n_epochs=1, save_top_k=0, save_last=True),
                   ModelCheckpoint(dirpath=str(a.output), every_n_epochs=10, save_top_k=-1, save_last=False),
                   LearningRateMonitor("step"), EMA(.999), Progress()],
        max_epochs=350, max_steps=a.stop_after if a.stop_after > 0 else -1,
        num_sanity_val_steps=2, log_every_n_steps=50, check_val_every_n_epoch=1,
        gradient_clip_val=1., gradient_clip_algorithm="norm", enable_progress_bar=False)
    trainer.fit(model, datamodule=Data(cfg), ckpt_path=cfg.MODEL.CKPT_PATH)
    trainer.save_checkpoint(a.output / "last.ckpt")
    if trainer.is_global_zero:
        atomic_json(a.output / ("STOPPED.json" if a.stop_after > 0 else "COMPLETE.json"),
                    dict(global_step=trainer.global_step, epochs=trainer.current_epoch,
                         smoke=a.stop_after > 0, global_batch=64, world_size=2))


if __name__ == "__main__":
    main()
