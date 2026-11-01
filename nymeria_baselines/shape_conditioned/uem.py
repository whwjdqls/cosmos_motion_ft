"""Native UEM with clean shape input and a motion-only diffusion state."""
from pathlib import Path
import argparse
import json
import os
import sys

from .common import require_compute, contract, save_contract, atomic_json


def components(upstream):
    sys.path.insert(0, str(upstream/'UniEgoMotion'))
    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader
    from dataset.nymeria_soma_dataset import NymeriaSOMADataset
    from module.uem_module import UEM_Module

    class ShapeDataset(NymeriaSOMADataset):
        def __getitem__(self, index):
            item = super().__getitem__(index)
            full = item['misc']['motion']
            assert full.shape == (97, 230)
            # Exact Phase2 input: all30 neutral joints, centroid-centered in meters.
            # No PCA truncation, pose, dynamic joints or motion normalization.
            import numpy as np
            neutral = np.array(self._array(self.rows[index]['array'],'neutral'),dtype=np.float32,copy=True)
            assert neutral.shape==(30,3)
            neutral -= neutral.mean(axis=0,keepdims=True)
            item['y']['neutral_joints'] = torch.from_numpy(neutral)
            item['misc']['motion'] = full[:, :220].clone()
            return item

    class ShapeModule(UEM_Module):
        def training_step(self, batch, batch_idx, mode='train'):
            # Preserve native diffusion loss. Native per-bin conditional
            # sync_dist logs can deadlock when two ranks sample different bins.
            x = batch['misc']['motion']
            t, weights = self.schedule_sampler.sample(x.shape[0], self.device)
            losses = self.diffusion.training_losses(self.model, x, t, model_kwargs={'y':batch['y']})
            loss = (losses['loss']*weights).mean()
            for k, v in losses.items():
                self.log(f'{mode}/{k}', v.mean(), on_step=True, on_epoch=True,
                         sync_dist=True, batch_size=x.shape[0])
            return loss

    class Data(pl.LightningDataModule):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
        def setup(self, stage=None):
            self.train = ShapeDataset(self.cfg.DATA.DATA_DIR, 'train')
            self.val = ShapeDataset(self.cfg.DATA.DATA_DIR, 'val')
        def loader(self, dataset, shuffle):
            return DataLoader(dataset, batch_size=self.cfg.DATA.BATCH_SIZE,
                              shuffle=shuffle, num_workers=4, persistent_workers=True,
                              pin_memory=True)
        def train_dataloader(self):
            return self.loader(self.train, True)
        def val_dataloader(self):
            return self.loader(self.val, False)
    return ShapeDataset, ShapeModule, Data


def make_config(upstream, cache, output, world=2):
    from config.defaults import get_cfg_defaults
    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(upstream/'UniEgoMotion/config/nymeria_soma.yaml'))
    cfg.DATA.REPRE_TYPE = 'soma_body220_shape'
    cfg.DATA.COND_BETAS = False
    cfg.DATA.DATA_DIR = str(cache)
    cfg.DATA.BATCH_SIZE = 64//world
    cfg.TRAIN.NUM_GPUS = world
    cfg.TRAIN.EXP_PATH = str(output)
    cfg.TRAIN.USE_CKPT_LR = True
    cfg.MODEL.CKPT_PATH = str(output/'last.ckpt') if (output/'last.ckpt').exists() else None
    cfg.freeze()
    return cfg


def main():
    p = argparse.ArgumentParser()
    for name in ('upstream', 'cache', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--max-steps', type=int, default=-1)
    p.add_argument('--stop-after', type=int, default=-1,
                   help='Acceptance-only early stop; does not change the experiment recipe')
    a = p.parse_args()
    require_compute()
    _, Model, Data = components(a.upstream)
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.strategies import DDPStrategy
    from pytorch_lightning.plugins.environments import TorchElasticEnvironment
    from module.ema import EMA
    pl.seed_everything(62, workers=True)
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world != 2:
        raise RuntimeError('Production/smoke training requires exactly two torchrun ranks')
    cfg = make_config(a.upstream, a.cache, a.output, world)
    value = contract(a.cache, a.upstream, 'UEM trajectory + supplied shape', 64, '350 epochs')
    value.update(lr=3e-5, weight_decay=.01, seed=62, state_dim=220, shape_dim=90,
                 shape='Phase2 SOMA30 neutral_joints minus all30-joint centroid, meters, no PCA',
                 shape_conditioning='Phase2-style Linear90->768/GELU/LayerNorm/Linear768->768 plus type and modality embeddings; clean in-context token',
                 max_steps=a.max_steps, native_trajectory_masking=True)
    if int(os.environ.get('RANK','0')) == 0:
        save_contract(a.output, value)
    model = Model(cfg)
    trainer = pl.Trainer(accelerator='gpu', devices=2,
        strategy=DDPStrategy(find_unused_parameters=True), plugins=[TorchElasticEnvironment()],
        default_root_dir=str(a.output),
        logger=TensorBoardLogger(str(a.output), name='', version='', default_hp_metric=False),
        callbacks=[ModelCheckpoint(dirpath=str(a.output), every_n_epochs=10, save_top_k=-1,
                                   save_last=True), LearningRateMonitor('step'), EMA(.999)],
        max_epochs=350, max_steps=a.stop_after if a.stop_after>0 else a.max_steps, num_sanity_val_steps=2,
        log_every_n_steps=50, check_val_every_n_epoch=1, gradient_clip_val=1.,
        gradient_clip_algorithm='norm', enable_progress_bar=False)
    trainer.fit(model, datamodule=Data(cfg), ckpt_path=cfg.MODEL.CKPT_PATH)
    trainer.save_checkpoint(a.output/'last.ckpt')
    if trainer.is_global_zero:
        atomic_json(a.output/('STOPPED.json' if a.stop_after>0 else 'COMPLETE.json'), {'global_step':trainer.global_step,
                                              'epochs':trainer.current_epoch,
                                              'smoke':a.stop_after>0 or a.max_steps>0})


if __name__ == '__main__':
    main()
