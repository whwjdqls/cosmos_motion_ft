"""Nymeria RGB/camera task ablations using the native UniEgoMotion trainer model."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path
import sys

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from nymeria_baselines.uem_visual_cache import atomic_json, frame_ids, require_compute, sha


class VisualDataset(Dataset):
    def __init__(self, cache, features, split, variant):
        from dataset.nymeria_soma_dataset import NymeriaSOMADataset
        self.base = NymeriaSOMADataset(str(cache), split)
        self.features, self.split, self.variant = Path(features), split, variant
        self.opened = OrderedDict()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        row = self.base.rows[idx]
        key = row['array']
        if key not in self.opened:
            prefix = self.features / 'records' / key
            self.opened[key] = (np.load(str(prefix)+'.frame_ids.npy', mmap_mode='r'),
                                np.load(str(prefix)+'.features.npy', mmap_mode='r'))
        self.opened.move_to_end(key)
        while len(self.opened) > 12:
            self.opened.popitem(last=False)
        ids, array = self.opened[key]
        wanted = frame_ids(int(row['start']))
        positions = np.searchsorted(ids, wanted)
        if np.any(positions >= len(ids)) or not np.array_equal(ids[positions], wanted):
            raise RuntimeError(f'Missing aligned DINO features: {row["uuid"]}@{row["start"]}')
        images = torch.from_numpy(np.array(array[positions], dtype=np.float32))
        if images.shape != (97, 1024) or not torch.isfinite(images).all():
            raise RuntimeError('Invalid visual conditioning')
        y = item['y']
        y['img_embs'] = images
        y['valid_img_embs'] = torch.ones(97, dtype=torch.long)
        task = 'recon'
        if self.variant == 'all':
            task = ('recon', 'fore', 'gen')[int(torch.randint(3, ()).item()) if self.split == 'train' else idx % 3]
        y['traj_mask'] = torch.zeros(97, dtype=torch.long)
        y['img_mask'] = torch.zeros(97, dtype=torch.long)
        if task == 'fore':
            y['traj_mask'][97//4:] = 1
            y['img_mask'][97//4:] = 1
        elif task == 'gen':
            y['traj_mask'][:] = 1
            y['img_mask'][1:] = 1
        if self.variant == 'RGB_recon':
            del y['traj']
            del y['traj_mask']
            item['misc'].pop('traj', None)
        item['misc']['task'] = task
        return item


class VisualDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        self.train_dataset = VisualDataset(self.cfg.DATA.DATA_DIR, self.cfg.DATA.VISUAL_CACHE,
                                           'train', self.cfg.DATA.UEM_VARIANT)
        self.val_dataset = VisualDataset(self.cfg.DATA.DATA_DIR, self.cfg.DATA.VISUAL_CACHE,
                                         'val', self.cfg.DATA.UEM_VARIANT)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.DATA.BATCH_SIZE, shuffle=True,
                          num_workers=4, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.DATA.BATCH_SIZE, shuffle=False,
                          num_workers=4, persistent_workers=True, pin_memory=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['all', 'recon', 'RGB_recon'], required=True)
    for name in ('upstream', 'cache', 'features', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    require_compute()
    sys.path.insert(0, str(args.upstream))
    from config.defaults import get_cfg_defaults
    from module.uem_module import UEM_Module
    from module.ema import EMA
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
    from pytorch_lightning.loggers import TensorBoardLogger

    ready = json.loads((args.features/'READY.json').read_text())
    if ready['cache_index_sha256'] != sha(args.cache/'index.json'):
        raise RuntimeError('Visual cache and motion cache identities differ')
    pl.seed_everything(62, workers=True)
    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(args.upstream/'config/nymeria_soma.yaml'))
    cfg.DATA.DATA_DIR = str(args.cache)
    cfg.DATA.COND_IMG_FEAT = True
    cfg.DATA.COND_TRAJ = args.variant != 'RGB_recon'
    cfg.DATA.IMG_FEAT_TYPE = 'dinov2'
    cfg.DATA.VISUAL_CACHE = str(args.features)
    cfg.DATA.VISUAL_CACHE_FINGERPRINT = ready['fingerprint']
    cfg.DATA.UEM_VARIANT = args.variant
    cfg.MODEL.FINETUNE_TYPE = None if args.variant == 'all' else 'recon'
    cfg.TRAIN.EXP_PATH = str(args.output)
    cfg.TRAIN.NUM_GPUS = 1
    cfg.TRAIN.USE_CKPT_LR = True
    contract = dict(alias='uniegomotion-'+args.variant, variant=args.variant,
                    cache=str(args.cache), motion_cache_sha256=ready['cache_index_sha256'],
                    normalization_sha256=sha(args.cache/'train_stats.npz'),
                    visual_cache=ready, frames=97, fps=20, batch_size=64, seed=62,
                    epochs=350, learning_rate=3e-5, weight_decay=.01,
                    native_architecture=True, native_loss_optimizer_diffusion_ema=True,
                    sampler_steps=1000, no_text=True, no_gt_shape_conditioning=True,
                    task_weights=({'recon': 1/3, 'fore': 1/3, 'gen': 1/3} if args.variant=='all' else {'recon':1.}),
                    masking='Explicit native masks override stochastic upstream masks. Recon observes all inputs; fore observes first24 RGB/camera frames; gen observes only first RGB, no camera. RGB_recon omits camera key entirely.',
                    target='Unchanged body230 (220 body motion +10 predicted SOMA shape PCA)',
                    official_difference='Equal explicit3-task sampling replaces upstream independent stochastic masking, which does not explicitly sample quarter-prefix forecasting.',
                    validation='Deterministic idx%3 tasks for all; reconstruction only for recon variants',
                    initialization='Fresh random initialization; future resumes use only this run last.ckpt')
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'experiment.json'
    if path.exists() and json.loads(path.read_text()) != contract:
        raise RuntimeError('Refusing to resume a different ablation recipe')
    atomic_json(path, contract)
    resume = args.output/'last.ckpt'
    cfg.MODEL.CKPT_PATH = str(resume) if resume.exists() else None
    cfg.freeze()
    data = VisualDataModule(cfg)
    model = UEM_Module(cfg)
    logger = TensorBoardLogger(str(args.output), name='', version='', default_hp_metric=False, flush_secs=60)
    callbacks = [ModelCheckpoint(dirpath=str(args.output), every_n_epochs=10,
                                  save_top_k=-1, save_last=True),
                 LearningRateMonitor('step'), TQDMProgressBar(refresh_rate=50), EMA(.999)]
    trainer = pl.Trainer(default_root_dir=str(args.output), logger=logger, devices=1,
                         accelerator='gpu', num_sanity_val_steps=2, log_every_n_steps=50,
                         callbacks=callbacks, max_epochs=350, check_val_every_n_epoch=1,
                         gradient_clip_algorithm='norm', gradient_clip_val=1.)
    print(json.dumps(dict(stage='training_start', alias=contract['alias'], resume=cfg.MODEL.CKPT_PATH)), flush=True)
    trainer.fit(model, datamodule=data, ckpt_path=cfg.MODEL.CKPT_PATH)
    atomic_json(args.output/'COMPLETE.json', dict(epochs=350, global_step=int(trainer.global_step)))


if __name__ == '__main__':
    main()
