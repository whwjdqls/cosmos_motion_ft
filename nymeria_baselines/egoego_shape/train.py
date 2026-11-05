"""Train on two allocated GPUs with the current supplied-shape UEM schedule."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.plugins.environments import TorchElasticEnvironment

from .data import CACHE, MotionDataset, require_compute
from .model import CameraShapeDiffusion
from .upstream import identity, sha

UEM = Path('/scratch2/whwjdqls99/cosmos/operations/shape_conditioned_baselines_20261101/upstream/UniEgoMotion')


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def contract(stats):
    return {'version':'egoego_camera_shape_body23_v1','upstream':identity(),
        'source_sha256':{p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        'cache_index_sha256':sha(CACHE/'index.json'),'stats_sha256':sha(stats),
        'data_preparation':json.loads(Path(stats).with_name('READY.json').read_text()),
        'target':'97x(23x3_global_positions_then_23x6_global_row6D_rotations)',
        'coordinates':'Yup_floor_aligned_initial_camera_horizontal_position_and_yaw_removed',
        'conditioning':'measured_camera_pose97x9_as_observed_head_input; clean_centered_SOMA30_shape_token',
        'shape_prediction':False,'camera_prediction':False,'RGB_input':False,
        'anatomical_head_target_replaced_by_camera':False,
        'model':{'d_model':512,'n_dec_layers':4,'n_head':4,'d_k':256,'d_v':256,'motion_dim':207,'shape_dim':90},
        'recipe':{'global_batch':64,'world_size':2,'per_gpu_batch':32,'accumulation':1,
          'epochs':350,'steps_per_epoch':1588,'target_optimizer_steps':555800,
          'optimizer':'AdamW','lr':3e-5,'betas':[0.9,0.999],'weight_decay':.01,
          'scheduler':'StepLR_epoch','lr_drop_epoch':300,'lr_drop_gamma':.1,
          'loss':'uniform_timestep_clean_motion_MSE','diffusion_steps':1000,
          'noise_schedule':'native_EgoEgo_cosine','objective':'pred_x0','sample_clip_denoised':False,
          'precision':'float32','seed':62,'gradient_clip_norm':1.,'ema':.999,
          'validation':'every_epoch_full_validation_cohort','save_last':'every_epoch','archive_every_epochs':10},
        'uem_reference':{'config_sha256':sha(UEM/'config/nymeria_soma.yaml'),
                         'optimizer_source_sha256':sha(UEM/'module/uem_module.py'),
                         'ema_source_sha256':sha(UEM/'module/ema.py')}}


def ema_callback():
    spec=importlib.util.spec_from_file_location('_egoego_uem_ema',UEM/'module/ema.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.EMA(.999)


class Model(pl.LightningModule):
    def __init__(self,value):
        super().__init__();self.save_hyperparameters({'experiment':value});self.net=CameraShapeDiffusion()

    def forward(self,motion,camera,shape):
        return self.net(motion,camera,shape)

    def training_step(self,batch,batch_idx):
        loss=self(batch['motion'],batch['camera'],batch['shape'])
        if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
        self.log('train/loss',loss,on_step=True,on_epoch=True,sync_dist=True,batch_size=len(batch['motion']))
        return loss

    def validation_step(self,batch,batch_idx):
        loss=self(batch['motion'],batch['camera'],batch['shape'])
        if not torch.isfinite(loss):raise RuntimeError('Non-finite validation loss')
        self.log('val/loss',loss,on_epoch=True,sync_dist=True,batch_size=len(batch['motion']))

    def on_before_optimizer_step(self,optimizer):
        grads=[p.grad for p in self.parameters() if p.grad is not None]
        norm=torch.stack([g.detach().float().norm() for g in grads]).norm()
        if not torch.isfinite(norm):raise RuntimeError('Non-finite gradients')
        self.log('train/grad_norm',norm,on_step=True,sync_dist=True)

    def configure_optimizers(self):
        optimizer=torch.optim.AdamW(self.parameters(),lr=3e-5,weight_decay=.01)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=300,gamma=.1)
        return [optimizer],[{'scheduler':scheduler,'interval':'epoch'}]


class Progress(Callback):
    def __init__(self,out):self.out=out;self.before=None;self.started=None;self.initial=None
    def on_train_start(self,trainer,model):
        self.started=time.monotonic();self.initial=trainer.global_step
        self.before={k:p.detach().cpu().clone() for k,p in model.net.denoise_fn.shape_encoder.named_parameters()}
    def on_train_batch_start(self,trainer,model,batch,batch_idx):self.batch_start=time.monotonic()
    def on_train_batch_end(self,trainer,model,outputs,batch,batch_idx):
        step=trainer.global_step
        if trainer.is_global_zero and (step==self.initial+1 or step%10==0):
            changed=any(not torch.equal(self.before[k],p.detach().cpu()) for k,p in model.net.denoise_fn.shape_encoder.named_parameters())
            assert changed,'Shape encoder parameters did not update'
            loss=outputs['loss'].detach().item()
            optimizer=trainer.optimizers[0]
            value={'stage':'training','step':step,'target_steps':555800,'epoch':trainer.current_epoch,
                   'loss':loss,'lr':optimizer.param_groups[0]['lr'],'seconds_per_update':time.monotonic()-self.batch_start,
                   'overall_seconds_per_update':(time.monotonic()-self.started)/(step-self.initial),
                   'peak_memory_gib':torch.cuda.max_memory_allocated()/2**30,
                   'shape_parameters_changed':changed,'time':time.time(),'job':os.environ['SLURM_JOB_ID']}
            save_json(self.out/'progress.json',value);print('EGOEGO_PROGRESS '+json.dumps(value),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--stats',type=Path,required=True)
    p.add_argument('--resume',default='auto');p.add_argument('--stop-after',type=int,default=-1)
    p.add_argument('--smoke',action='store_true');p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();require_compute()
    assert int(os.environ.get('WORLD_SIZE','1'))==2,'Two torchrun ranks required'
    pl.seed_everything(62,workers=True);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    value=contract(a.stats);a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'experiment.json').exists():assert json.loads((a.out/'experiment.json').read_text())==value,'Changed experiment/source'
    if int(os.environ['RANK'])==0:save_json(a.out/'experiment.json',value)
    training=MotionDataset(a.stats,'train');validation=MotionDataset(a.stats,'val')
    assert len(training)==101576 and len(validation)==10927
    opts=dict(batch_size=32,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0)
    train_loader=DataLoader(training,shuffle=True,**opts);val_loader=DataLoader(validation,shuffle=False,**opts)
    model=Model(value)
    last=a.out/'last.ckpt'
    resume=str(last) if a.resume=='auto' and last.exists() else (None if a.resume in ('auto','none') else a.resume)
    trainer=pl.Trainer(accelerator='gpu',devices=2,strategy=DDPStrategy(find_unused_parameters=False),
        plugins=[TorchElasticEnvironment()],precision='32-true',max_epochs=350,
        max_steps=a.stop_after,limit_val_batches=2 if a.smoke else 1.0,
        num_sanity_val_steps=2,log_every_n_steps=10,enable_progress_bar=False,
        gradient_clip_val=1.,gradient_clip_algorithm='norm',default_root_dir=str(a.out),
        logger=TensorBoardLogger(str(a.out),name='',version='',default_hp_metric=False),
        callbacks=[ModelCheckpoint(dirpath=str(a.out),every_n_epochs=10,save_top_k=-1,save_last=True),
                   LearningRateMonitor('step'),ema_callback(),Progress(a.out)])
    trainer.fit(model,train_loader,val_loader,ckpt_path=resume)
    trainer.save_checkpoint(last)
    if trainer.is_global_zero:
        save_json(a.out/('SMOKE_COMPLETE.json' if a.smoke else 'COMPLETE.json'),
            {'step':trainer.global_step,'epochs':trainer.current_epoch,'smoke':a.smoke,'time':time.time()})


if __name__=='__main__':main()
