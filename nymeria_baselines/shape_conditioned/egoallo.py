"""DDP training of the native EgoAllo motion objective with supplied shape."""
from pathlib import Path
import argparse
import dataclasses
import json
import os
import shutil
import sys
import time

from .common import require_compute, contract, save_contract, atomic_json


def activate(upstream):
    sys.path.insert(0, str(upstream/'egoallo/src'))


def main():
    p = argparse.ArgumentParser()
    for name in ('upstream', 'cache', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--max-steps', type=int, default=200000)
    p.add_argument('--save-every', type=int, default=5000)
    p.add_argument('--stop-after', type=int, default=0,
                   help='Acceptance-only early stop; does not change the experiment recipe')
    a = p.parse_args()
    require_compute()
    activate(a.upstream)
    import torch
    from accelerate import Accelerator, DataLoaderConfiguration
    from accelerate.utils import set_seed
    from egoallo import network, training_loss
    from egoallo.data.nymeria_soma import NymeriaSomaDataset
    from egoallo.data.dataclass import collate_dataclass
    from torch.utils.data import DataLoader
    from tensorboardX import SummaryWriter
    accelerator = Accelerator(dataloader_config=DataLoaderConfiguration(
        split_batches=True, use_seedable_sampler=True, data_seed=62))
    if accelerator.num_processes != 2:
        raise RuntimeError('Training requires exactly two torchrun ranks')
    set_seed(62)
    cfg = network.EgoDenoiserConfig(include_hands=False, body_joint_count=22,
                                   contact_joint_count=4, include_hand_positions_cond=False)
    value = contract(a.cache, a.upstream, 'EgoAllo trajectory + supplied shape', 256,
                     f'{a.max_steps} optimizer updates')
    value.update(lr=1e-4, weight_decay=1e-4, warmup_steps=1000, seed=62,
                 state_dim=202, shape_dim=16, model_config=dataclasses.asdict(cfg),
                 checkpoint_every=a.save_every, shape_conditioning='MLP into camera encoder stream',
                 resume='Accelerate model/optimizer/scheduler/RNG plus seeded sampler epoch and cursor')
    if accelerator.is_main_process:
        save_contract(a.output, value)
    accelerator.wait_for_everyone()
    model = network.EgoDenoiser(cfg)
    data = NymeriaSomaDataset(a.cache, split='train')
    loader = DataLoader(data, batch_size=256, shuffle=True, num_workers=4,
                        persistent_workers=True, pin_memory=True, drop_last=True,
                        collate_fn=collate_dataclass)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda step:min(1.,step/1000))
    model, loader, optim, scheduler = accelerator.prepare(model, loader, optim, scheduler)
    helper = training_loss.TrainingLossComputer(training_loss.TrainingLossConfig(), accelerator.device)
    writer = SummaryWriter(str(a.output/'tb')) if accelerator.is_main_process else None
    step, epoch, cursor = 0, 0, 0
    resume_file = a.output/'latest.json'
    if resume_file.exists():
        state = json.loads(resume_file.read_text())
        accelerator.load_state(state['checkpoint'])
        step, epoch, cursor = state['step'], state['epoch'], state['next_batch']
    else:
        # Common initialization, distinct diffusion RNG for the two ranks.
        set_seed(62, device_specific=True)
    last_clock, last_step = time.monotonic(), step

    def save(next_batch):
        destination = a.output/f'checkpoints_{step:06d}'
        accelerator.save_state(str(destination))  # collective on both ranks
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            atomic_json(destination/'COMPLETE.json', {'step':step,'epoch':epoch,'next_batch':next_batch})
            atomic_json(resume_file, {'checkpoint':str(destination),'step':step,
                                      'epoch':epoch,'next_batch':next_batch})
            # Only this new run's checkpoints; retain 100k multiples and newest.
            for old in a.output.glob('checkpoints_*'):
                n = int(old.name.split('_')[-1])
                if old != destination and n % 100000 and (old/'COMPLETE.json').exists():
                    shutil.rmtree(old)
        accelerator.wait_for_everyone()

    stop = min(a.max_steps,a.stop_after) if a.stop_after else a.max_steps
    while step < stop:
        loader.set_epoch(epoch)
        active = accelerator.skip_first_batches(loader, cursor) if cursor else loader
        for batch_index, batch in enumerate(active, start=cursor):
            model.train()
            loss, values = helper.compute_denoising_loss(
                model, unwrapped_model=accelerator.unwrap_model(model), train_batch=batch)
            finite = accelerator.reduce(torch.isfinite(loss).int(), reduction='sum')
            if finite.item() != accelerator.num_processes:
                raise FloatingPointError('Nonfinite loss on at least one rank')
            accelerator.backward(loss)
            norm = accelerator.clip_grad_norm_(model.parameters(), 1.)
            finite = accelerator.reduce(torch.isfinite(norm).int(), reduction='sum')
            if finite.item() != accelerator.num_processes:
                raise FloatingPointError('Nonfinite gradient on at least one rank')
            optim.step()
            scheduler.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0 or step <= 3:
                measured = accelerator.reduce(loss.detach(), reduction='mean').item()
                if accelerator.is_main_process:
                    now = time.monotonic()
                    status = {'step':step,'epoch':epoch,'loss':measured,
                              'lr':scheduler.get_last_lr()[0],
                              'seconds_per_update':(now-last_clock)/max(1,step-last_step),
                              'peak_memory_gib':torch.cuda.max_memory_allocated()/2**30}
                    print(json.dumps(status), flush=True)
                    atomic_json(a.output/'progress.json', status)
                    for key, val in values.items():
                        writer.add_scalar(key, val, step)
                    writer.add_scalar('learning_rate',status['lr'],step)
                    last_clock, last_step = now, step
            if step % a.save_every == 0 or step == stop:
                save(batch_index+1)
            if step >= stop:
                break
        epoch += 1
        cursor = 0
    if accelerator.is_main_process:
        writer.close()
        atomic_json(a.output/('COMPLETE.json' if step>=a.max_steps else 'STOPPED.json'),
                    {'step':step,'epoch':epoch})


if __name__ == '__main__':
    main()
