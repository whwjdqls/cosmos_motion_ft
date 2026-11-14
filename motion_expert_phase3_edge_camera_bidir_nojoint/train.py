"""Reuse the verified camera trainer, with explicit bidirectional policy/data."""
import argparse
import inspect
from pathlib import Path
import sys
import torch
from motion_expert_phase3_edge_camera_directional_nojoint import train as original
from motion_expert_phase3_edge_joint.settings import RECIPE,sha256
from .policy import configure
from .data import Phase3Dataset

def prepare_trainer(probe_updates=0):
    namespace=original.__dict__.copy()
    namespace.update(configure=configure,Phase3Dataset=Phase3Dataset)
    exec(compile(inspect.getsource(original.prepare_trainer),__file__+':native_camera_adapter','exec'),namespace)
    train=namespace['prepare_trainer'](probe_updates)
    original_identity=train.source_identity
    def source_identity(args,dataset,model):
        contract=original_identity(args,dataset,model)
        contract['camera_bidir_nojoint_implementation']={p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
        return contract
    train.source_identity=source_identity
    return train

def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--probe-updates',type=int,default=0)
    args,rest=parser.parse_known_args()
    if args.probe_updates<0:raise ValueError('Invalid probe count')
    configure()
    if '--conditioning-attention' not in rest:rest+=['--conditioning-attention','bidirectional']
    elif rest[rest.index('--conditioning-attention')+1]!='bidirectional':raise ValueError('This experiment requires bidirectional attention')
    if '--clean-motion-timestep' not in rest:rest+=['--clean-motion-timestep','zero']
    train=prepare_trainer(args.probe_updates)
    sys.argv=[sys.argv[0],*rest]
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from scripts.phase3_performance import execution_policy
    from motion_expert_phase3_edge_camera_directional_nojoint.attention import attention_scope
    with attention_scope(RECIPE['bridge_attention_scope']),execution_policy(cache_bridge_masks=True,checkpoint_motion=True):
        train.main()

if __name__=='__main__':main()
