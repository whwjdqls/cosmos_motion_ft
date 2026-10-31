"""Standard-library provenance and the exact expected caption-dropout mixture."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from ablation_settings import conditioning_partition


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',required=True)
    parser.add_argument('--task-weights',type=json.loads,required=True)
    parser.add_argument('--bones-frac',type=float,required=True)
    parser.add_argument('--ti-dropout',type=float,required=True)
    parser.add_argument('--batch',type=int,required=True)
    parser.add_argument('--accum',type=int,required=True)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    patterns=('motion_expert_t2m_edge/*.py','motion_expert_t2m_edge/*.sh',
              'motion_expert_joint_attention/*.py','motion_expert_joint_attention/*.sh',
              'motion_expert/*.py','migration/custom*.py','migration/custom*.sh')
    paths=sorted({p for pattern in patterns for p in root.glob(pattern)})
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    tracked=json.loads(json.dumps(hashes,sort_keys=True))
    result=dict(variant=args.variant,task_weights=args.task_weights,bones_within_t2m=args.bones_frac,
        cfg_dropout=.1,ti2m_cfg_dropout=args.ti_dropout,
        effective_bones_dropout=.19,caption_null_is_shape_and_length_conditioned=True,
        expected_partition=conditioning_partition(ti2m_fraction=args.task_weights['textimg2motion'],
            bones_within_t2m=args.bones_frac,ti2m_caption_dropout=args.ti_dropout),
        microbatch=args.batch,grad_accum=args.accum,global_batch=args.batch*args.accum,
        objective=os.environ['EDGE_PHASE2_OBJECTIVE'],codec=os.environ['EDGE_PHASE2_FEATURE_CODEC'],
        source_checkout=str(root),source_sha256=hashes,
        source_fingerprint=hashlib.sha256(json.dumps(tracked,sort_keys=True).encode()).hexdigest(),
        git_revision=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        working_tree=subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).splitlines(),
        environment={key:os.environ.get(key) for key in ('COSMOS_STORAGE_ROOT','WEKA_ROOT','RUN_ROOT',
            'COSMOS_ENV_ROOT','COSMOS_FRAMEWORK_EDGE_ROOT','KIMODO_ROOT','PHASE2_KIMODO_STATS_DIR')})
    if args.out:
        # Resume must use the same implementation; do not overwrite provenance.
        path=args.out/'ablation_launch.json'
        if path.exists():
            old=json.loads(path.read_text())
            keys=('variant','task_weights','bones_within_t2m','cfg_dropout','ti2m_cfg_dropout',
                  'microbatch','grad_accum','global_batch','objective','codec','source_fingerprint')
            if any(old.get(key)!=result[key] for key in keys):
                raise RuntimeError('Launch/source drift; review before resuming an experiment')
        else:
            path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ('source_sha256','working_tree')},indent=2,sort_keys=True))


if __name__=='__main__':
    main()
