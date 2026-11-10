"""Guarded overlapping replacement launch; both GPU pairs must be coordinated."""
import argparse
import os
from pathlib import Path
import shutil
import sys
from .policy import configure, SCOPES

def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--camera-cache',type=Path,required=True)
    parser.add_argument('--attention-scope',choices=SCOPES,required=True)
    parser.add_argument('--conditioning-attention',choices=('directional','bidirectional'),default='directional')
    options,rest=parser.parse_known_args()
    if not options.camera_cache.resolve().is_relative_to('/mnt/local') or not (options.camera_cache/'index.json').is_file():
        raise ValueError('Need the complete existing node-local camera cache')
    os.environ['PHASE3_CAMERA_PREDICTION_CACHE']=str(options.camera_cache)
    os.environ['PHASE3_CAMERA_ATTENTION_SCOPE']=options.attention_scope
    configure(options.attention_scope, options.conditioning_attention)
    from .validation import install
    install()
    names={x.split('=',1)[0] for x in rest}
    for key,value in [('conditioning-attention',options.conditioning_attention),('bridge-parameterization','mmdit'),
                      ('bridge-attention-backend','grouped'),('bridge-precision','bf16_bridge'),
                      ('ema-decay','0.999'),('validation-weights','ema'),('validation-every','10000'),
                      ('validation-batch-size','32'),('validation-decode-batch-size','4')]:
        if '--'+key not in names:rest+=['--'+key,value]
    if '--validation-config' not in names:raise ValueError('Supply pinned fixed71 validation assets')
    import motion_expert_phase3_edge_image_joint.launch as original
    original_freeze=original.freeze_sources
    def freeze(repo,snapshot):
        manifest=original_freeze(repo,snapshot)
        from motion_expert_phase3_edge_joint.settings import sha256
        for name in ['motion_expert_phase3_edge_image_joint_camera','motion_expert_phase3_edge_camera_directional_nojoint']:
            for source in sorted((repo/name).glob('*.py')):
                target=snapshot/source.relative_to(repo);target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source,target);manifest[str(source.relative_to(repo))]=sha256(source)
        return manifest
    original.freeze_sources=freeze
    sys.argv=[sys.argv[0],*rest]
    original.main(experiment_package='motion_expert_phase3_edge_camera_directional_nojoint')

if __name__=='__main__':main()
