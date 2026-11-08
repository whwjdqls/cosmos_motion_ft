"""Prepare truthful standard-SOMA mesh/error data from explicit saved inputs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import numpy as np
from .mesh import BODY_IN30,BODY_NAMES,load_skin,mesh_motion,joint_error_surface,sha
from .colors import LOW,HIGH


def require_compute():
    host=os.uname().nodename
    if 'login' in host or host in {'cs-gpu-01','node37','node29'}:
        raise RuntimeError('Mesh/data processing must use an allowed compute node')
    if os.environ.get('COSMOS_SITE','yonsei')=='yonsei' and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use an owned Slurm allocation on Yonsei')


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def load_prediction(spec):
    """No shape guessing, hidden stats, pose fitting or grounding."""
    path=Path(spec['path'])
    if spec.get('sha256'):assert sha(path)==spec['sha256']
    provenance=dict(path=str(path),sha256=sha(path),format=spec['format'],specification=spec)
    if spec['format']=='joints':
        with np.load(path,allow_pickle=False) as z:
            p=z[spec.get('positions_key','joints')].copy()
            r=z[spec.get('rotations_key','rotations')].copy()
        if spec.get('position_units','m')!='m':raise ValueError('Convert input positions explicitly to meters')
        if spec.get('rotation_convention','global_matrices')!='global_matrices':
            raise ValueError('Only declared global rotation matrices accepted')
    elif spec['format']=='uniego283':
        from nymeria_baselines.visualize_official_baselines import decode
        if path.suffix=='.npy':raw=np.load(path,allow_pickle=False)
        else:
            with np.load(path,allow_pickle=False) as z:raw=z[spec['features_key']].copy()
        if raw.ndim==3 and len(raw)==1:raw=raw[0]
        assert raw.ndim==2 and raw.shape[1]==283
        if 'normalized' not in spec:raise ValueError('Declare normalized true/false explicitly')
        if spec['normalized']:
            mean,std=np.load(spec['mean'],allow_pickle=False),np.load(spec['std'],allow_pickle=False)
            assert mean.shape==std.shape==(283,) and np.all(std>0)
            for name in ('mean','std'):
                digest=sha(spec[name]);provenance[name+'_sha256']=digest
                if spec.get(name+'_sha256'):assert digest==spec[name+'_sha256']
            raw=raw*std+mean
        p,r=decode(raw,30)
        provenance['decoder']='nymeria_baselines.visualize_official_baselines.decode, existing evaluation path'
    else:raise ValueError('Unsupported input format')
    if p.shape[1:]==(30,3):p=p[:,BODY_IN30];r=r[:,BODY_IN30]
    assert p.shape[1:]==(23,3) and r.shape==(*p.shape[:2],3,3)
    assert np.isfinite(p).all() and np.isfinite(r).all()
    # Position errors are the decoded raw benchmark coordinates. No centering,
    # alignment or shape retargeting is performed by this loader.
    return p.astype(np.float64),r.astype(np.float64),provenance


def prepare(config_path):
    require_compute();config_path=Path(config_path)
    cfg=json.loads(config_path.read_text());out=Path(cfg['out']);out.mkdir(parents=True,exist_ok=True)
    skin=load_skin(cfg['skin'])
    assert len(skin.vertices)==18056 and len(skin.faces)==36108
    asset=out/'assets';asset.mkdir(exist_ok=True)
    np.savez_compressed(asset/'skin.npz',bind_vertices=skin.vertices.astype(np.float32),faces=skin.faces,
        vertex_joint_weights=skin.vertex_joint_weights.astype(np.float32),
        body_joint_names=np.asarray(BODY_NAMES))
    source=asset/'source';source.mkdir(exist_ok=True)
    for path in Path(__file__).parent.glob('*.py'):shutil.copy2(path,source/path.name)
    fingerprints={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    reference=cfg.get('error_reference','root_relative')
    assert reference in ('root_relative','world')
    manifest=dict(schema='soma_joint_error_mesh_v2',fps=cfg.get('fps',20),units='meters',
        source_config_sha256=sha(config_path),source_config=cfg,
        skin=dict(path=skin.source,sha256=skin.source_sha256,vertices=len(skin.vertices),faces=len(skin.faces),
            identity='standard SOMA visualization template, not subject-specific SOMA-X'),
        pose_policy='All23 body global rotations and translations retained exactly; unmapped77 rig bones follow bind/relaxed-hand local transforms. No FK substitution of measured/predicted body joints.',
        error_policy='23-joint Euclidean error after independent per-frame pelvis subtraction when root_relative; world preserves translation. No rotation/scale alignment. Interpolated by summed SOMA LBS influence; not measured per-vertex GT distance.',
        error_reference=reference,shadow_style=cfg.get('shadow_style','single_back'),
        error_mode=cfg.get('error_mode','joint_lbs'),maximum_cm=cfg.get('maximum_cm',30),
        palette=dict(low=LOW,high=HIGH,space='OKLab interpolation, sRGB output'),
        display_transform='Right-handed Y-up data -> Blender(X,-Z,Y), common rigid display framing only; no X mirror',
        source_sha256=fingerprints,cases=[])
    for case in cfg['cases']:
        root=out/case['id'];root.mkdir(exist_ok=True)
        gt_p,gt_r,gt_source=load_prediction(case['GT'])
        gt_v=mesh_motion(gt_p,gt_r,skin,relaxed_hands=True)
        models=[]
        for spec in [dict(label='GT',title='Ground truth',**case['GT']),*case['models']]:
            label=spec['label'];is_gt=label=='GT'
            if is_gt:p,r,provenance,v=gt_p,gt_r,gt_source,gt_v
            else:
                p,r,provenance=load_prediction(spec)
                assert p.shape==gt_p.shape
                v=mesh_motion(p,r,skin,relaxed_hands=True)
            world_joint,world_surface=joint_error_surface(p,gt_p,skin,reference='world')
            local_joint,local_surface=joint_error_surface(p,gt_p,skin,reference='root_relative')
            joint,surface=(local_joint,local_surface) if reference=='root_relative' else (world_joint,world_surface)
            world_vertex=np.linalg.norm(v.astype(np.float64)-gt_v.astype(np.float64),axis=-1)
            local_vertex=np.linalg.norm((v.astype(np.float64)-p[:,:1])-(gt_v.astype(np.float64)-gt_p[:,:1]),axis=-1)
            actual_surface=local_vertex if reference=='root_relative' else world_vertex
            error=surface if manifest['error_mode']=='joint_lbs' else actual_surface
            assert manifest['error_mode'] in ('joint_lbs','template_vertex_distance')
            dest=root/(label+'.npz')
            np.savez_compressed(dest,vertices=v,joint_positions=p,joint_rotations=r,
                joint_errors_m=joint.astype(np.float32),vertex_errors_m=error.astype(np.float32),
                joint_lbs_errors_m=surface.astype(np.float32),template_vertex_distances_m=actual_surface.astype(np.float32),
                joint_errors_world_m=world_joint.astype(np.float32),joint_errors_root_relative_m=local_joint.astype(np.float32),
                joint_lbs_errors_world_m=world_surface.astype(np.float32),joint_lbs_errors_root_relative_m=local_surface.astype(np.float32),
                error_reference=np.asarray(reference),
                frame_times_s=np.arange(len(p))/manifest['fps'])
            row=dict(label=label,title=spec.get('title',label),input=spec.get('input',''),frames=len(p),
                mesh=str(dest.relative_to(out)),mesh_sha256=sha(dest),provenance=provenance,
                mpjpe_cm=float(world_joint.mean()*100),root_relative_mpjpe_cm=float(local_joint.mean()*100),
                color_error_mean_cm=float(joint.mean()*100),maximum_joint_error_cm=float(joint.max()*100),
                color_saturated_fraction=float((error*100>=manifest['maximum_cm']).mean()))
            models.append(row)
        all_vertices=np.concatenate([np.load(out/m['mesh'])['vertices'].reshape(-1,3) for m in models])
        minimum=all_vertices.min(0);maximum=all_vertices.max(0)
        view=dict(center_yup=((minimum+maximum)/2).tolist(),extent_yup=(maximum-minimum).tolist(),
            camera_azimuth_deg=cfg.get('camera_azimuth_deg',28),camera_elevation_deg=cfg.get('camera_elevation_deg',12),
            camera_policy='One common fixed camera/bounds for all methods and all frames; no per-method root tracking')
        item=dict(id=case['id'],caption=case.get('caption',''),selection=case.get('selection','explicit user/config case'),
            frames=len(gt_p),fps=manifest['fps'],models=models,view=view)
        write(root/'case.json',item);manifest['cases'].append(item)
        print('PREPARED_CASE',case['id'],[(m['label'],round(m['mpjpe_cm'],3)) for m in models],flush=True)
    write(out/'manifest.json',manifest)
    shutil.copy2(config_path,out/'configuration.json')
    write(out/'PREPARED.json',dict(passed=True,cases=len(manifest['cases']),manifest_sha256=sha(out/'manifest.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    prepare(p.parse_args().config)
