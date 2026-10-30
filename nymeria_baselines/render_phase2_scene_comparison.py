"""Predetermined paired GT/text/image/both point-cloud comparisons."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import textwrap

import numpy as np
from PIL import Image,ImageDraw

from nymeria_baselines.phase2_collision_scene import SceneScorer
from nymeria_baselines.evaluate_gt_scene_collision import (
    AXES, decode_uniego, ground_features, canonicalize_frame0, render, score, save, sha,
)
from nymeria_baselines.generated_video_motion_report import publish_video


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operation',required=True,type=Path)
    parser.add_argument('--pipeline-operation',type=Path)
    parser.add_argument('--review-indices',nargs='+',type=int,default=[0,1,3])
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or 'login' in socket.gethostname():raise RuntimeError('Compute allocation required')
    from nymeria_world.nymeria_camera_dataset import decode_window_pyav
    from motion_expert_joint_attention.render_motion import write_mp4
    import av
    cfg=json.loads((args.operation/'configuration.json').read_text())
    rows=json.loads((args.operation/'cases.json').read_text())
    # Original fixed71 review cases0,1,3, selected independently of new predictions.
    reference=Path(cfg['audited_gt_operation'])
    prior=json.loads((reference/'per_case.json').read_text())
    selected=[]
    for i in args.review_indices:
        target=prior[i]
        selected.append(next(r for r in rows if r['uuid']==target['uuid'] and r['start_frame']==target['start']))
    out=(args.pipeline_operation or args.operation)/'scene_comparisons';out.mkdir(exist_ok=True)
    save(out/'configuration.json',{'selected_case_ids':[r['case_id'] for r in selected],
        'selection':'fixed prior71review cases, before looking at generated outcomes',
        'review_indices':args.review_indices,'pipeline_operation':str(args.pipeline_operation) if args.pipeline_operation else None,
        'point_count_cap':None,'common_camera_for_all_conditions':True,
        'collision_body':'exclude_thigh_shin_foot','source_sha256':sha(__file__)})
    scorer=SceneScorer(args.operation)
    calibration=json.loads((Path(__file__).parents[1]/'motion_expert_joint_attention/head_camera_calibration_train.json').read_text())
    rh=np.asarray(calibration['rotation_head_to_upright_camera'])
    media=json.loads((out/'media.json').read_text()) if (out/'media.json').exists() else []
    for row in selected:
        target=out/(row['case_id']+'.mp4')
        if target.with_suffix('.publication.json').exists():continue
        batch=row['index']//4*4
        with np.load(args.operation/'samples'/f'batch_{batch:07d}.npz') as z:
            offset=row['index']-batch;assert z['case_id'][offset]==row['case_id']
            pred=z['raw_features'][:,offset].copy()
        raw=np.load(Path(cfg['reference_request']['data_dir'])/row['features_file']).astype(np.float32)
        gt=canonicalize_frame0(ground_features(raw,row['floor_offset']))
        features=[gt,pred[1],pred[0],pred[2]]
        labels=['GT','Text only','Image only','Image + text']
        colors=[(.1,.4,.7,1),(.85,.45,.08,1),(.1,.55,.28,1),(.55,.3,.7,1)]
        if args.pipeline_operation:
            pipeline_cfg=json.loads((args.pipeline_operation/'configuration.json').read_text())
            assert Path(pipeline_cfg['paired_baseline_operation']).resolve()==args.operation.resolve()
            assert pipeline_cfg['cases_sha256']==cfg['cases_sha256']
            with np.load(args.pipeline_operation/'case_outputs'/row['case_id']/'motion.npz') as z:
                features.append(z['raw_features'].copy())
            labels.append('Generated video -> motion');colors.append((.75,.24,.18,1))
        motions=[decode_uniego(a) for a in features]
        context=scorer.recording(row['uuid']);start=int(row['start_frame'])
        rotation=context['rotations'][start];origin=context['origins'][start];off=np.array([0,row['floor_offset'],0])
        points=(context['points']-origin-off)@rotation
        keys=np.floor(points/.02).astype(np.int64)
        _,inverse,counts=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
        voxels=np.stack([np.bincount(inverse,weights=points[:,i])/counts for i in range(3)],axis=1).astype(np.float32)
        framing=np.concatenate([m.world_positions for m in motions])
        rgb=decode_window_pyav(row['image_path'],row['image_start'],97,20.)
        bundle=out/(row['case_id']+'_RGB.npz');np.savez_compressed(bundle,rgb=rgb)
        camera=(context['camera_positions'][start:start+97]-origin-off)@rotation
        camera_file=Path(scorer.gt_cfg['data_root'])/'camera_rgb'/(row['uuid']+'.npz')
        with np.load(camera_file) as z:cam_rot=rotation.T@AXES@z['cam_world_rot_upright'][start:start+97]
        panels=[]
        for j,(motion,label,color) in enumerate(zip(motions,labels,colors)):
            metrics,ids,hits,_=score(voxels,motion.world_positions,context['radii'])
            arrays={'points':voxels,'display_points':points.astype(np.float32),'joints':motion.world_positions,
                'point_ids':ids,'hits':hits,'radii':context['radii'],'head_rotation0':motion.world_rotations[0,6],
                'camera_positions':camera,'camera_forward':cam_rot[:,:,2],
                'head_camera_forward':(motion.world_rotations[:,6]@rh)[:,:,2],
                'torso_forward':motion.world_rotations[:,0,:,2],'excluded_capsule_names':metrics['excluded_capsule_names'],
                'framing_joints':framing,'body_color':color}
            case={'id':row['case_id']+'_'+str(j),'bundle':str(bundle)}
            options=argparse.Namespace(out=out,renderer='opengl')
            item=render(options,case,arrays)
            with av.open(item['mp4']) as container:
                panels.append(np.stack([f.to_ndarray(format='rgb24')[30:500,800:1600] for f in container.decode(video=0)]))
        observed=np.load(row['image_file']).transpose(1,2,0)
        output=[]
        canvas_height=250+((len(panels)+1)//2)*495
        canvas_height+=canvas_height%2  # H.264/yuv420p requires even dimensions.
        for frame in range(97):
            canvas=Image.new('RGB',(1600,canvas_height),'white');draw=ImageDraw.Draw(canvas)
            canvas.paste(Image.fromarray(observed).resize((300,225)),(10,10))
            draw.rectangle((8,8,312,237),outline='#18754b',width=5)
            draw.multiline_text((330,15),'Observed first RGB (green): image-conditioned methods\nCaption: text-conditioned methods\n'+
                '\n'.join(textwrap.wrap(row['text'],150))+'\nSame initial noise; no observed motion or camera input. Red points: capsule overlap.',fill='#222222',spacing=7)
            for i,(clip,label) in enumerate(zip(panels,labels)):
                x=(i%2)*800;y=250+(i//2)*495
                canvas.paste(Image.fromarray(clip[frame]),(x,y+25))
                draw.rectangle((x,y,x+799,y+24),fill=tuple(int(v*255) for v in colors[i][:3]))
                draw.text((x+10,y+7),label+' | thighs/shins/feet ignored',fill='white')
            output.append(np.asarray(canvas))
        write_mp4(np.stack(output),str(target),fps=20)
        Image.fromarray(output[0]).save(target.with_suffix('.png'))
        publish_video(args.pipeline_operation or args.operation,target,'Paired scene collision | '+', '.join(labels)+' | '+row['case_id'])
        media.append({'case_id':row['case_id'],'mp4':str(target),'points':len(points),'same_camera_all_conditions':True})
        save(out/'media.json',media)
        print(json.dumps(media[-1]),flush=True)
    save(out/'COMPLETE.json',{'cases':len(selected),'remote_media_hashes_verified':True})


if __name__=='__main__':main()
