"""Sparse temporal scenes with observed camera poses and synchronized RGB.

Spacing is a common display-only translation per time, never a new trajectory.
Camera glyphs are schematic; lens distortion/FOV is not inferred from raw K.
"""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .pipeline import require_compute,write
from .mesh import sha
from .compose import font,label_center,white_image,error_label
from .camera_geometry import camera_axis_segments


def camera_in_common_frame(poses,calibration):
    poses=np.asarray(poses,dtype=np.float64)
    r_head_in_camera=np.asarray(calibration['head_rotation_in_camera'])
    offset=np.asarray(calibration['head_offset_camera'])
    forward=(poses[0,:3,:3]@r_head_in_camera)[:,2]
    angle=np.arctan2(forward[0],forward[2]);c,s=np.cos(angle),np.sin(angle)
    yaw=np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])
    origin=(poses[0,:3,3]+poses[0,:3,:3]@offset)*[1.,0.,1.]
    common=poses.copy()
    common[:,:3,:3]=yaw.T@poses[:,:3,:3]
    common[:,:3,3]=(poses[:,:3,3]-origin)@yaw
    return common,yaw,origin


def prepare(config_path):
    require_compute();config_path=Path(config_path);cfg=json.loads(config_path.read_text())
    root=Path(cfg['out']);root.mkdir(parents=True,exist_ok=True)
    gallery=Path(cfg['source_gallery']);original=json.loads((gallery/'manifest.json').read_text())
    assert original['error_reference']=='root_relative' and original['shadow_style']=='single_back'
    calibration=json.loads(Path(cfg['calibration']).read_text());assert calibration['train_recordings']==578
    from nymeria_world.nymeria_camera_dataset import decode_window_pyav
    frames=np.asarray(cfg.get('frames',[0,48,96]),dtype=int)
    assert len(frames)==3 and np.all(np.diff(frames)>0) and frames.min()>=0 and frames.max()<97
    az=np.deg2rad(28);right=np.array([np.cos(az),0,-np.sin(az)])
    # A short symmetric optical-CV glyph; this is explicitly not a FOV claim.
    corners=np.array([[-.12,-.09,.20],[.12,-.09,.20],[.12,.09,.20],[-.12,.09,.20]])
    manifest=dict(schema='soma_sparse_camera_v1',fps=20,frames=frames.tolist(),maximum_cm=30,
        palette=original['palette'],error_reference='root_relative',error_mode='joint_lbs',shadow_style='single_back',
        source_gallery=str(gallery),source_manifest_sha256=sha(gallery/'manifest.json'),
        calibration=dict(path=cfg['calibration'],sha256=sha(cfg['calibration']),**calibration),
        camera_policy='Measured upright RGB camera in the same train-calibrated common frame as camera baselines; shared reference, not a predicted camera or a V2M input.',
        camera_glyph=cfg.get('camera_glyph','axes'),camera_axis_length_m=cfg.get('camera_axis_length_m',.18),
        glyph_policy='Camera-local axes: X red/right, Y green/down, Z blue/forward (optical CV). Axis length is for display; no field-of-view claim.',
        layout_policy='Three snapshots in one scene per method. A recorded common translation per timestamp separates snapshots; camera and body move together. Never connect translated cameras into a fictitious trajectory.',
        cases=[])
    shutil.copy2(gallery/'assets/skin.npz',root/'skin.npz')
    rows={r['id']:r for r in cfg['cases']}
    for case in original['cases']:
        row=rows[case['id']];folder=root/case['id'];folder.mkdir(exist_ok=True)
        poses=np.array(np.load(Path(cfg['cache'])/'records'/(row['array']+'.pose_yup.npy'),mmap_mode='r')[row['start']:row['start']+97],dtype=np.float64)
        poses[:,1,3]-=float(row['floor_offset'] or 0)
        common,yaw,origin=camera_in_common_frame(poses,calibration)
        with np.load(row['camera_sidecar'],allow_pickle=False) as a:
            timestamps=a['timestamps_us'][row['start']+frames].copy()
            raw_r=a['cam_world_rot_upright'][row['start']:row['start']+97].astype(np.float64)
            raw_p=a['cam_world_pos_upright'][row['start']:row['start']+97].astype(np.float64)
        basis=np.array([[1.,0,0],[0,0,1],[0,-1,0]])
        np.testing.assert_allclose(basis@raw_r,poses[:,:3,:3],atol=2e-7)
        raw_p=raw_p@basis.T;raw_p[:,1]-=float(row['floor_offset'] or 0)
        np.testing.assert_allclose(raw_p,poses[:,:3,3],atol=2e-6)
        rgb=decode_window_pyav(row['rgb'],row['start'],97,20.)
        rgb_specs=[]
        for frame in frames:
            target=folder/f'rgb_{frame:06d}.png';Image.fromarray(rgb[frame]).save(target)
            rgb_specs.append(dict(frame=int(frame),source_frame=int(row['start']+frame),path=str(target.relative_to(root)),sha256=sha(target),shape=list(rgb[frame].shape)))
        models=[];points=[]
        for model in case['models']:
            with np.load(gallery/model['mesh'],allow_pickle=False) as z:
                v=z['vertices'][frames].copy();p=z['joint_positions'][frames].copy();r=z['joint_rotations'][frames].copy();errors=z['vertex_errors_m'][frames].copy()
            target=folder/(model['label']+'.npz')
            np.savez_compressed(target,vertices=v,joint_positions=p,joint_rotations=r,vertex_errors_m=errors)
            models.append(dict(label=model['label'],title=model['title'],input=model['input'],mesh=str(target.relative_to(root)),sha256=sha(target),source_mesh=model['mesh'],source_mesh_sha256=model['mesh_sha256']))
            points.append(v)
            if model['label']=='GT':gt_head_r=r[:,6]
        expected_head_r=common[frames,:3,:3]@np.asarray(calibration['head_rotation_in_camera'])
        rotation_error=float(np.max(np.abs(expected_head_r-gt_head_r)))
        assert rotation_error<2e-4,('Camera/body rotation frame mismatch',rotation_error)
        glyph=(camera_axis_segments(common[frames],manifest['camera_axis_length_m']).reshape(3,-1,3)
            if manifest['camera_glyph']=='axes' else
            common[frames,None,:3,3]+np.einsum('tij,kj->tki',common[frames,:3,:3],corners))
        all_points=np.concatenate(points+[glyph],axis=1)
        projection=all_points@right
        intervals=np.stack((projection.min(1),projection.max(1)),axis=1)
        shifts=np.zeros(3)
        for i in range(1,3):shifts[i]=intervals[i-1,1]+shifts[i-1]+.4-intervals[i,0]
        offsets=shifts[:,None]*right
        # Extents also include floor and camera origins. Identical for all methods/variants.
        bounds=(all_points+offsets[:,None]).reshape(-1,3)
        minimum=bounds.min(0);maximum=bounds.max(0);minimum[1]=min(0,minimum[1])
        view=dict(center_yup=((minimum+maximum)/2).tolist(),extent_yup=(maximum-minimum).tolist(),camera_azimuth_deg=28,camera_elevation_deg=12)
        dest=folder/'camera_layout.npz'
        np.savez_compressed(dest,camera_to_common=common,display_offsets_yup=offsets,schematic_corners_optical=corners,frames=frames,timestamps_us=timestamps,world_from_common_rotation=yaw,world_common_origin=origin)
        item=dict(id=case['id'],caption=case['caption'],models=models,view=view,rgb=rgb_specs,camera_layout=str(dest.relative_to(root)),camera_layout_sha256=sha(dest),camera_head_rotation_max_abs=rotation_error,source=dict(**row,rgb_stat=dict(bytes=Path(row['rgb']).stat().st_size,mtime_ns=Path(row['rgb']).stat().st_mtime_ns)))
        write(folder/'case.json',item);manifest['cases'].append(item)
        print('PREPARED',case['id'],'camera rotation audit',rotation_error,flush=True)
    write(root/'manifest.json',manifest)
    source=root/'source';source.mkdir(exist_ok=True)
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,source/p.name)


def compose(root):
    require_compute();root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    from .compose import legend_figure
    legend_figure(root,30,error_label(manifest))
    entries=[]
    for case in manifest['cases']:
        variants={}
        for variant in ['clean','cameras','cameras_video']:
            columns=3 if len(case['models'])>4 else 2;rows=(len(case['models'])+columns-1)//columns
            panel_w=1200;w=panel_w*columns;header=80;foot=185;strip=220 if variant=='cameras_video' else 0;panel_h=750+strip
            result=Image.new('RGB',(w,header+rows*panel_h+foot),'white');d=ImageDraw.Draw(result)
            label_center(d,'Sparse motion snapshots · 0.0 / 2.4 / 4.8 seconds',w/2,12,28,True)
            label_center(d,'Spaced for readability; one shared display offset per time for all bodies and cameras.',w/2,50,18,color='#666477')
            cameras=variant!='clean'
            for i,model in enumerate(case['models']):
                x=(i%columns)*panel_w;y=header+(i//columns)*panel_h
                label_center(d,model['title'],x+panel_w/2,y+6,26,True)
                label_center(d,'Reference' if model['label']=='GT' else model['input'],x+panel_w/2,y+40,17,color='#666477')
                receipt=json.loads((root/case['id']/'renders'/model['label']/'COMPLETE.json').read_text())
                assert receipt['passed'] and receipt['frames']==manifest['frames']
                anchors=receipt['frame_anchor_pixels']
                if strip:
                    for j,spec in enumerate(case['rgb']):
                        image=white_image(root/spec['path']);image.thumbnail((170,170))
                        center=x+anchors[j];left=int(center-image.width/2);top=y+75
                        result.paste(image,(left,top));d.rectangle((left-2,top-2,left+image.width+1,top+image.height+1),outline='#24866E',width=2)
                        label_center(d,f'Observed RGB · {spec["frame"]/20:.1f} s',center,top+image.height+9,17)
                path=root/case['id']/'renders'/model['label']/('cameras.png' if cameras else 'clean.png')
                image=white_image(path);assert image.size==(1200,640)
                result.paste(image,(x,y+65+strip))
                for j,frame in enumerate(manifest['frames']):label_center(d,f'{frame/20:.1f} s',x+anchors[j],y+710+strip,17,color='#666477')
            if columns*rows>len(case['models']):
                import textwrap
                i=len(case['models']);x=(i%columns)*panel_w;y=header+(i//columns)*panel_h
                label_center(d,'Caption supplied to Ours + text',x+panel_w/2,y+75,26,True)
                for j,line in enumerate(textwrap.wrap(case['caption'],width=66)):
                    label_center(d,line,x+panel_w/2,y+140+j*34,24)
                label_center(d,'Same checkpoint, video, shape, seed and 35-step sampler',x+panel_w/2,y+420,21,color='#666477')
                label_center(d,'Caption supplied to both reasoner paths; guidance 1',x+panel_w/2,y+460,21,color='#666477')
            top=result.height-foot
            if cameras:
                if manifest.get('camera_glyph')=='axes':
                    from .camera_geometry import AXIS_COLORS
                    label_center(d,'Measured camera axes:',w/2-310,top+6,19)
                    for j,(text,color) in enumerate(zip(['X right','Y down','Z forward'],AXIS_COLORS)):
                        label_center(d,text,w/2-80+j*180,top+6,19,True,color=color)
                else:label_center(d,'Green camera glyphs: measured reference poses · schematic size/FOV · shared across methods',w/2,top+6,19,color='#24866E')
            if strip:label_center(d,'RGB repeated for alignment; these are shared observed images, not generated views of each predicted body.',w/2,top+34,17,color='#666477')
            with Image.open(root/'error_legend.png') as image:
                legend=Image.new('RGBA',image.size,'white');legend.alpha_composite(image.convert('RGBA'));legend=legend.convert('RGB');legend.thumbnail((750,125))
                result.paste(legend,((w-legend.width)//2,top+60))
            path=root/case['id']/(variant+'.png');result.save(path,dpi=(300,300))
            result.save(path.with_suffix('.pdf'),resolution=300.)
            variants[variant]=dict(image=str(path.relative_to(root)),pdf=str(path.with_suffix('.pdf').relative_to(root)),sha256=sha(path))
        entries.append(dict(id=case['id'],variants=variants))
        print('COMPOSED',case['id'],flush=True)
    import html
    cards=''
    for item in entries:
        cards+='<section><h2>'+html.escape(item['id'])+'</h2>'
        for variant,paths in item['variants'].items():cards+='<h3>'+variant+'</h3><a href="'+paths['pdf']+'">PDF</a><img src="'+paths['image']+'">'
        cards+='</section>'
    camera_description=('Camera-local axes: X red/right, Y green/down, Z blue/forward.' if manifest.get('camera_glyph')=='axes' else 'Green schematic camera frustums.')
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Sparse SOMA + cameras</title><style>body{max-width:1800px;margin:30px auto;font:16px system-ui;background:white;color:#27233c}img{width:100%}section{margin:50px 0}</style><h1>Sparse motion, cameras and observed RGB</h1><p>Three snapshots per scene. Display translations separate timepoints; they do not depict a camera trajectory. Colors are pelvis-relative errors. Cameras show measured references. '+camera_description+'</p>'+cards)
    write(root/'COMPOSED.json',dict(passed=True,cases=entries))


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('prepare');q.add_argument('--config',type=Path,required=True)
    q=s.add_parser('compose');q.add_argument('--root',type=Path,required=True)
    a=p.parse_args();prepare(a.config) if a.command=='prepare' else compose(a.root)
