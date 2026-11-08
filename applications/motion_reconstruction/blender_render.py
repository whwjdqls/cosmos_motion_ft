"""Run with Blender4.5+ --background --python THIS_FILE -- ...

Smooth full-resolution SOMA surfaces, fixed cameras, soft studio lighting and
per-vertex error colors. Never changes predicted poses, floor or error values.
"""
import argparse
import itertools
import hashlib
import json
from pathlib import Path
import sys
import time

import bpy
from mathutils import Vector
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from applications.motion_reconstruction.colors import error_colors,srgb_to_linear


def cycle_device():
    preferences=bpy.context.preferences.addons['cycles'].preferences
    for backend in ('OPTIX','CUDA'):
        try:
            preferences.compute_device_type=backend
            preferences.get_devices()
            available=[d for d in preferences.devices if d.type==backend]
            if available:
                for d in preferences.devices:d.use=d in available
                return dict(backend=backend,devices=[d.name for d in available])
        except (TypeError,RuntimeError):pass
    raise RuntimeError('No owned GPU available for Cycles; not silently using CPU')


def material(name,color):
    m=bpy.data.materials.new(name);m.use_nodes=True
    node=m.node_tree.nodes.get('Principled BSDF')
    node.inputs['Base Color'].default_value=(*color,1)
    node.inputs['Roughness'].default_value=.8
    node.inputs['Metallic'].default_value=0
    node.inputs['Specular IOR Level'].default_value=.18
    return m


def area(name,position,power,size,target):
    light=bpy.data.lights.new(name,'AREA');light.energy=power;light.shape='DISK';light.size=size
    obj=bpy.data.objects.new(name,light);bpy.context.collection.objects.link(obj)
    obj.location=position;obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--case',required=True);p.add_argument('--model',required=True)
    p.add_argument('--frames',default='all');p.add_argument('--width',type=int,default=768)
    p.add_argument('--height',type=int,default=1024);p.add_argument('--samples',type=int,default=64)
    p.add_argument('--tag',default='video');p.add_argument('--export-glb',action='store_true')
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    root=a.manifest.parent;manifest=json.loads(a.manifest.read_text())
    case=next(c for c in manifest['cases'] if c['id']==a.case)
    spec=next(m for m in case['models'] if m['label']==a.model)
    with np.load(root/spec['mesh'],allow_pickle=False) as z:
        vertices=z['vertices'].copy();errors=z['vertex_errors_m'].copy()
    with np.load(root/'assets/skin.npz',allow_pickle=False) as z:faces=z['faces'].copy()
    selected=list(range(len(vertices))) if a.frames=='all' else list(map(int,a.frames.split(',')))
    assert selected and min(selected)>=0 and max(selected)<len(vertices)
    output=root/a.case/'renders'/a.model/a.tag;output.mkdir(parents=True,exist_ok=True)
    request=dict(manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        case=a.case,model=a.model,width=a.width,height=a.height,samples=a.samples,tag=a.tag)
    request_path=output/'REQUEST.json'
    if request_path.exists():
        assert json.loads(request_path.read_text())==request,'Changed render recipe: use a fresh tag/output'
    else:request_path.write_text(json.dumps(request,indent=2)+'\n')
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene;scene.render.engine='CYCLES'
    device=cycle_device();scene.cycles.device='GPU';scene.cycles.samples=a.samples
    scene.cycles.use_adaptive_sampling=True;scene.cycles.adaptive_threshold=.02
    scene.cycles.use_denoising=True
    if device['backend']=='OPTIX':scene.cycles.denoiser='OPTIX'
    scene.cycles.max_bounces=4;scene.cycles.diffuse_bounces=2;scene.cycles.glossy_bounces=2
    scene.render.resolution_x=a.width;scene.render.resolution_y=a.height;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA'
    scene.render.image_settings.color_depth='8';scene.render.film_transparent=True
    scene.view_settings.view_transform='Standard';scene.view_settings.look='None'
    scene.view_settings.exposure=0;scene.view_settings.gamma=1
    scene.render.threads_mode='FIXED';scene.render.threads=4
    world=bpy.data.worlds.new('neutral studio');world.use_nodes=True
    world.node_tree.nodes['Background'].inputs[0].default_value=(1,1,1,1)
    world.node_tree.nodes['Background'].inputs[1].default_value=.25;scene.world=world
    conversion=np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
    center=np.asarray(case['view']['center_yup'])@conversion.T
    ext=np.asarray(case['view']['extent_yup']);half=ext/2
    corners=np.array([np.asarray(case['view']['center_yup'])+np.array(s)*half
        for s in itertools.product((-1,1),repeat=3)])@conversion.T
    az=np.deg2rad(case['view']['camera_azimuth_deg']);el=np.deg2rad(case['view']['camera_elevation_deg'])
    direction=np.array([np.sin(az)*np.cos(el),-np.cos(az)*np.cos(el),np.sin(el)])
    camera_data=bpy.data.cameras.new('common orthographic camera')
    camera=bpy.data.objects.new('common orthographic camera',camera_data)
    bpy.context.collection.objects.link(camera);camera.location=center+10*direction
    camera.rotation_euler=(Vector(center)-camera.location).to_track_quat('-Z','Y').to_euler()
    camera_data.type='ORTHO';camera_data.clip_start=.01;camera_data.clip_end=200
    rotation=np.asarray(camera.rotation_euler.to_matrix())
    projections=(corners-center)@rotation
    spans=np.ptp(projections,axis=0)
    camera_data.ortho_scale=float(max(spans[1]*1.15,spans[0]*a.height/a.width*1.15))
    scene.camera=camera
    mesh=bpy.data.meshes.new('full SOMA standard surface')
    mesh.from_pydata((vertices[0]@conversion.T).tolist(),[],faces.tolist());mesh.update()
    mesh.polygons.foreach_set('use_smooth',np.ones(len(faces),dtype=bool))
    body=bpy.data.objects.new('predicted body, unmodified world positions',mesh)
    bpy.context.collection.objects.link(body)
    attribute=mesh.color_attributes.new(name='joint_error',type='FLOAT_COLOR',domain='POINT')
    mesh.color_attributes.active_color_index=0
    m=material('indigo → rose measured joint error',(.5,.5,.5))
    nodes=m.node_tree.nodes;links=m.node_tree.links;principled=nodes.get('Principled BSDF')
    color=nodes.new('ShaderNodeVertexColor');color.layer_name='joint_error'
    links.new(color.outputs['Color'],principled.inputs['Base Color'])
    links.new(color.outputs['Color'],principled.inputs['Emission Color'])
    principled.inputs['Emission Strength'].default_value=.12
    body.data.materials.append(m)
    # This floor is exactly the benchmark Y=0 after the proper basis rotation.
    bpy.ops.mesh.primitive_plane_add(size=200,location=(float(center[0]),float(center[1]),0))
    floor=bpy.context.object;floor.name='unchanged benchmark floor Y=0'
    floor.data.materials.append(material('soft ivory floor',(.94,.945,.955)))
    floor.is_shadow_catcher=True
    target=center.tolist()
    shadow_style=manifest.get('shadow_style','studio_three_light')
    if shadow_style=='single_back':
        # One distant soft directional source on the viewer-facing side.
        # Its horizontal rays point away from the camera: one rear shadow.
        light=bpy.data.lights.new('single soft front key','SUN')
        light.energy=2.;light.angle=np.deg2rad(6.)
        lamp=bpy.data.objects.new('single soft front key',light)
        bpy.context.collection.objects.link(lamp)
        horizontal=direction.copy();horizontal[2]=0;horizontal/=np.linalg.norm(horizontal)
        lamp.location=center+horizontal*4+np.array([0.,0.,7.])
        lamp.rotation_euler=(Vector(center)-lamp.location).to_track_quat('-Z','Y').to_euler()
        shadow_direction=-horizontal
    elif shadow_style=='studio_three_light':
        area('large soft key',(center[0]-3,center[1]-4,center[2]+5),450,5,target)
        area('soft fill',(center[0]+4,center[1]-1,center[2]+3),180,4,target)
        area('rim',(center[0],center[1]+4,center[2]+4),280,3,target)
        shadow_direction=None
    else:raise ValueError('Unknown shadow style')
    started=time.monotonic();rendered=[]
    for frame in selected:
        image=output/f'{frame:06d}.png'
        rgb=error_colors(errors[frame],manifest['maximum_cm'],**{k:manifest['palette'][k] for k in ('low','high')})
        rgba=np.column_stack((srgb_to_linear(rgb),np.ones(len(rgb)))).astype(np.float32)
        mesh.vertices.foreach_set('co',(vertices[frame]@conversion.T).astype(np.float32).ravel())
        attribute.data.foreach_set('color',rgba.ravel());mesh.update()
        if not image.exists():
            scene.render.filepath=str(image);bpy.ops.render.render(write_still=True)
        rendered.append(str(image.relative_to(root)))
        print('RENDERED',a.case,a.model,frame,round(time.monotonic()-started,2),flush=True)
    if a.export_glb:
        bpy.ops.object.select_all(action='DESELECT');body.select_set(True);bpy.context.view_layer.objects.active=body
        bpy.ops.export_scene.gltf(filepath=str(output/'colored_mesh.glb'),export_format='GLB',use_selection=True)
    record=dict(passed=True,case=a.case,model=a.model,frames=selected,resolution=[a.width,a.height],samples=a.samples,
        device=device,blender=bpy.app.version_string,seconds=time.monotonic()-started,files=rendered,
        camera_matrix_world=[list(row) for row in camera.matrix_world],ortho_scale=camera_data.ortho_scale,
        color_management='Standard/sRGB exposure0 gamma1; linearized vertex albedo, soft neutral studio lighting',
        error_mode=manifest['error_mode'],maximum_cm=manifest['maximum_cm'],floor_y_m=0.,
        error_reference=manifest.get('error_reference','world'),shadow_style=shadow_style,
        shadow_ground_direction_blender=shadow_direction.tolist() if shadow_direction is not None else None,
        directional_light_count=1 if shadow_style=='single_back' else 3,
        exported_glb_frame=selected[-1] if a.export_glb else None)
    (output/'COMPLETE.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
