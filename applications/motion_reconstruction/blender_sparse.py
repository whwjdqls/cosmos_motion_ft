"""Blender entrypoint: three spaced poses in one scene, optional camera-coordinate axes."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from applications.motion_reconstruction.blender_render import cycle_device,material
from applications.motion_reconstruction.colors import error_colors,srgb_to_linear
from applications.motion_reconstruction.camera_geometry import camera_axis_segments,AXIS_COLORS


def line(name,points,mat,radius=.0025):
    data=bpy.data.curves.new(name,'CURVE');data.dimensions='3D';data.bevel_depth=radius;data.bevel_resolution=3
    poly=data.splines.new('POLY');poly.points.add(len(points)-1)
    for point,value in zip(poly.points,points):point.co=(*map(float,value),1)
    obj=bpy.data.objects.new(name,data);bpy.context.collection.objects.link(obj);obj.data.materials.append(mat)
    obj.visible_shadow=False
    return obj


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--case',required=True);p.add_argument('--model',required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);root=a.manifest.parent;manifest=json.loads(a.manifest.read_text())
    case=next(c for c in manifest['cases'] if c['id']==a.case);model=next(m for m in case['models'] if m['label']==a.model)
    with np.load(root/model['mesh'],allow_pickle=False) as z:vertices=z['vertices'].copy();errors=z['vertex_errors_m'].copy();roots=z['joint_positions'][:,0].copy()
    with np.load(root/'skin.npz',allow_pickle=False) as z:faces=z['faces'].copy()
    with np.load(root/case['camera_layout'],allow_pickle=False) as z:offsets=z['display_offsets_yup'].copy();poses=z['camera_to_common'][manifest['frames']].copy();glyph=z['schematic_corners_optical'].copy()
    assert hashlib.sha256((root/model['mesh']).read_bytes()).hexdigest()==model['sha256']
    output=root/a.case/'renders'/a.model;output.mkdir(parents=True,exist_ok=True)
    recipe=dict(renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),case=a.case,model=a.model,resolution=[1200,640],samples=128)
    request=output/'REQUEST.json'
    if request.exists():assert json.loads(request.read_text())==recipe,'Changed recipe: use fresh output'
    else:request.write_text(json.dumps(recipe,indent=2)+'\n')
    bpy.ops.wm.read_factory_settings(use_empty=True);scene=bpy.context.scene;scene.render.engine='CYCLES'
    device=cycle_device();scene.cycles.device='GPU';scene.cycles.samples=128;scene.cycles.use_denoising=True
    if device['backend']=='OPTIX':scene.cycles.denoiser='OPTIX'
    scene.cycles.use_adaptive_sampling=True;scene.cycles.adaptive_threshold=.015
    scene.cycles.max_bounces=4;scene.cycles.diffuse_bounces=2;scene.cycles.glossy_bounces=2
    scene.render.resolution_x=1200;scene.render.resolution_y=640;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA';scene.render.film_transparent=True
    scene.view_settings.view_transform='Standard';scene.view_settings.look='None';scene.view_settings.exposure=0;scene.view_settings.gamma=1
    scene.render.threads_mode='FIXED';scene.render.threads=4
    world=bpy.data.worlds.new('neutral ambient');world.use_nodes=True;world.node_tree.nodes['Background'].inputs[0].default_value=(1,1,1,1);world.node_tree.nodes['Background'].inputs[1].default_value=.25;scene.world=world
    basis=np.array([[1.,0,0],[0,0,-1],[0,1,0]])
    center=np.asarray(case['view']['center_yup'])@basis.T;extent=np.asarray(case['view']['extent_yup'])
    corners=np.array([np.asarray(case['view']['center_yup'])+np.asarray(s)*extent/2 for s in itertools.product((-1,1),repeat=3)])@basis.T
    az=np.deg2rad(28);el=np.deg2rad(12);direction=np.array([np.sin(az)*np.cos(el),-np.cos(az)*np.cos(el),np.sin(el)])
    data=bpy.data.cameras.new('shared observer');camera=bpy.data.objects.new('shared observer',data);bpy.context.collection.objects.link(camera)
    camera.location=center+10*direction;camera.rotation_euler=(Vector(center)-camera.location).to_track_quat('-Z','Y').to_euler();data.type='ORTHO';data.clip_start=.01;data.clip_end=200
    projected=(corners-center)@np.asarray(camera.rotation_euler.to_matrix());spans=np.ptp(projected,axis=0)
    # In landscape output Blender's ortho_scale spans the image width.
    data.ortho_scale=float(max(spans[0]*1.12,spans[1]*1200/640*1.12));scene.camera=camera
    for i,frame in enumerate(manifest['frames']):
        mesh=bpy.data.meshes.new(f'SOMA frame {frame}');mesh.from_pydata(((vertices[i]+offsets[i])@basis.T).tolist(),[],faces.tolist());mesh.update()
        mesh.polygons.foreach_set('use_smooth',np.ones(len(faces),dtype=bool));body=bpy.data.objects.new(f'Body frame {frame}, display translation only',mesh);bpy.context.collection.objects.link(body)
        attribute=mesh.color_attributes.new(name='joint_error',type='FLOAT_COLOR',domain='POINT');mesh.color_attributes.active_color_index=0
        rgb=error_colors(errors[i],30,**{k:manifest['palette'][k] for k in ['low','high']});rgba=np.column_stack((srgb_to_linear(rgb),np.ones(len(rgb)))).astype(np.float32);attribute.data.foreach_set('color',rgba.ravel())
        mat=material('pelvis-relative error',(.5,.5,.5));node=mat.node_tree.nodes.new('ShaderNodeVertexColor');node.layer_name='joint_error';principled=mat.node_tree.nodes.get('Principled BSDF')
        mat.node_tree.links.new(node.outputs['Color'],principled.inputs['Base Color']);mat.node_tree.links.new(node.outputs['Color'],principled.inputs['Emission Color']);principled.inputs['Emission Strength'].default_value=.12;mesh.materials.append(mat)
    bpy.ops.mesh.primitive_plane_add(size=200,location=(float(center[0]),float(center[1]),0));floor=bpy.context.object;floor.name='unchanged benchmark floor';floor.is_shadow_catcher=True;floor.data.materials.append(material('floor',(.94,.945,.955)))
    light=bpy.data.lights.new('single soft front key','SUN');light.energy=2;light.angle=np.deg2rad(6)
    obj=bpy.data.objects.new('single soft front key',light);bpy.context.collection.objects.link(obj)
    horizontal=direction.copy();horizontal[2]=0;horizontal/=np.linalg.norm(horizontal);obj.location=center+horizontal*4+[0,0,7];obj.rotation_euler=(Vector(center)-obj.location).to_track_quat('-Z','Y').to_euler()
    glyph_objects=[]
    style=manifest.get('camera_glyph','frustum')
    if style=='axes':
        displayed=poses.copy();displayed[:,:3,3]+=offsets
        for axis,color in enumerate(AXIS_COLORS):
            rgb=srgb_to_linear(np.array([int(color[i:i+2],16) for i in [1,3,5]])/255.)
            mat=material('camera '+['X right','Y down','Z forward'][axis],tuple(rgb))
            principled=mat.node_tree.nodes.get('Principled BSDF');principled.inputs['Emission Color'].default_value=(*rgb,1);principled.inputs['Emission Strength'].default_value=.5
            for segment in camera_axis_segments(displayed,manifest.get('camera_axis_length_m',.18))[:,axis]:
                points=segment@basis.T;glyph_objects.append(line('camera '+['X','Y','Z'][axis],points,mat,.003))
                direction=points[1]-points[0];direction/=np.linalg.norm(direction)
                bpy.ops.mesh.primitive_cone_add(vertices=16,radius1=.009,radius2=0,depth=.028,location=points[1]-direction*.014)
                arrow=bpy.context.object;arrow.rotation_euler=Vector(direction).to_track_quat('Z','Y').to_euler();arrow.data.materials.append(mat);arrow.visible_shadow=False;glyph_objects.append(arrow)
    elif style=='frustum':
        green=srgb_to_linear(np.array([0x24,0x86,0x6E])/255);mat=material('observed camera, green',tuple(green));principled=mat.node_tree.nodes.get('Principled BSDF');principled.inputs['Emission Color'].default_value=(*green,1);principled.inputs['Emission Strength'].default_value=.4
        for i,pose in enumerate(poses):
            apex=pose[:3,3]+offsets[i];ring=glyph@pose[:3,:3].T+apex
            glyph_objects.append(line('image-plane outline',np.concatenate((ring,ring[:1]))@basis.T,mat))
            for corner in ring:glyph_objects.append(line('camera ray',np.array([apex,corner])@basis.T,mat))
            glyph_objects.append(line('camera forward',np.array([apex,apex+pose[:3,2]*.27])@basis.T,mat,.003))
            up=pose[:3,3]+offsets[i]+pose[:3,2]*.20-pose[:3,1]*.12
            glyph_objects.append(line('camera top',np.array([ring[0],up,ring[1]])@basis.T,mat,.002))
    else:raise ValueError('Unknown camera glyph')
    bpy.context.view_layer.update()
    for variant in ['clean','cameras']:
        for obj in glyph_objects:obj.hide_render=variant=='clean'
        scene.render.filepath=str(output/(variant+'.png'));bpy.ops.render.render(write_still=True)
    anchors=[world_to_camera_view(scene,camera,Vector((point+offset)@basis.T)).x*1200 for point,offset in zip(roots,offsets)]
    ndc=[list(world_to_camera_view(scene,camera,Vector(point))) for point in corners]
    assert all(.02<=p[0]<=.98 and .02<=p[1]<=.98 for p in ndc),'Observer crops scene bounds'
    receipt=dict(passed=True,**recipe,frames=manifest['frames'],device=device,camera_matrix_world=[list(row) for row in camera.matrix_world],ortho_scale=data.ortho_scale,frame_anchor_pixels=anchors,bounds_ndc=ndc,body_offsets_yup=offsets.tolist(),camera_offsets_yup=offsets.tolist(),directional_shadow_lights=1,shadow_ground_direction_blender=(-horizontal).tolist(),camera_glyph=style,camera_kind='shared measured camera-coordinate reference',camera_axis_colors=list(AXIS_COLORS) if style=='axes' else None,variants=['clean','cameras'])
    (output/'COMPLETE.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
