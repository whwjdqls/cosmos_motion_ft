"""Persistent GPU scene inspector. All overlay geometry is inspection-only."""
import argparse
import json
from pathlib import Path
import time
import traceback
import numpy as np
from PIL import Image
from .camera import CV_FROM_GL,validate_pose
from .render import Renderer
from .runtime import require_compute


def save_json(path,data):
    temporary=path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(data));temporary.replace(path)


class Inspector:
    def __init__(self,config):
        self.renderer=Renderer(config['scene'],config.get('navmesh'),config.get('cube_face_size',384),inspection=True)
        with np.load(config['camera'],allow_pickle=False) as data:
            self.rays,self.valid=data['rays'],data['valid']
        self.cached_pose=None
        self.cached_preview=None
        # The camera glyph follows actual calibrated fisheye boundary rays.
        y,x=np.nonzero(self.valid)
        center=np.array([127.5,127.5])
        angle=np.arctan2(y-center[1],x-center[0])
        radius=np.hypot(y-center[1],x-center[0])
        ring=[]
        for a in np.linspace(-np.pi,np.pi,24,endpoint=False):
            distance=np.abs(np.angle(np.exp(1j*(angle-a))))
            candidates=np.flatnonzero(distance<np.pi/48)
            index=candidates[np.argmax(radius[candidates])]
            ray=self.rays[y[index],x[index]]
            ring.append(ray/np.linalg.norm(ray)*.25)
        self.ring=np.array(ring)

    def line(self,a,b,color):
        mn=self.renderer.mn
        self.renderer.sim.get_debug_line_render().draw_transformed_line(mn.Vector3(*map(float,a)),mn.Vector3(*map(float,b)),mn.Color4(*color))

    def glyph(self,pose,color):
        pose=validate_pose(pose)
        origin=pose[:3,3]
        ring=self.ring@pose[:3,:3].T+origin
        for i in range(len(ring)):
            self.line(ring[i],ring[(i+1)%len(ring)],color)
            if i%3==0:self.line(origin,ring[i],color)
        self.line(origin,origin+pose[:3,2]*.4,color)
        for axis in range(3):
            delta=pose[:3,axis]*.035
            self.line(origin-delta,origin+delta,color)

    def marker(self,position,color):
        mn=self.renderer.mn
        debug=self.renderer.sim.get_debug_line_render()
        for normal in ([0,1,0],[1,0,0],[0,0,1]):
            debug.draw_circle(mn.Vector3(*map(float,position)),.045,mn.Color4(*color),16,mn.Vector3(*normal))

    def render(self,request):
        import quaternion
        observer=validate_pose(request['observer_pose'])
        candidate=validate_pose(request['candidate_pose'])
        # Conditioning preview is drawn BEFORE any overlay lines are submitted.
        changed=self.cached_pose is None or not np.array_equal(candidate,self.cached_pose)
        if changed:
            self.cached_preview=self.renderer.render(candidate,self.rays,self.valid)
            self.cached_pose=candidate.copy()
        state=self.renderer.hs.AgentState()
        gl=observer@CV_FROM_GL
        state.position=gl[:3,3];state.rotation=quaternion.from_rotation_matrix(gl[:3,:3])
        self.renderer.sim.get_agent(1).set_state(state,reset_sensors=True)
        debug=self.renderer.sim.get_debug_line_render();debug.set_line_width(2.)
        if request.get('overlays',True):
            self.glyph(candidate,(1.,.75,.1,1.))
            trajectories=request.get('trajectories',[])
            for trajectory in trajectories:
                poses=np.asarray(trajectory,float)
                if poses.ndim!=3 or poses.shape[1:]!=(4,4) or not np.isfinite(poses).all():raise ValueError('Invalid trajectory')
                for a,b in zip(poses[:-1,:3,3],poses[1:,:3,3]):self.line(a,b,(.1,.9,1.,1.))
                self.marker(poses[0,:3,3],(.2,1.,.3,1.))
                self.marker(poses[-1,:3,3],(1.,.2,.2,1.))
            selected=int(request.get('trajectory_index',len(trajectories)-1))
            if 0<=selected<len(trajectories):
                poses=trajectories[selected]
                frame=min(max(0,int(request.get('frame',0))),len(poses)-1)
                self.glyph(poses[frame],(1.,.3,1.,1.))
        obs=self.renderer.sim.get_sensor_observations(agent_ids=1)
        return np.asarray(obs['inspection'])[...,:3].copy(),self.cached_preview.copy(),changed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();require_compute()
    args.root.mkdir(parents=True,exist_ok=True)
    inspector=Inspector(json.loads(args.config.read_text()))
    last=None
    save_json(args.root/'worker.json',dict(status='ready'))
    try:
        while True:
            path=args.root/'request.json'
            if not path.exists():time.sleep(.02);continue
            request=json.loads(path.read_text())
            if request['id']==last:time.sleep(.01);continue
            last=request['id'];start=time.perf_counter()
            try:
                image,preview,changed=inspector.render(request)
                for name,array,fmt in [('view.jpg',image,'JPEG'),('candidate.png',preview,'PNG')]:
                    if name=='candidate.png' and not changed and (args.root/name).exists():continue
                    temporary=args.root/(name+'.partial')
                    Image.fromarray(array).save(temporary,format=fmt,**({'quality':85} if fmt=='JPEG' else {}))
                    temporary.replace(args.root/name)
                save_json(args.root/'response.json',dict(id=last,status='ready',render_ms=1000*(time.perf_counter()-start),
                    observer_pose=request['observer_pose'],candidate_pose=request['candidate_pose'],frame=request.get('frame',0)))
            except Exception as exc:
                traceback.print_exc()
                save_json(args.root/'response.json',dict(id=last,status='error',error=str(exc)))
    finally:inspector.renderer.sim.close()


if __name__=='__main__':main()
