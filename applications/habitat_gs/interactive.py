"""Local browser UI around the unchanged evaluated rollout pipeline."""
import argparse
import copy
import fcntl
import hmac
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import secrets
import subprocess
import threading
import time
from urllib.parse import unquote, urlsplit, parse_qs
import numpy as np
from .camera import sha256,validate_pose
from .profile import verify
from .rollout import atomic_json
from .runtime import require_compute


class Session:
    def __init__(self, config_path, root):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.file_lock = (self.root/'server.lock').open('a')
        fcntl.flock(self.file_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.config = json.loads(os.path.expandvars(config_path.read_text()))
        profile = json.loads(Path(self.config['profile']).read_text())
        verify(profile)
        with np.load(self.config['camera'],allow_pickle=False) as camera:
            if str(camera['mode'])!='measured_aria_unprojection_rays':
                raise ValueError('Interactive mode requires measured Nymeria fisheye calibration')
        identity = dict(config=self.config,profile_sha256=sha256(self.config['profile']),
                        camera_sha256=sha256(self.config['camera']),scene_sha256=sha256(self.config['scene']))
        frozen = self.root/'session_config.json'
        if frozen.exists() and json.loads(frozen.read_text())!=identity:
            raise ValueError('Session configuration changed; use a fresh session directory')
        atomic_json(frozen,identity)
        self.lock = threading.RLock()
        self.process = None
        self.inspect_process = None
        self.access_token = None
        self.state = json.loads((self.root/'state.json').read_text()) if (self.root/'state.json').exists() else dict(
            stage='initializing',busy=True,history=[],attempt=0,error=None,current_image=None,current_pose=None)
        self.state.update(busy=False,error=None)
        if self.state['current_pose'] is not None:
            previous=self.state['history'][-1]['output'] if self.state['history'] else None
            image=Path(previous)/'segment_000/next_condition.png' if previous else self.root/'initial.png'
            accepted=self.state.get('accepted_image') or self.url(image)
            self.update(stage='ready',current_image=accepted,accepted_image=accepted)
        else:
            self.update(busy=True,stage='rendering_initial_view')
            threading.Thread(target=self.initialize,daemon=True).start()
        self.inspect_root=self.root/'inspector'
        self.inspect_root.mkdir(exist_ok=True)
        # JSON numbers must remain exactly representable in browser JavaScript.
        self.view_id=int(time.time()*1000)
        inspector_config=self.inspect_root/'config.json'
        atomic_json(inspector_config,self.config)
        env=self.render_env()
        if os.environ.get('APP_INSPECT_GPU'):
            env['CUDA_VISIBLE_DEVICES']=os.environ['APP_INSPECT_GPU']
        self.inspect_log=(self.root/'inspector.log').open('a')
        self.inspect_process=subprocess.Popen([self.config['habitat_python'],'-m','applications.habitat_gs.inspect_worker',
            '--config',str(inspector_config),'--root',str(self.inspect_root)],
            stdout=self.inspect_log,stderr=subprocess.STDOUT,env=env,start_new_session=True)

    def update(self, **fields):
        with self.lock:
            self.state.update(fields)
            atomic_json(self.root/'state.json',self.state)

    def url(self, path):
        return '/files/'+str(Path(path).resolve().relative_to(self.root))

    def render_env(self):
        env = os.environ.copy()
        cuda = Path(os.environ.get('CUDA_HOME','/opt/ohpc/pub/apps/cuda/12.8'))
        env['LD_LIBRARY_PATH']=os.pathsep.join(str(p) for p in (cuda/'lib64',cuda/'targets/x86_64-linux/lib') if p.is_dir())
        return env

    def run(self, command, log, env=None, monitor=None):
        with Path(log).open('a') as stream:
            stream.write('\nCOMMAND '+repr(command)+'\n');stream.flush()
            self.process = subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            while self.process.poll() is None:
                if monitor:monitor()
                time.sleep(.5)
            code = self.process.returncode
            self.process = None
            if code:raise RuntimeError(f'Process failed (exit {code}); see {log}')

    def initialize(self):
        try:
            image = self.root/'initial.png'
            command=[self.config['habitat_python'],'-m','applications.habitat_gs.render',
                '--scene',self.config['scene'],'--camera',self.config['camera'],
                '--face-size',str(self.config.get('cube_face_size',384)),'--seed',str(self.config['seed']),
                '--camera-height',str(self.config.get('initial_camera_height_m',1.5)*self.config['scene_units_per_meter']),
                '--out',str(image)]
            if self.config.get('navmesh'):command+=['--navmesh',self.config['navmesh']]
            if self.config.get('start_pose'):
                pose=self.root/'start.pose.json';atomic_json(pose,self.config['start_pose']);command+=['--pose',str(pose)]
            self.run(command,self.root/'initial_render.log',env=self.render_env())
            self.update(current_image=self.url(image),accepted_image=self.url(image),current_pose=json.loads(image.with_suffix('.pose.json').read_text()),
                        stage='ready',busy=False)
        except Exception as exc:self.update(stage='error',busy=False,error=str(exc))

    def submit(self, prompt):
        if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>4000:
            raise ValueError('Enter a prompt between 1 and 4000 characters')
        with self.lock:
            if self.state['busy']:raise ValueError('An iteration is already running')
            if self.state['current_pose'] is None:raise ValueError('Initial view is not ready')
            self.update(busy=True,error=None,stage='generating',attempt=self.state['attempt']+1,
                        active_prompt=prompt,generated_video=None,replay_video=None,replay_frame=None)
            threading.Thread(target=self.step,args=(prompt,),daemon=True).start()

    def inspection_request(self,data):
        observer=validate_pose(data['observer_pose']).tolist()
        candidate=validate_pose(data['candidate_pose']).tolist()
        with self.lock:
            if self.inspect_process.poll() is not None:raise ValueError('Scene inspector stopped; see inspector.log')
            history=copy.deepcopy(self.state['history'])
            trajectories=[]
            for entry in history:
                with np.load(Path(entry['output'])/'segment_000/camera_trajectory.npz',allow_pickle=False) as source:
                    trajectories.append(source['camera_to_world_optical'].tolist())
            self.view_id+=1
            request=dict(id=self.view_id,observer_pose=observer,candidate_pose=candidate,
                trajectories=trajectories,trajectory_index=int(data.get('trajectory_index',len(trajectories)-1)),
                frame=int(data.get('frame',0)),overlays=bool(data.get('overlays',True)))
            atomic_json(self.inspect_root/'request.json',request)
            return self.view_id

    def apply_pose(self,pose):
        pose=validate_pose(pose).tolist()
        with self.lock:
            if self.state['busy']:raise ValueError('Wait for the active iteration before changing its start pose')
            self.update(busy=True,stage='setting_start_pose',error=None)
            threading.Thread(target=self._apply_pose,args=(pose,),daemon=True).start()

    def _apply_pose(self,pose):
        previous_image=self.state.get('accepted_image',self.state['current_image'])
        try:
            root=self.root/'pose_edits'/str(time.time_ns())
            root.mkdir(parents=True)
            path=root/'pose.json';atomic_json(path,pose)
            image=root/'conditioning.png'
            command=[self.config['habitat_python'],'-m','applications.habitat_gs.render',
                '--scene',self.config['scene'],'--camera',self.config['camera'],'--pose',str(path),
                '--face-size',str(self.config.get('cube_face_size',384)),'--out',str(image)]
            if self.config.get('navmesh'):command+=['--navmesh',self.config['navmesh']]
            self.run(command,root/'render.log',env=self.render_env())
            atomic_json(root/'edit.json',dict(previous_pose=self.state['current_pose'],pose=pose,
                previous_image=previous_image,conditioning_sha256=sha256(image),explicit_user_pose_edit=True))
            self.update(current_pose=pose,current_image=self.url(image),accepted_image=self.url(image),
                        stage='ready',busy=False)
        except Exception as exc:self.update(stage='error',busy=False,error=str(exc),current_image=previous_image)

    def step(self,prompt):
        try:
            index=len(self.state['history'])
            turn=self.root/f'turn_{index:03d}_attempt_{self.state["attempt"]:03d}'
            config=copy.deepcopy(self.config)
            config.update(output=str(turn),segments=1,prompt=prompt,start_pose=self.state['current_pose'],seed=int(config['seed'])+index)
            path=self.root/f'config_{index:03d}_attempt_{self.state["attempt"]:03d}.json'
            atomic_json(path,config)
            def monitor():
                progress=turn/'progress.json'
                fields={}
                if progress.exists():
                    phase=json.loads(progress.read_text())['stage']
                    fields['stage']='finishing_inference' if phase=='complete' else phase
                sample=turn/'segment_000/inference/rollout_000_image2video/sample_outputs.json'
                if sample.exists():
                    fields['generated_video']=self.url(sample.parent/'vision.mp4')
                if fields:self.update(**fields)
            self.run([os.environ['COSMOS_PYTHON'],'-m','applications.habitat_gs.rollout','--config',str(path)],
                     self.root/f'{turn.name}.log',monitor=monitor)
            segment=turn/'segment_000'
            replay=turn/'camera_replay'
            self.update(stage='rendering_camera_path',generated_video=self.url(turn/'rollout_preview.mp4'))
            def monitor_replay():
                progress=replay/'progress.json'
                if progress.exists():
                    frame=json.loads(progress.read_text())['frame']
                    self.update(replay_frame=frame,current_image=self.url(replay/'latest.png'))
            self.run([config['habitat_python'],'-m','applications.habitat_gs.replay','--config',str(path),
                '--trajectory',str(segment/'camera_trajectory.npz'),'--out',str(replay)],
                turn/'camera_replay.log',env=self.render_env(),monitor=monitor_replay)
            receipt=json.loads((segment/'COMPLETE.json').read_text())
            # The replay must finish at the exact next conditioning pose/image.
            from PIL import Image
            if not np.array_equal(np.asarray(Image.open(replay/'latest.png')),np.asarray(Image.open(segment/'next_condition.png'))):
                raise RuntimeError('Replay endpoint differs from next conditioning render')
            entry=dict(step=index+1,prompt=prompt,seed=config['seed'],output=str(turn),
                generated_video=self.url(turn/'rollout_preview.mp4'),replay_video=self.url(replay/'camera_replay.mp4'),
                boundary=self.url(segment/'boundary.png'))
            with self.lock:
                self.update(history=self.state['history']+[entry],stage='ready',busy=False,
                    current_pose=receipt['end_pose'],current_image=self.url(segment/'next_condition.png'),
                    accepted_image=self.url(segment/'next_condition.png'),
                    replay_video=entry['replay_video'],replay_frame=97)
        except Exception as exc:
            # Preserve the last accepted pose. Failed attempts never advance it.
            with self.lock:
                self.update(stage='error',busy=False,error=str(exc),current_image=self.state.get('accepted_image',self.state['current_image']))


def handler_for(session):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass

        def authenticated(self):
            if session.access_token is None:return True
            cookie=SimpleCookie()
            try:cookie.load(self.headers.get('Cookie',''))
            except Exception:return False
            value=cookie.get('habitat_access')
            return value is not None and hmac.compare_digest(value.value.encode(),session.access_token.encode())

        def reply(self,status,value):
            body=json.dumps(value).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)))
            self.end_headers();self.wfile.write(body)

        def do_GET(self):
            token=parse_qs(urlsplit(self.path).query).get('token',[''])[0]
            if session.access_token and hmac.compare_digest(token.encode(),session.access_token.encode()):
                self.send_response(303)
                self.send_header('Set-Cookie','habitat_access='+session.access_token+'; HttpOnly; SameSite=Strict; Path=/')
                self.send_header('Location','/');self.send_header('Cache-Control','no-store')
                self.end_headers();return
            if not self.authenticated():self.send_error(401,'Open the private browser URL from browser_url.txt');return
            route=unquote(urlsplit(self.path).path)
            if route=='/api/state':
                with session.lock:state=copy.deepcopy(session.state)
                state['scene']=Path(session.config['scene']).stem
                state['default_prompt']=session.config['prompt']
                self.reply(200,state);return
            if route=='/api/view':
                path=session.inspect_root/'response.json'
                result=json.loads(path.read_text()) if path.exists() else dict(status='starting')
                if session.inspect_process.poll() is not None:result=dict(status='error',error='Inspector stopped; see inspector.log')
                self.reply(200,result);return
            if route=='/api/trajectories':
                with session.lock:history=copy.deepcopy(session.state['history'])
                result=[]
                for entry in history:
                    with np.load(Path(entry['output'])/'segment_000/camera_trajectory.npz',allow_pickle=False) as data:
                        result.append(dict(step=entry['step'],prompt=entry['prompt'],poses=data['camera_to_world_optical'].tolist()))
                self.reply(200,result);return
            if route=='/':
                path=Path(__file__).with_name('interactive3d.html')
            elif route.startswith('/files/'):
                path=(session.root/route[len('/files/'):]).resolve()
                if not path.is_relative_to(session.root):self.send_error(403);return
            else:self.send_error(404);return
            if not path.is_file():self.send_error(404);return
            size=path.stat().st_size;start=0;end=size-1;status=200
            requested=self.headers.get('Range')
            if requested:
                try:
                    unit,span=requested.split('=',1);left,right=span.split('-',1)
                    if unit!='bytes' or ',' in span:raise ValueError()
                    if left:start=int(left);end=min(int(right),end) if right else end
                    else:start=max(0,size-int(right))
                    if not 0<=start<=end<size:raise ValueError()
                    status=206
                except ValueError:self.send_error(416);return
            self.send_response(status)
            self.send_header('Content-Type',mimetypes.guess_type(str(path))[0] or 'application/octet-stream')
            self.send_header('Accept-Ranges','bytes');self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length',str(end-start+1))
            if status==206:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.end_headers()
            try:
                with path.open('rb') as stream:
                    stream.seek(start);remaining=end-start+1
                    while remaining:
                        chunk=stream.read(min(1024*1024,remaining))
                        if not chunk:break
                        self.wfile.write(chunk);remaining-=len(chunk)
            except (BrokenPipeError,ConnectionResetError):pass

        def do_POST(self):
            if not self.authenticated():self.send_error(401);return
            if self.path not in ('/api/step','/api/view','/api/pose'):self.send_error(404);return
            origin=self.headers.get('Origin')
            if origin and urlsplit(origin).netloc!=self.headers.get('Host'):
                self.send_error(403);return
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':
                self.send_error(415);return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=20000:raise ValueError('Invalid request size')
                data=json.loads(self.rfile.read(length))
                if self.path=='/api/view':
                    self.reply(202,{'id':session.inspection_request(data)});return
                if self.path=='/api/pose':session.apply_pose(data['pose'])
                else:session.submit(data['prompt'])
                self.reply(202,{'accepted':True})
            except (ValueError,KeyError,TypeError) as exc:self.reply(409,{'error':str(exc)})
    return Handler


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--session',type=Path,required=True)
    p.add_argument('--port',type=int,default=8765)
    p.add_argument('--host',default='127.0.0.1',help='Use 0.0.0.0 only for token-protected access through the login host')
    args=p.parse_args();require_compute()
    session=Session(args.config,args.session)
    if args.host not in ('127.0.0.1','localhost','::1'):
        token_path=session.root/'access.token'
        if not token_path.exists():
            with token_path.open('x') as stream:
                os.chmod(token_path,0o600);stream.write(secrets.token_urlsafe(32))
        session.access_token=token_path.read_text().strip()
        browser=session.root/'browser_url.txt'
        browser.write_text(f'http://localhost:{args.port}/?token='+session.access_token+'\n')
        os.chmod(browser,0o600)
    server=ThreadingHTTPServer((args.host,args.port),handler_for(session))
    print(f'Interactive rollout listening on {args.host}:{args.port}; access through your SSH tunnel.',flush=True)
    import signal
    def stop(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        server.server_close()
        if session.process is not None and session.process.poll() is None:
            import signal
            os.killpg(session.process.pid,signal.SIGTERM)
        if session.inspect_process is not None and session.inspect_process.poll() is None:
            os.killpg(session.inspect_process.pid,signal.SIGTERM)
        session.inspect_log.close()


if __name__=='__main__':main()
