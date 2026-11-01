"""Separate-process Habitat-GS renderer; cubemap reprojection honors measured rays."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .camera import CV_FROM_GL, cube_lookup, cube_rotations, validate_pose, sha256


class Renderer:
    def __init__(self, scene, navmesh, face_size=384, inspection=False):
        import habitat_sim
        import magnum as mn
        self.hs, self.mn, self.size = habitat_sim, mn, face_size
        cfg = habitat_sim.SimulatorConfiguration()
        cfg.scene_id = 'NONE'
        cfg.enable_physics = False
        cfg.gpu_device_id = 0
        cfg.gaussian_auto_play = False
        cfg.create_renderer = True
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        sensors = []
        for key, kind in [('rgb', habitat_sim.SensorType.COLOR), ('depth', habitat_sim.SensorType.DEPTH)]:
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid, spec.sensor_type = key, kind
            spec.resolution, spec.position, spec.hfov = [face_size, face_size], [0.,0.,0.], 90.
            if key == 'depth':
                spec.channels = 1
            sensors.append(spec)
        agent_cfg.sensor_specifications = sensors
        agents = [agent_cfg]
        if inspection:
            observer = habitat_sim.agent.AgentConfiguration()
            observer.sensor_specifications = []
            for key, kind in [('inspection',habitat_sim.SensorType.COLOR),('inspection_depth',habitat_sim.SensorType.DEPTH)]:
                spec = habitat_sim.CameraSensorSpec()
                spec.uuid, spec.sensor_type = key,kind
                spec.resolution, spec.position, spec.hfov = [512,768],[0.,0.,0.],80.
                if key=='inspection_depth':spec.channels=1
                observer.sensor_specifications.append(spec)
            agents.append(observer)
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, agents))
        self.helper = habitat_sim.RenderInstanceHelper(self.sim, use_xyzw_orientations=False)
        self.helper.add_instance(asset_filepath=str(Path(scene).resolve()), semantic_id=0,
                                 scale=mn.Vector3(1.,1.,1.))
        self.helper.set_world_poses(np.zeros((1,3), np.float32), np.array([[1.,0.,0.,0.]], np.float32))
        if navmesh is not None and not self.sim.pathfinder.load_nav_mesh(str(navmesh)):
            raise RuntimeError('Cannot load scene navmesh')

    def initial_pose(self, seed, camera_height):
        if not self.sim.pathfinder.is_loaded:
            raise ValueError('An explicit start pose or a valid navmesh is required')
        self.sim.pathfinder.seed(seed)
        floor = np.array(self.sim.pathfinder.get_random_navigable_point(), float)
        if not np.isfinite(floor).all():
            raise ValueError('No navigable initial point')
        rng = np.random.default_rng(seed)
        # Habitat world: Y up. Optical CV: right/down/forward. Identity GL looks -Z.
        yaw = rng.uniform(-np.pi, np.pi)
        c, s = np.cos(yaw), np.sin(yaw)
        pose = np.eye(4)
        pose[:3,:3] = np.array([[c,0,s],[0,1,0],[-s,0,c]]) @ CV_FROM_GL[:3,:3]
        pose[:3,3] = floor + [0, camera_height, 0]
        return validate_pose(pose)

    def render(self, pose, rays, valid):
        import quaternion
        import cv2
        agent = self.sim.get_agent(0)
        images = []
        for rotation in cube_rotations():
            face = np.eye(4)
            face[:3,:3] = rotation
            gl = validate_pose(pose) @ face @ CV_FROM_GL
            state = self.hs.AgentState()
            state.position = gl[:3,3]
            state.rotation = quaternion.from_rotation_matrix(gl[:3,:3])
            agent.set_state(state, reset_sensors=True)
            obs = self.sim.get_sensor_observations()
            images.append(np.asarray(obs['rgb'])[...,:3].copy())
        faces, uv, ray_valid = cube_lookup(rays, self.size)
        result = np.zeros((*rays.shape[:2],3), dtype=np.uint8)
        for index, image in enumerate(images):
            sampled = cv2.remap(image, uv[...,0], uv[...,1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            mask = (faces == index) & valid & ray_valid
            result[mask] = sampled[mask]
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--navmesh', type=Path)
    p.add_argument('--camera', type=Path, required=True)
    p.add_argument('--pose', type=Path)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--camera-height', type=float, default=1.5, help='Scene units; only for initial navmesh sampling')
    p.add_argument('--face-size', type=int, default=384)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    from .runtime import require_compute
    require_compute()
    with np.load(args.camera, allow_pickle=False) as data:
        rays, valid, mode = data['rays'], data['valid'], str(data['mode'])
    renderer = Renderer(args.scene, args.navmesh, args.face_size)
    try:
        pose = (np.asarray(json.loads(args.pose.read_text())) if args.pose else
                renderer.initial_pose(args.seed, args.camera_height))
        image = renderer.render(pose, rays, valid)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(args.out)
        args.out.with_suffix('.pose.json').write_text(json.dumps(validate_pose(pose).tolist(), indent=2)+'\n')
        args.out.with_suffix('.render.json').write_text(json.dumps(dict(camera_mode=mode,
            image_sha256=sha256(args.out), camera_sha256=sha256(args.camera), scene=str(args.scene),
            camera_to_world_optical=pose.tolist(), cube_face_size=args.face_size,
            valid_fraction=float(valid.mean())), indent=2)+'\n')
    finally:
        renderer.sim.close()


if __name__ == '__main__':
    main()
