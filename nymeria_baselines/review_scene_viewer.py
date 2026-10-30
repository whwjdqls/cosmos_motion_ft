"""Render fixed GT cases with the legacy Nymeria viewer's confidence filter.

Run inside an owned compute allocation. The display uses raw points; collision
diagnostics continue to use all 2cm voxels and remain explicitly unvalidated.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np

from nymeria_baselines.evaluate_gt_scene_collision import (
    Publisher, collision_exclusions, read_scene, render, save, score,
)
from motion_expert_joint_attention.camera_head_recanonicalization import (
    ARIA_Z_UP_TO_KIMODO_Y_UP as AXES, decode_uniego,
)


LEGACY_REV = '47ef97cc7db00c239a5edbf887f5da1b5a3bbea1'
PROTOCOL = (
    'Legacy Nymeria confidence filter: inv_dist_std <0.0004 and dist_std <0.02m '
    'after SDK float32 confidence storage and promotion to Python floats. '
    'Display: every finite raw filtered head-map point in the matching SLAM graph. '
    'No point-count cap, spatial crop, ceiling filter, voxelization or GT-overlap removal for display. '
    'Legacy Rerun draws available head/wrist/observer clouds, gray RGBA128, '
    'point radius0.008m; its provider requests100000/device, but a fresh cache '
    'uses the recording-loader default50000/device. Our videos use head only '
    'and an offscreen OpenGL depth buffer, not the Rerun renderer. '
    'Collision diagnostics use separate full 2cm voxel geometry and are provisional.'
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operation', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--case-indices', nargs='+', type=int, default=[3, 0, 1, 2])
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--collision-body', choices=['exclude_thigh_shin_foot', 'full_body'], default='exclude_thigh_shin_foot')
    args = parser.parse_args()
    host = socket.gethostname().split('.')[0]
    if not os.environ.get('SLURM_JOB_ID') or host in {'cs-gpu-01', 'node37'} or 'login' in host:
        raise RuntimeError('An owned allowed compute allocation is required')
    args.out.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.operation / 'configuration.json').read_text())['arguments']
    cases = json.loads(Path(config['cohort']).read_text())['cases']
    mean, std = [np.load(config[k]).astype(np.float32) for k in ['mean', 'std']]
    repo = Path(__file__).resolve().parents[1]
    rh = np.asarray(json.loads((repo / 'motion_expert_joint_attention/head_camera_calibration_train.json').read_text())[
        'rotation_head_to_upright_camera'])
    args.notion_helper = Path(config['notion_helper']) if args.publish else None
    args.display_point_limit = None
    args.renderer = 'opengl'
    pub = Publisher(args)
    if args.publish and not (args.out / 'publication.json').exists():
        prior = args.operation / 'visual_review_v2/publication.json'
        save(args.out / 'publication.json', json.loads(prior.read_text()))
    save(args.out / 'configuration.json', {
        'operation': str(args.operation), 'case_indices': args.case_indices,
        'legacy_revision': LEGACY_REV, 'viewer_protocol': PROTOCOL,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'compute_host': host, 'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'collision_scores_validated': False,
        'collision_body': args.collision_body,
        'excluded_capsule_names': list(collision_exclusions(args.collision_body)),
    })
    media, reports = [], []
    for index in args.case_indices:
        case = cases[index]
        prior = args.operation / 'cases' / case['id']
        old = json.loads((prior / 'metrics.json').read_text())
        with np.load(prior / 'geometry.npz') as z:
            radii = z['radii']
        alignment = old['alignment']
        rotation = np.asarray(alignment['canonical_rotation_world_from_window'])
        origin = np.asarray(alignment['canonical_origin_world_yup'])
        off = alignment['floor_offset_m']
        with np.load(case['bundle']) as z:
            motion = decode_uniego(z['motion'] * std + mean)
        with np.load(Path(config['data_root']) / 'camera_rgb' / (case['uuid'] + '.npz')) as z:
            start = int(case['start'])
            positions = z['cam_world_pos_upright'][start:start+97].astype(float)
            cam_rot = rotation.T @ AXES @ z['cam_world_rot_upright'][start:start+97].astype(float)
            cam_positions = (positions @ AXES.T - origin - [0, off, 0]) @ rotation
        cloud = Path(config['cloud_root']) / case['uuid'].split('/')[-1] / 'recording_head/mps/slam/semidense_points.csv.gz'
        points, coverage, raw = read_scene(
            cloud, alignment['graph_uid'], rotation, origin, off, motion.world_positions,
            max_inv_std=.0004, max_dist_std=.02, strict_confidence=True, return_display_points=True)
        metrics, ids, hits, _ = score(points, motion.world_positions, radii,
            excluded_capsules=collision_exclusions(args.collision_body))
        arrays = {
            'points': points, 'display_points': raw, 'joints': motion.world_positions,
            'hits': hits, 'radii': radii, 'point_ids': ids,
            'head_rotation0': motion.world_rotations[0, 6],
            'camera_positions': cam_positions, 'camera_forward': cam_rot[:, :, 2],
            'head_camera_forward': (motion.world_rotations[:, 6] @ rh)[:, :, 2],
            'torso_forward': motion.world_rotations[:, 0, :, 2],
            'excluded_capsule_names': metrics['excluded_capsule_names'],
        }
        preview = render(args, case, arrays)
        if preview['display_point_count'] != coverage['confident_point_count']:
            raise RuntimeError('Display must retain every finite confidence-filtered graph point')
        media.append(preview)
        reports.append({'case': case['id'], 'coverage': coverage,
                        'diagnostic_capsule_overlap_not_validated_collision': metrics})
        save(args.out / 'media.json', media)
        save(args.out / 'per_case.json', reports)
        print(json.dumps({'event': 'rendered', **preview}), flush=True)
    summary = {'state': 'visual_review_legacy_allpoints_complete_collision_unvalidated',
               'visualization_only': True, 'completed': len(media), 'failed': 0,
               'viewer_protocol': PROTOCOL + (' Collision checks exclude both thighs, shins and feet; hips remain included.'
                   if args.collision_body == 'exclude_thigh_shin_foot' else ' Historical full-body collision checks.'),
               'collision_body': args.collision_body}
    save(args.out / 'summary.json', summary)
    if args.publish:
        pub.publish(summary, media)
        state = json.loads((args.out / 'publication.json').read_text())
        for item in media:
            for kind in ['png', 'mp4']:
                digest = hashlib.sha256(Path(item[kind]).read_bytes()).hexdigest()
                if not state['uploaded'].get(digest, {}).get('remote_sha256_verified'):
                    raise RuntimeError('Notion media not remotely hash-verified')
    save(args.out / 'COMPLETE.json', {**summary, 'notion_verified': args.publish})


if __name__ == '__main__':
    main()
