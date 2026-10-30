"""Recompute a collision-body policy using an audited approved-cloud GT cache."""
import argparse
import csv
import json
import os
from pathlib import Path
import socket

import numpy as np

from nymeria_baselines.evaluate_gt_scene_collision import (
    CAPSULES, collision_exclusions, decode_uniego, geometry_acceptance, log, save, score, sha,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--collision-body', choices=['exclude_thigh_shin_foot', 'full_body'],
                        default='exclude_thigh_shin_foot')
    args = parser.parse_args()
    host = socket.gethostname()
    if not os.environ.get('SLURM_JOB_ID') or 'login' in host or host in {'cs-gpu-01', 'node37'}:
        raise RuntimeError('An allowed owned compute allocation is required')
    if args.out.exists():
        raise RuntimeError('Choose a fresh output directory for the changed metric policy')
    done = json.loads((args.source / 'COMPLETE.json').read_text())
    assert done['state'] == 'complete_verified' and done['windows'] == 71 and done['failed'] == 0
    old_config = json.loads((args.source / 'configuration.json').read_text())
    assert old_config['arguments']['cloud_filter'] == 'nymeria_legacy_sdk'
    assert old_config['arguments'].get('collision_body', 'full_body') == 'full_body'
    assert sha(args.source / 'evaluator_source.py') == old_config['code_sha256']
    old_cases = json.loads((args.source / 'per_case.json').read_text())
    cohort = json.loads(Path(old_config['arguments']['cohort']).read_text())['cases']
    lookup = {c['id']: c for c in cohort}
    assert len(old_cases) == len(lookup) == 71
    mean, std = [np.load(old_config['arguments'][k]).astype(np.float32) for k in ['mean', 'std']]
    excluded = collision_exclusions(args.collision_body)
    source_code = Path(__file__).with_name('evaluate_gt_scene_collision.py')
    config = {**old_config, 'arguments': {**old_config['arguments'], 'out': str(args.out),
              'collision_body': args.collision_body, 'previews': 0}, 'code_sha256': sha(source_code),
              'rescorer_sha256': sha(Path(__file__)), 'reused_geometry_from': str(args.source),
              'source_configuration_sha256': sha(args.source / 'configuration.json'),
              'excluded_capsule_names': list(excluded), 'host': host, 'job': os.environ['SLURM_JOB_ID']}
    args.out.mkdir(parents=True)
    save(args.out / 'configuration.json', config)
    save(args.out / 'geometry_acceptance.json', geometry_acceptance())
    save(args.out / 'publication.json', json.loads((args.source / 'publication.json').read_text()))
    completed, checks = [], []
    def summary(state):
        value = {'state': state, 'model': 'GT', 'requested': 71, 'completed': len(completed),
                 'failed': 0, 'failures': [], 'cloud_filter': 'nymeria_legacy_sdk',
                 'collision_body': args.collision_body, 'physical_collision_validated': False,
                 'excluded_capsule_names': list(excluded), 'reused_geometry_from': str(args.source)}
        if completed:
            keys = ['collision_frame_percent', 'colliding_body_part_frame_percent',
                    'intrusion_depth_mean_cm', 'intrusion_depth_p95_cm']
            value['aggregate'] = {k: float(np.mean([c[k] for c in completed])) for k in keys}
            value['aggregate'].update({'weighting': 'equal fixed window; depth summaries include zero-intrusion windows',
                'collision_window_percent': float(np.mean([c['collision_frames'] > 0 for c in completed])*100),
                'collision_frames': sum(c['collision_frames'] for c in completed),
                'total_frames': sum(c['frames'] for c in completed)})
            value['body_parts'] = {k: float(np.mean([c['body_parts'][k] for c in completed]))
                                   for k in completed[0]['body_parts']}
            value['sensitivity'] = [{**s, 'collision_frame_percent': float(np.mean([
                c['sensitivity'][i]['collision_frame_percent'] for c in completed]))}
                for i, s in enumerate(completed[0]['sensitivity'])]
        save(args.out / 'progress.json', value)
        return value
    summary('running')
    for old in old_cases:
        case = lookup[old['case']]
        source = args.source / 'cases' / old['case'] / 'geometry.npz'
        with np.load(source) as z:
            arrays = {k: z[k] for k in z.files}
        with np.load(case['bundle']) as z:
            assert str(z['motion_mean_sha256']) == old_config['arguments']['mean_sha']
            assert str(z['motion_std_sha256']) == old_config['arguments']['std_sha']
            motion = decode_uniego(z['motion'] * std + mean)
        assert np.array_equal(motion.world_positions.astype(np.float32), arrays['joints'])
        points, radii = arrays['points'], arrays['radii']
        # Full-policy control must reproduce every earlier numeric measurement.
        baseline, old_ids, old_hits, _ = score(points, motion.world_positions, radii, excluded_capsules=())
        assert all(baseline[k] == old[k] for k in [
            'collision_frame_percent', 'collision_frames', 'colliding_body_part_frame_percent',
            'intrusion_depth_mean_cm', 'intrusion_depth_p95_cm', 'intruding_voxel_frame_count',
            'near_body_scene_voxels_min', 'near_body_scene_voxels_median', 'body_parts', 'sensitivity']), old['case']
        assert np.array_equal(old_hits, arrays['hits'])
        metrics, ids, hits, depths = score(points, motion.world_positions, radii, excluded_capsules=excluded)
        assert all(np.isin(a, b).all() for a, b in zip(ids, old_ids)), old['case']
        assert metrics['collision_frames'] <= baseline['collision_frames']
        inactive = [i for i, c in enumerate(CAPSULES) if c[0] in excluded]
        assert not hits[:, inactive].any()
        dest = args.out / 'cases' / old['case']; dest.mkdir(parents=True)
        arrays.update({'hits': hits, 'point_ids': np.concatenate(ids),
                       'point_offsets': np.r_[0, np.cumsum([len(v) for v in ids])],
                       'intrusion_depths': np.concatenate(depths)})
        np.savez_compressed(dest / 'geometry.npz', **arrays)
        result = {**old, **metrics, 'collision_body': args.collision_body,
                  'source_geometry_sha256': sha(source), 'source_metrics_sha256': sha(source.with_name('metrics.json'))}
        save(dest / 'metrics.json', result); completed.append(result)
        checks.append({'case': old['case'], 'historical_full_body_exact': True,
                       'new_intrusion_points_subset': True, 'excluded_capsules_zero_hits': True})
        summary('running')
        log('case_rescored', case=old['case'], completed=len(completed),
            collision_frame_percent=metrics['collision_frame_percent'])
    final = summary('complete')
    save(args.out / 'summary.json', final); save(args.out / 'per_case.json', completed)
    save(args.out / 'recomputation_checks.json', checks)
    # This records the control reproduction of the accepted visual protocol;
    # changed-policy flags intentionally differ and will get fresh overlays.
    save(args.out / 'approved_visual_parity.json', {'full_body_control_reproduces_all71': True,
        'approved_cloud_and_coordinates_unchanged': True, 'current_collision_body': args.collision_body})
    keys = ['case', 'model', 'collision_frame_percent', 'colliding_body_part_frame_percent',
            'intrusion_depth_mean_cm', 'intrusion_depth_p95_cm', 'near_body_scene_voxels_min']
    with (args.out / 'per_case.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore'); writer.writeheader(); writer.writerows(completed)
    log('rescoring_complete', **final)


if __name__ == '__main__':
    main()
