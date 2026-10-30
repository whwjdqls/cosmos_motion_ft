"""CPU scene scoring for the immutable Phase2 modality benchmark."""
import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from nymeria_baselines.evaluate_gt_scene_collision import (
    AXES, CAPSULES, EXCLUDED_COLLISION_CAPSULES, cont6d_to_matrix, decode_uniego,
    ground_features, canonicalize_frame0, score, sha, save,
)

CONTEXT = None


def initialize(operation):
    global CONTEXT
    CONTEXT = SceneScorer(Path(operation))


def measure_case(row, predicted_features):
    return CONTEXT.measure(row, predicted_features)


class SceneScorer:
    def __init__(self, operation):
        self.operation = operation
        self.cfg = json.loads((operation / 'configuration.json').read_text())
        self.old = self.cfg['reference_configuration']
        self.reference = self.old['reference_request']
        self.gt_source = Path(self.cfg['audited_gt_operation'])
        self.gt_cfg = json.loads((self.gt_source / 'configuration.json').read_text())['arguments']
        self.audit = {r['uuid']: r for r in json.loads((self.gt_source / 'per_case.json').read_text())}
        self.current = None

    def recording(self, uuid):
        if self.current is not None and self.current['uuid'] == uuid:
            return self.current
        record = self.audit[uuid]
        graph = record['alignment']['graph_uid']
        assert record['coverage']['raw_point_count'] == record['coverage']['matching_graph_count'], (
            'Multi-graph maps require per-window graph selection', uuid)
        name = uuid.split('/')[-1]
        cloud = Path(self.gt_cfg['cloud_root']) / name / 'recording_head/mps/slam/semidense_points.csv.gz'
        source = Path(self.old['checkpoint_contract']['nymeria_uniego_root']) / (uuid + '.npz')
        key = hashlib.sha256(uuid.encode()).hexdigest()[:24]
        cache = self.operation / 'scene_cache' / key
        cache.mkdir(parents=True, exist_ok=True)
        with (cache / 'prepare.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not (cache / 'complete.json').exists():
                pieces = []
                raw_count = 0
                for chunk in pd.read_csv(cloud, usecols=['graph_uid', 'px_world', 'py_world', 'pz_world',
                        'inv_dist_std', 'dist_std'], chunksize=200000):
                    raw_count += len(chunk)
                    assert (chunk.graph_uid.to_numpy() == graph).all(), uuid
                    inv = chunk.inv_dist_std.to_numpy().astype(np.float32).astype(np.float64)
                    dist = chunk.dist_std.to_numpy().astype(np.float32).astype(np.float64)
                    mask = (inv < .0004) & (dist < .02)
                    xyz = chunk.loc[mask, ['px_world', 'py_world', 'pz_world']].to_numpy(dtype=np.float64) @ AXES.T
                    pieces.append(xyz[np.isfinite(xyz).all(axis=1)])
                points = np.concatenate(pieces)
                assert len(points) == record['coverage']['confident_point_count'], uuid
                assert raw_count == record['coverage']['raw_point_count']
                np.save(cache / 'world_points.npy', points, allow_pickle=False)
                with np.load(source) as z:
                    features = z['features'].astype(np.float32)
                rotations = cont6d_to_matrix(features[:, 270:276])
                origins = features[:, 276:279].astype(np.float64)
                for i in range(1, len(origins)):
                    origins[i] = origins[i-1] + rotations[i-1] @ origins[i]
                    rotations[i] = rotations[i-1] @ rotations[i]
                np.save(cache / 'canonical_rotations.npy', rotations, allow_pickle=False)
                np.save(cache / 'canonical_origins.npy', origins, allow_pickle=False)
                save(cache / 'complete.json', {'uuid': uuid, 'graph_uid': graph,
                    'source_motion_sha256': sha(source), 'source_cloud_size': cloud.stat().st_size,
                    'source_cloud_mtime_ns': cloud.stat().st_mtime_ns, 'raw_points': raw_count,
                    'approved_points': len(points), 'world_basis': 'Kimodo Y-up',
                    'point_count_sampling': False})
        metadata = json.loads((cache / 'complete.json').read_text())
        assert metadata['uuid'] == uuid
        assert cloud.stat().st_size == metadata['source_cloud_size'] and cloud.stat().st_mtime_ns == metadata['source_cloud_mtime_ns']
        with np.load(source) as z:
            features = z['features'].astype(np.float32)
        assert sha(source) == metadata['source_motion_sha256']
        points = np.load(cache / 'world_points.npy', mmap_mode='r')
        participant = json.loads((Path(self.gt_cfg['cloud_root']) / name / 'metadata.json').read_text())
        height = float(participant['participant_height_cm']) / 100
        assert .9 < height < 2.5
        radii = np.asarray([c[3] for c in CAPSULES]) * height / 1.75
        camera_path = Path(self.gt_cfg['data_root']) / 'camera_rgb' / (uuid + '.npz')
        with np.load(camera_path) as z:
            camera_positions = z['cam_world_pos_upright'].astype(np.float64) @ AXES.T
        self.current = {'uuid': uuid, 'points': points, 'tree': cKDTree(points),
            'rotations': np.load(cache / 'canonical_rotations.npy', mmap_mode='r'),
            'origins': np.load(cache / 'canonical_origins.npy', mmap_mode='r'),
            'features': features, 'radii': radii, 'camera_positions': camera_positions}
        return self.current

    def measure(self, row, predictions):
        context = self.recording(row['uuid'])
        start = int(row['start_frame']); end = start + 97
        feature_path = Path(self.reference['data_dir']) / row['features_file']
        assert sha(feature_path) == row['features_sha256']
        features = np.load(feature_path).astype(np.float32)
        assert np.array_equal(features, context['features'][start:end]), row['case_id']
        # The benchmark stores raw slices. Only GT's frame-zero delta is made
        # canonical; predictions retain their complete original sampled features.
        gt = canonicalize_frame0(ground_features(features, row['floor_offset']))
        decoded = [decode_uniego(gt)] + [decode_uniego(p) for p in predictions]
        joints = np.stack([d.world_positions for d in decoded])
        rotation = context['rotations'][start]
        origin = context['origins'][start]
        offset = np.array([0., float(row['floor_offset']), 0.])
        camera = (context['camera_positions'][start:end] - origin - offset) @ rotation
        separation = np.linalg.norm(camera - joints[0, :, 6], axis=1)
        # Anatomical separation can contain source-tracking outliers even when
        # the world transform is correct. Preserve the immutable benchmark and
        # flag these frames; never reject or fit poses to the camera here.
        source_outlier = (separation <= .05) | (separation >= .35)
        roundtrip = 0.
        if source_outlier.any():
            independent = decode_uniego(context['features']).world_positions[start:end]
            restored = joints[0] @ rotation.T + origin + offset
            roundtrip = float(np.abs(independent-restored).max())
            if roundtrip > 1e-4:
                raise RuntimeError('Source/world window transform mismatch: ' + row['case_id'])
        active_joints = sorted({j for name, a, b, _ in CAPSULES if name not in EXCLUDED_COLLISION_CAPSULES for j in [a, b]})
        relevant = joints[:, :, active_joints]
        low, high = relevant.min(axis=(0, 1, 2)) - 1.5, relevant.max(axis=(0, 1, 2)) + 1.5
        # The crop covers GT AND every generated trajectory. No condition can
        # escape scoring by leaving a GT-only neighborhood.
        center = ((low + high)/2) @ rotation.T + origin + offset
        radius = float(np.linalg.norm(high-low)/2)
        candidates = context['tree'].query_ball_point(center, radius, return_sorted=True)
        points = (context['points'][candidates] - origin - offset) @ rotation
        points = points[(points >= low).all(axis=1) & (points <= high).all(axis=1)]
        if not len(points):
            raise RuntimeError('No observed map support in common comparison region')
        keys = np.floor(points / .02).astype(np.int64)
        _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        voxels = np.stack([np.bincount(inverse, weights=points[:, i])/counts for i in range(3)], axis=1).astype(np.float32)
        results = {}
        evidence = {}
        evidence['GT__source_head_camera_outlier'] = source_outlier
        for label, motion in zip(['GT', 'image_only', 'text_only', 'image_text'], decoded):
            metrics, ids, hits, depths = score(voxels, motion.world_positions, context['radii'])
            results[label] = metrics
            evidence[label+'__frame_hits'] = hits.any(axis=1)
            evidence[label+'__body_hits'] = hits
            evidence[label+'__depths'] = np.concatenate(depths)
            evidence[label+'__intruding_voxel_counts'] = np.asarray([len(x) for x in ids])
        return {'case_id': row['case_id'], 'metrics': results,
                'coverage': {'raw_approved_points': len(context['points']), 'common_metric_voxels': len(voxels),
                    'raw_points_in_shared_region': len(points), 'shared_roi_low': low.tolist(), 'shared_roi_high': high.tolist(),
                    'gt_head_camera_min_m': float(separation.min()), 'gt_head_camera_max_m': float(separation.max()),
                    'gt_head_camera_outlier_frames': int(source_outlier.sum()),
                    'flagged_source_world_roundtrip_m': roundtrip},
                'evidence': evidence}
