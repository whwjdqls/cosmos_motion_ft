"""GT UniEgo283 scene intrusion diagnostics, without scene reconstruction.

All data computation requires an owned compute allocation. Paths are explicit
so the same evaluator can be used on Yonsei and Grasp.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'motion_expert_joint_attention'))
from camera_head_recanonicalization import (  # noqa: E402
    ARIA_Z_UP_TO_KIMODO_Y_UP as AXES, cont6d_to_matrix, decode_uniego,
)
from uniego_layout import canonicalize_frame0, ground_features  # noqa: E402

PARENTS = [-1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 3, 10, 11, 12, 13, 13, 3, 16, 17, 18, 19, 19,
           0, 22, 23, 24, 0, 26, 27, 28]
# (name, parent joint, child joint, radius at a 1.75m subject height).
# Body only: retain wrists; omit eyes, jaw and articulated fingertips.
CAPSULES = [
    ('lower_spine', 0, 1, .105), ('mid_spine', 1, 2, .120),
    ('upper_spine', 2, 3, .130), ('neck_lower', 3, 4, .055),
    ('neck_upper', 4, 5, .045), ('head', 6, 6, .090),
    ('left_clavicle', 3, 10, .045), ('left_shoulder', 10, 11, .055),
    ('left_upper_arm', 11, 12, .055), ('left_forearm', 12, 13, .040),
    ('right_clavicle', 3, 16, .045), ('right_shoulder', 16, 17, .055),
    ('right_upper_arm', 17, 18, .055), ('right_forearm', 18, 19, .040),
    ('left_hip', 0, 22, .075), ('left_thigh', 22, 23, .075),
    ('left_shin', 23, 24, .050), ('left_foot', 24, 25, .035),
    ('right_hip', 0, 26, .075), ('right_thigh', 26, 27, .075),
    ('right_shin', 27, 28, .050), ('right_foot', 28, 29, .035),
]
EXCLUDED_COLLISION_CAPSULES = (
    'left_thigh', 'right_thigh', 'left_shin', 'right_shin', 'left_foot', 'right_foot',
)


def collision_exclusions(body_policy='exclude_thigh_shin_foot'):
    if body_policy == 'exclude_thigh_shin_foot':
        return EXCLUDED_COLLISION_CAPSULES
    if body_policy == 'full_body':
        return ()
    raise ValueError('Unknown collision body policy: ' + body_policy)


VOXEL = .02
EPSILONS = [.01, .02, .03]
SCALES = [.8, 1.0, 1.2]
MIN_CLUSTER = 3
CLOCK_SOURCES = None


def fetch_head_clock(args, name):
    global CLOCK_SOURCES
    if not args.clock_manifests or not args.notion_helper:
        raise RuntimeError('No original head motion.vrs for measured clock conversion')
    if CLOCK_SOURCES is None:
        sources = {}
        for path in args.clock_manifests:
            data = json.loads(path.read_text())
            for sequence, record in data['sequences'].items():
                if 'timesync_and_imu' in record:
                    sources.setdefault(sequence, []).append((str(path), record['timesync_and_imu']))
        CLOCK_SOURCES = sources
    spec = importlib.util.spec_from_file_location('clock_download_helper', args.notion_helper)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    old = helper.load_prior_module()
    reader_class = helper.paced_reader_class(old.RangeFile)
    reader_class.block_size = int(getattr(args, 'clock_range_mib', 8)) * 1024 * 1024
    relative = 'recording_head/data/motion.vrs'
    root = args.out / 'head_sync' / 'recordings' / name
    target = root / relative
    receipt_path = root / 'head_clock_receipt.json'
    if target.is_file() and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('status') == 'complete' and target.stat().st_size == receipt['bytes']:
            return target
    if target.exists():
        raise RuntimeError('Unreceipted head clock file; refusing overwrite')
    target.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for manifest, source in CLOCK_SOURCES.get(name, []):
        try:
            with reader_class(source['download_url'], source['file_size_bytes']) as remote:
                with zipfile.ZipFile(remote) as archive:
                    found = [x for x in archive.infolist()
                             if x.filename == relative or x.filename.endswith('/' + relative)]
                    if len(found) != 1:
                        raise RuntimeError('Expected exactly one head clock VRS archive member')
                    member = found[0]
                    log('clock_download_start', recording=name, compressed_bytes=member.compress_size)
                    partial = target.with_name(target.name + '.part')
                    digest, count = hashlib.sha256(), 0
                    with archive.open(member) as input_file, partial.open('wb') as output:
                        while block := input_file.read(1024 * 1024):
                            output.write(block); digest.update(block); count += len(block)
                    if count != member.file_size:
                        raise RuntimeError('Head clock ZIP member size mismatch')
                    os.replace(partial, target)
                    save(receipt_path, {'status': 'complete', 'bytes': count,
                        'sha256': digest.hexdigest(), 'zip_crc_verified': True,
                        'zip_crc32': f'{member.CRC:08x}', 'manifest': manifest,
                        'archive': source['filename'], 'full_archive_sha1_verified': False})
                    log('clock_download_complete', recording=name, bytes=count)
                    return target
        except Exception as error:
            errors.append(str(error) if isinstance(error, RuntimeError) else type(error).__name__)
    raise RuntimeError('Head clock download failed: ' + json.dumps(errors))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def log(event, **values):
    print(json.dumps({'time': time.time(), 'event': event, **values}), flush=True)


def clock_times(args, name, times):
    candidates = [p / 'recordings' / name / 'recording_head/data/motion.vrs'
                  for p in args.raw_roots]
    found = [p for p in candidates if p.is_file()]
    if not found:
        found = [fetch_head_clock(args, name)]
    answer = subprocess.run(
        [str(args.aria_python), str(Path(__file__).with_name('scene_timecode.py')), str(found[0])],
        input=json.dumps({'timecode_ns': [int(t) * 1000 for t in times]}),
        text=True, capture_output=True, check=False, timeout=120,
        env={**os.environ, 'PYTHONPATH': ''})
    tagged = [line for line in answer.stdout.splitlines() if line.startswith('SCENE_CLOCK_JSON=')]
    if answer.returncode or len(tagged) != 1:
        raise RuntimeError('Headset clock conversion failed: ' + answer.stderr[-1200:])
    device = np.asarray(json.loads(tagged[0].split('=', 1)[1])['device_ns'], dtype=np.int64) / 1000.
    if device.shape != times.shape or np.any(np.diff(device) <= 0):
        raise RuntimeError('Converted device timestamps are not a valid increasing window')
    return device, str(found[0])


def check_scene_frame(path, device_times, camera_positions, camera_rotations):
    cols = ['graph_uid', 'tracking_timestamp_us', 'tx_world_device', 'ty_world_device',
            'tz_world_device', 'qx_world_device', 'qy_world_device', 'qz_world_device', 'qw_world_device']
    parts = []
    for chunk in pd.read_csv(path, usecols=cols, chunksize=100000):
        t = chunk.tracking_timestamp_us.to_numpy()
        mask = (t >= device_times[0] - 100000) & (t <= device_times[-1] + 100000)
        if mask.any():
            parts.append(chunk.loc[mask])
        if t[-1] > device_times[-1] + 100000:
            break
    if not parts:
        raise RuntimeError('Converted times are outside the SLAM trajectory')
    data = pd.concat(parts, ignore_index=True)
    t = data.tracking_timestamp_us.to_numpy()
    if np.any(np.diff(t) <= 0):
        raise RuntimeError('Non-increasing SLAM trajectory in the requested interval')
    right = np.clip(np.searchsorted(t, device_times), 0, len(t) - 1)
    left = np.maximum(0, right - 1)
    idx = np.where(abs(t[left] - device_times) <= abs(t[right] - device_times), left, right)
    sample = data.iloc[idx]
    graphs = sample.graph_uid.unique().tolist()
    if len(graphs) != 1:
        raise RuntimeError('Evaluation window crosses SLAM world graphs')
    positions = sample[['tx_world_device', 'ty_world_device', 'tz_world_device']].to_numpy()
    rotations = Rotation.from_quat(sample[['qx_world_device', 'qy_world_device', 'qz_world_device',
                                          'qw_world_device']].to_numpy()).as_matrix()
    position_error = np.linalg.norm(positions - camera_positions, axis=-1)
    angle_error = np.rad2deg(Rotation.from_matrix(rotations @ camera_rotations.swapaxes(-1, -2)).magnitude())
    time_error = abs(t[idx] - device_times)
    report = {'graph_uid': graphs[0], 'nearest_time_error_us_max': float(time_error.max()),
              'camera_translation_error_m_max': float(position_error.max()),
              'camera_rotation_error_deg_max': float(angle_error.max())}
    if time_error.max() > 25000 or position_error.max() > .02 or angle_error.max() > 2.0:
        raise RuntimeError('Scene and saved camera poses disagree: ' + json.dumps(report))
    return report


def prepare_motion(args, case, mean, std):
    start = int(case['start'])
    with np.load(case['motion_source'], allow_pickle=False) as source:
        full = source['features'].astype(np.float32)
        neutral = source['neutral_joints'].astype(np.float32)
        timestamps = source['timestamps_us'][start:start + 97]
    raw = full[start:start + 97].copy()
    if raw.shape != (97, 283) or not np.isfinite(raw).all():
        raise RuntimeError('Missing or invalid fixed motion window')
    off = float(case['floor_offset'])
    expected = (canonicalize_frame0(ground_features(raw, off)) - mean) / std
    with np.load(case['bundle'], allow_pickle=False) as bundle:
        if str(bundle['motion_mean_sha256']) != args.mean_sha or str(bundle['motion_std_sha256']) != args.std_sha:
            raise RuntimeError('GT bundle uses different normalization statistics')
        normalized = bundle['motion'].astype(np.float32)
    if not np.array_equal(normalized, expected):
        raise RuntimeError('GT bundle no longer matches its exact source processing')
    canonical = decode_uniego(normalized * std + mean)
    prefix = full[:start + 1, 270:279]
    delta_rot = cont6d_to_matrix(prefix[:, :6])
    delta_pos = prefix[:, 6:].astype(np.float64)
    rotation, position = delta_rot[0].copy(), delta_pos[0].copy()
    for i in range(1, len(prefix)):
        position = position + rotation @ delta_pos[i]
        rotation = rotation @ delta_rot[i]
    # Replace the slice's first residual with identity, then restore the actual
    # cumulative frame at start. This handles starts beyond frame zero correctly.
    source_relative = decode_uniego(canonicalize_frame0(raw))
    original_world = source_relative.world_positions @ rotation.T + position
    reconstructed = canonical.world_positions @ rotation.T + position + [0, off, 0]
    roundtrip = float(np.abs(reconstructed - original_world).max())
    if roundtrip > 1e-4:
        raise RuntimeError(f'Canonical/world round trip differs by {roundtrip}m')
    return canonical, neutral, timestamps, rotation, position, off, roundtrip


def read_scene(path, graph, rotation, position, off, joints, max_inv_std=.005, max_dist_std=.01,
               *, strict_confidence=False, return_display_points=False):
    low = joints.min(axis=(0, 1)) - 1.5
    high = joints.max(axis=(0, 1)) + 1.5
    selected, display_chunks, total, graph_count, confidence_count = [], [], 0, 0, 0
    cols = ['graph_uid', 'px_world', 'py_world', 'pz_world', 'inv_dist_std', 'dist_std']
    for chunk in pd.read_csv(path, usecols=cols, chunksize=200000):
        total += len(chunk)
        same = chunk.graph_uid.to_numpy() == graph
        graph_count += int(same.sum())
        inv, dist = [chunk[k].to_numpy() for k in ['inv_dist_std', 'dist_std']]
        if strict_confidence:
            # Aria GlobalPointPosition stores these as float32, then exposes
            # Python floats. Promote after rounding so .0004 boundaries match
            # the SDK's strict comparison, rather than NumPy scalar coercion.
            inv, dist = [x.astype(np.float32).astype(np.float64) for x in [inv, dist]]
        compare = np.less if strict_confidence else np.less_equal
        mask = same & compare(inv, max_inv_std) & compare(dist, max_dist_std)
        confidence_count += int(mask.sum())
        points = chunk.loc[mask, ['px_world', 'py_world', 'pz_world']].to_numpy(dtype=np.float64)
        points = (points @ AXES.T - position - [0, off, 0]) @ rotation
        if return_display_points:
            display_chunks.append(points[np.isfinite(points).all(axis=-1)].astype(np.float32))
        keep = np.isfinite(points).all(axis=-1) & (points >= low).all(axis=-1) & (points <= high).all(axis=-1)
        if keep.any():
            selected.append(points[keep])
    if not selected:
        raise RuntimeError('No confident scene points in the GT motion neighborhood')
    points = np.concatenate(selected)
    keys = np.floor(points / VOXEL).astype(np.int64)
    unique, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    centers = np.stack([np.bincount(inverse, weights=points[:, k]) / counts for k in range(3)], axis=-1)
    if len(centers) < 50:
        raise RuntimeError('Insufficient observed scene support after confidence filtering')
    coverage = {
        'raw_point_count': total, 'matching_graph_count': graph_count,
        'confident_point_count': confidence_count, 'crop_point_count': len(points),
        'scene_voxels': len(centers), 'voxel_m': VOXEL,
        'inv_dist_std_max': max_inv_std, 'dist_std_max_m': max_dist_std,
        'confidence_comparison': '<' if strict_confidence else '<=',
        'confidence_storage': 'SDK float32 promoted to Python float' if strict_confidence else 'CSV float64',
        'roi_canonical_low': low.tolist(), 'roi_canonical_high': high.tolist()}
    result = (centers.astype(np.float32), coverage)
    if return_display_points:
        # Raw retained positions for viewer parity; scoring keeps voxel centroids.
        return (*result, np.concatenate(display_chunks))
    return result


def segment_distance(points, a, b):
    axis = b - a
    length2 = float(axis @ axis)
    u = np.zeros(len(points)) if length2 < 1e-12 else np.clip((points - a) @ axis / length2, 0, 1)
    return np.linalg.norm(points - (a + u[:, None] * axis), axis=-1)


def supported_clusters(points, indices):
    if len(indices) < MIN_CLUSTER:
        return np.empty(0, dtype=np.int64)
    pairs = cKDTree(points[indices]).query_pairs(VOXEL * math.sqrt(3) * 1.05, output_type='ndarray')
    parent = np.arange(len(indices))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in pairs:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[rb] = ra
    roots = np.array([root(i) for i in range(len(indices))])
    counts = np.bincount(roots, minlength=len(indices))
    return indices[counts[roots] >= MIN_CLUSTER]


def score(points, joints, radii, *, excluded_capsules=EXCLUDED_COLLISION_CAPSULES):
    excluded = set(excluded_capsules)
    names = [cap[0] for cap in CAPSULES]
    if excluded - set(names):
        raise ValueError('Unknown excluded collision capsules')
    active = [i for i, name in enumerate(names) if name not in excluded]
    if not active:
        raise ValueError('No collision capsules selected')
    tree = cKDTree(points)
    sweeps = {(scale, epsilon): [] for scale in SCALES for epsilon in EPSILONS}
    primary_points, part_hits, primary_depths, nearby_counts = [], [], [], []
    for frame in joints:
        depths = np.full((len(SCALES), len(points)), -np.inf, dtype=np.float32)
        bone_candidates = [None] * len(CAPSULES)
        nearby = set()
        for bi in active:
            _, ja, jb, _ = CAPSULES[bi]
            a, b = frame[ja], frame[jb]
            ids = np.asarray(tree.query_ball_point((a + b) / 2,
                             np.linalg.norm(b - a) / 2 + radii[bi] * max(SCALES) + .25), dtype=np.int64)
            dist = segment_distance(points[ids], a, b)
            nearby.update(ids[dist <= radii[bi] + .25].tolist())
            for si, scale in enumerate(SCALES):
                depths[si, ids] = np.maximum(depths[si, ids], radii[bi] * scale - dist)
            bone_candidates[bi] = (ids, radii[bi] - dist)
        primary = np.empty(0, dtype=np.int64)
        for si, scale in enumerate(SCALES):
            for epsilon in EPSILONS:
                selected = supported_clusters(points, np.flatnonzero(depths[si] > epsilon))
                sweeps[scale, epsilon].append(bool(len(selected)))
                if scale == 1.0 and epsilon == .02:
                    primary = selected
        primary_points.append(primary)
        primary_depths.append(depths[1, primary].copy())
        hit_mask = np.zeros(len(points), dtype=bool)
        hit_mask[primary] = True
        part_hits.append([bool(np.any(hit_mask[item[0]] & (item[1] > .02))) if item is not None else False
                          for item in bone_candidates])
        nearby_counts.append(len(nearby))
    hits = np.asarray(part_hits, dtype=bool)
    packed = np.concatenate(primary_depths) if any(len(v) for v in primary_depths) else np.empty(0)
    frames = np.array([bool(len(v)) for v in primary_points])
    metrics = {'collision_frame_percent': float(frames.mean() * 100),
        'collision_frames': int(frames.sum()), 'frames': len(joints),
        'colliding_body_part_frame_percent': float(hits[:, active].mean() * 100),
        'active_capsule_names': [names[i] for i in active],
        'excluded_capsule_names': [name for name in names if name in excluded],
        'intrusion_depth_mean_cm': float(packed.mean() * 100) if len(packed) else 0.0,
        'intrusion_depth_p95_cm': float(np.percentile(packed, 95) * 100) if len(packed) else 0.0,
        'intruding_voxel_frame_count': len(packed),
        'near_body_scene_voxels_min': int(min(nearby_counts)),
        'near_body_scene_voxels_median': float(np.median(nearby_counts)),
        'body_parts': {names[i]: float(hits[:, i].mean() * 100) for i in active},
        'sensitivity': [{'radius_scale': s, 'tolerance_cm': e * 100,
                         'collision_frame_percent': float(np.mean(value) * 100)}
                        for (s, e), value in sweeps.items()]}
    return metrics, primary_points, hits, primary_depths


def geometry_acceptance():
    a, b = np.array([0., 0., 0.]), np.array([0., 1., 0.])
    points = np.array([[0, .4, 0], [.09, .4, 0], [.2, .4, 0]])
    depth = .1 - segment_distance(points, a, b)
    if not np.array_equal(depth > .02, [True, False, False]):
        raise RuntimeError('Known interior/contact/exterior geometry check failed')
    p = np.array([[0, 0, 0], [.02, 0, 0], [.04, 0, 0], [2, 0, 0]], dtype=np.float32)
    got = supported_clusters(p, np.arange(4))
    if not np.array_equal(got, [0, 1, 2]):
        raise RuntimeError('Independent voxel cluster check failed')
    return {'analytic_interior_contact_exterior': 'passed', 'three_voxel_cluster': 'passed'}


def capsule_wire(a, b, radius):
    axis = b - a
    length = np.linalg.norm(axis)
    axis = axis / length if length > 1e-8 else np.array([0., 1., 0.])
    helper = np.array([1., 0., 0.]) if abs(axis[0]) < .8 else np.array([0., 0., 1.])
    u = np.cross(axis, helper); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    angle = np.linspace(0, 2 * np.pi, 17)
    rings = []
    for z, r in [(-radius, 0), (-radius * .707, radius * .707), (0, radius),
                 (length, radius), (length + radius * .707, radius * .707), (length + radius, 0)]:
        rings.append(a + z * axis + r * (np.cos(angle)[:, None] * u + np.sin(angle)[:, None] * v))
    lines = [rings[2], rings[3]]
    lines.extend(np.stack([ring[k] for ring in rings]) for k in [0, 4, 8, 12])
    return lines


def render(args, case, arrays):
    if getattr(args, 'renderer', 'matplotlib') == 'opengl':
        from nymeria_baselines.scene_gl_renderer import render as render_gl
        return render_gl(args, case, arrays)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from matplotlib.animation import FFMpegWriter
    points, joints, hits, radii = arrays['points'], arrays['joints'], arrays['hits'], arrays['radii']
    ids = arrays['point_ids']
    with np.load(case['bundle'], allow_pickle=False) as bundle:
        rgb = bundle['rgb']
    # A proper rotation, determinant +1. The former (X,Z,Y) display was a
    # reflection and reversed perceived turn handedness in both scene/body.
    display = lambda p: np.asarray(p)[..., [0, 2, 1]] * [1., -1., 1.]
    display_points = arrays.get('display_points', points)
    point_limit = getattr(args, 'display_point_limit', None)
    shown = display_points
    if point_limit is not None:
        shown = shown[np.random.default_rng(0).choice(len(shown), min(int(point_limit), len(shown)), replace=False)]
    low = joints.min(axis=(0, 1)) - [.75, .25, .75]
    high = joints.max(axis=(0, 1)) + [.75, .35, .75]
    if high[0] - low[0] < 2.5:
        mid = (high[0] + low[0]) / 2; low[0], high[0] = mid - 1.25, mid + 1.25
    if high[2] - low[2] < 2.5:
        mid = (high[2] + low[2]) / 2; low[2], high[2] = mid - 1.25, mid + 1.25
    fig = plt.figure(figsize=(16, 10), dpi=100, facecolor='white')
    ax_rgb = fig.add_subplot(221)
    views = [fig.add_subplot(222, projection='3d'), fig.add_subplot(223, projection='3d')]
    top = fig.add_subplot(224)
    image = ax_rgb.imshow(rgb[0]); ax_rgb.set_title('Ground-truth RGB'); ax_rgb.axis('off')
    artists = []
    for ax, angle in zip(views, [-65, 25]):
        p = display(shown)
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=.9, c='#808080', alpha=.5, depthshade=False)
        ax.set(xlim=(low[0], high[0]), ylim=(-high[2], -low[2]), zlim=(low[1], high[1]),
               xlabel='X (m)', ylabel='-Z (m)', zlabel='Y / height (m)')
        ax.set_box_aspect((high - low)[[0, 2, 1]])
        ax.view_init(elev=18, azim=angle)
        ax.set_title('GT body and scene; red: provisional capsule overlap')
        bones = Line3DCollection([np.zeros((2, 3))], linewidths=1.6)
        wires = Line3DCollection([np.zeros((2, 3))], linewidths=.35, alpha=.3)
        ax.add_collection3d(wires, autolim=False); ax.add_collection3d(bones, autolim=False)
        scatter = ax.scatter([], [], [], c='#e64231', s=17, depthshade=False)
        origin = display(joints[0, 6])
        for color, direction in zip(['#ca3d32', '#32964d', '#2b64bd'], arrays['head_rotation0'].T):
            direction = display(direction) * .18
            ax.quiver(*origin, *direction, color=color, linewidth=1.2)
        artists.append((bones, wires, scatter))
    shown_display = display(shown)
    top.scatter(shown_display[:, 0], shown_display[:, 1], s=.9, c='#808080', alpha=.5)
    top.set(xlim=(low[0], high[0]), ylim=(-high[2], -low[2]),
            xlabel='X (m)', ylabel='-Z (m)', title='Top view: body position and forward directions')
    top.set_aspect('equal', adjustable='box')
    top_bones = LineCollection([np.zeros((2, 2))], linewidths=1.6)
    top.add_collection(top_bones)
    top_points = top.scatter([], [], c='#e64231', s=10)
    top.plot([], [], color='#c18b18', label='Camera optical forward')
    top.plot([], [], color='#218650', label='Head-derived camera forward')
    top.plot([], [], color='#735a9a', label='Torso forward')
    top.legend(loc='upper right', fontsize=8)
    direction_artists = []
    fig.subplots_adjust(left=.025, right=.975, bottom=.065, top=.95, wspace=.09, hspace=.2)
    output = args.out / 'visualizations'
    output.mkdir(exist_ok=True)
    def update(frame):
        image.set_data(rgb[frame])
        edges, colors, wires = [], [], []
        for bi, (_, ja, jb, _) in enumerate(CAPSULES):
            edges.append(display(joints[frame, [ja, jb]]))
            colors.append('#e64231' if hits[frame, bi] else '#226ba1')
            wires.extend(display(line) for line in capsule_wire(joints[frame, ja], joints[frame, jb], radii[bi]))
        intrusions = display(points[ids[frame]])
        for bones, wire, scatter in artists:
            bones.set_segments(edges); bones.set_color(colors)
            wire.set_segments(wires); wire.set_color('#377b9c')
            scatter._offsets3d = (intrusions[:, 0], intrusions[:, 1], intrusions[:, 2])
        top_bones.set_segments([e[:, :2] for e in edges]); top_bones.set_color(colors)
        top_points.set_offsets(intrusions[:, :2])
        for old in direction_artists:
            old.remove()
        direction_artists.clear()
        if 'camera_positions' in arrays:
            directions = [
                (arrays['camera_positions'][frame], arrays['camera_forward'][frame], '#c18b18'),
                (joints[frame, 6], arrays['head_camera_forward'][frame], '#218650'),
                (joints[frame, 0], arrays['torso_forward'][frame], '#735a9a')]
            for origin3, direction3, color in directions:
                origin3, direction3 = display(origin3), display(direction3)
                direction_artists.append(top.quiver(origin3[0], origin3[1], direction3[0] * .4,
                    direction3[1] * .4, angles='xy', scale_units='xy', scale=1, color=color, width=.006))
                for ax in views:
                    direction_artists.append(ax.quiver(*origin3, *(direction3 * .35), color=color,
                                                       linewidth=1.8, arrow_length_ratio=.18))
    worst = int(np.argmax([len(v) for v in ids]))
    update(0)
    png = output / (case['id'] + '.png')
    fig.savefig(png, dpi=130)
    update(worst)
    fig.savefig(output / (case['id'] + '_worst.png'), dpi=130)
    executable = shutil.which('ffmpeg') or str(Path(sys.executable).with_name('ffmpeg'))
    if not Path(executable).is_file():
        import imageio_ffmpeg
        executable = imageio_ffmpeg.get_ffmpeg_exe()
    matplotlib.rcParams['animation.ffmpeg_path'] = executable
    mp4 = output / (case['id'] + '.mp4')
    writer = FFMpegWriter(fps=20, codec='libx264', extra_args=['-crf', '20', '-pix_fmt', 'yuv420p', '-threads', '1'])
    with writer.saving(fig, str(mp4), 100):
        for frame in range(97):
            update(frame); writer.grab_frame()
    plt.close(fig)
    return {'case': case['id'], 'png': str(png), 'mp4': str(mp4), 'snapshot_frame': 0,
            'worst_frame': worst, 'display_point_limit': point_limit, 'display_point_count': len(shown),
            'display_point_source': 'raw confidence-filtered positions' if 'display_points' in arrays else 'voxel centroids',
            'display_basis': '(X,-Z,Y), determinant +1; right handed',
            'selection': 'fixed cohort windows; primary snapshot is frame zero'}


class Publisher:
    def __init__(self, args):
        self.args = args
        self.helper = None
        if args.notion_helper:
            spec = importlib.util.spec_from_file_location('scene_download_api', args.notion_helper)
            self.helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.helper)
    def publish(self, summary, media):
        save(self.args.out / 'prepared_report.json', {'summary': summary, 'media': media})
        if self.helper is None:
            return
        try:
            path = self.args.out / 'publication.json'
            state = json.loads(path.read_text()) if path.exists() else None
            if state is None:
                page = self.helper.notion_request('POST', 'pages', {
                    'parent': {'type': 'page_id', 'page_id': '396d2a44-f44b-8053-b736-ceb63350f26b'},
                    'properties': {'title': {'type': 'title', 'title': [{'type': 'text', 'text': {
                        'content': 'GT Nymeria motion and scene collision'}}]}}})
                state = {'page_id': page['id'], 'url': page['url'], 'uploaded': {}}
                save(path, state)
            paragraphs = [
                'GT-only scene intrusion evaluation: ' + summary['state'] + '. '
                f"Scored {summary['completed']}/71, failed {summary['failed']}. "
                'Source: exact saved Phase3 held-out windows, camera/head UniEgo283 and its recorded mean/std.',
                'The scene and body share the same per-window yaw/translation and calibrated floor shift. '
                'Original head VRS converts synchronized timecode to device time; sampled SLAM poses must '
                'match saved camera poses within 2cm/2 degrees and 25ms. No fitted scene/body registration.',
                'Primary proxy: 2cm voxel cloud; inv_dist_std <=0.005 and dist_std <=0.01m; '
                'at least three neighboring voxel centers more than 2cm inside the union of body capsules. '
                'Capsules use the same subject height scaling, ignore face/finger joints and retain wrists. '
                'Sensitivity: 1/2/3cm margins, radius scales 0.8/1.0/1.2. No model collision results yet.',
                'Capsule intrusion depth is not distance inside the physical scene. Sparse observations can '
                'miss collisions. Capsule anatomy, accumulated static maps/dynamic objects and reconstructed '
                'motion errors can produce GT intersections. Floor contact is included in the raw proxy '
                'and cannot be assigned a floor/object category without additional scene analysis.',
                'Results and per-case arrays: ' + str(self.args.out),
            ]
            if summary.get('cloud_filter') == 'nymeria_legacy_sdk':
                paragraphs[2] = (
                    'Approved cloud: inverse-distance std <0.0004 and distance std <0.02m, '
                    'with SDK float32 confidence storage promoted to Python floats. No point-count '
                    'sampling or GT-overlap removal. The complete filtered cloud is represented by '
                    '2cm voxel centroids near the body for scoring, without display downsampling. '
                    'A collision-proxy frame requires at least three connected voxels more than '
                    '2cm inside the body-capsule union. Same subject-scaled body capsules and '
                    '1/2/3cm, radius0.8/1.0/1.2 sensitivities as the original diagnostic. '
                    'These are GT capsule/cloud intrusion measurements, not validated physical scene penetration.')
                if summary.get('collision_body') == 'exclude_thigh_shin_foot':
                    paragraphs[2] += (' Both thighs, shins and feet are excluded before computing the capsule union, '
                                      'cluster support, frame flags and depths. Hips remain included. '
                                      'The body-part/frame denominator is16capsules.')
            if summary.get('visualization_only'):
                paragraphs = [
                    'GT scene visualization review only. ' + summary['state'] + '. '
                    f"Rendered {summary['completed']} fixed cases. Collision scores remain unvalidated; bulk scoring is paused.",
                    summary['viewer_protocol'],
                    'Red markers show provisional body-capsule overlap, not established physical penetration. '
                    'No points were removed because of GT overlap. The display cap does not change metric geometry.',
                    'Artifacts: ' + str(self.args.out),
                ]
            if summary.get('aggregate'):
                paragraphs.append('Measured GT aggregate (equal window weighting): ' + json.dumps(summary['aggregate']))
            blocks = [{'object': 'block', 'type': 'paragraph', 'paragraph': {'rich_text': [
                {'type': 'text', 'text': {'content': s[:1950]}}]}} for s in paragraphs]
            result = self.helper.notion_request('PATCH', 'blocks/' + state['page_id'] + '/children', {'children': blocks})
            state['last_report_block_ids'] = [b['id'] for b in result['results']]
            save(path, state)
            for entry in media:
                for kind in ['png', 'mp4']:
                    source = Path(entry[kind]); digest = sha(source)
                    if digest in state['uploaded']:
                        continue
                    mime = 'image/png' if kind == 'png' else 'video/mp4'
                    upload = self.helper.notion_request('POST', 'file_uploads', {
                        'mode': 'single_part', 'filename': source.name, 'content_type': mime})
                    boundary = 'SceneCollisionUploadBoundary'
                    data = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                            f'filename="{source.name}"\r\nContent-Type: {mime}\r\n\r\n').encode()
                    data += source.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
                    token = Path('/home/whwjdqls99/.config/notion/token').read_text().strip()
                    req = urllib.request.Request('https://api.notion.com/v1/file_uploads/' + upload['id'] + '/send',
                        data=data, method='POST', headers={'Authorization': 'Bearer ' + token,
                        'Notion-Version': '2026-03-11', 'Content-Type': 'multipart/form-data; boundary=' + boundary})
                    with urllib.request.urlopen(req, timeout=120) as response:
                        sent = json.load(response)
                    if sent.get('status') != 'uploaded':
                        raise RuntimeError('Notion media upload incomplete')
                    block_type = 'image' if kind == 'png' else 'video'
                    ack = self.helper.notion_request('PATCH', 'blocks/' + state['page_id'] + '/children', {'children': [
                        {'object': 'block', 'type': block_type, block_type: {'type': 'file_upload',
                         'file_upload': {'id': upload['id']}, 'caption': [{'type': 'text', 'text': {
                             'content': entry['case'] + ' | GT RGB and two scene views; red intrusion evidence'}}]}}]})
                    block = ack['results'][0]
                    # Read the uploaded artifact returned by Notion, not the local source.
                    remote = block[block_type]
                    if remote.get('type') == 'file':
                        with urllib.request.urlopen(remote['file']['url'], timeout=120) as response:
                            remote_sha = hashlib.sha256(response.read()).hexdigest()
                        if remote_sha != digest:
                            raise RuntimeError('Remote media hash differs')
                        verified = True
                    else:
                        verified = False
                    state['uploaded'][digest] = {'path': str(source), 'block_id': block['id'],
                        'upload_acknowledged': True, 'remote_sha256_verified': verified}
                    save(path, state)
            state['state'] = summary['state']; save(path, state)
            log('published', url=state['url'], media=len(state['uploaded']))
        except Exception as exc:
            log('publication_failed', error=str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__)


def run_case(args, case, mean, std):
    name = case['uuid'].split('/')[-1]
    dest = args.out / 'cases' / case['id']; dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'metrics.json').exists():
        cached = json.loads((dest / 'metrics.json').read_text())
        if cached.get('cloud_filter', 'historical') != args.cloud_filter:
            raise RuntimeError('Cached case uses a different point-cloud filter; use a new output directory')
        if cached.get('collision_body', 'full_body') != args.collision_body:
            raise RuntimeError('Cached case uses a different collision body policy; use a new output directory')
        return cached
    canonical, neutral, times, rotation, position, off, roundtrip = prepare_motion(args, case, mean, std)
    camera_path = args.data_root / 'camera_rgb' / (case['uuid'] + '.npz')
    with np.load(camera_path, allow_pickle=False) as camera:
        start = int(case['start'])
        if not np.array_equal(times, camera['timestamps_us'][start:start + 97]):
            raise RuntimeError('Motion and saved camera timestamps disagree')
        cp = camera['cam_world_pos_device'][start:start + 97].astype(np.float64)
        cr = camera['cam_world_rot_device'][start:start + 97].astype(np.float64)
    device_times, clock_source = clock_times(args, name, times)
    slam = args.cloud_root / name / 'recording_head/mps/slam'
    alignment = check_scene_frame(slam / 'closed_loop_trajectory.csv', device_times, cp, cr)
    alignment.update({'clock_vrs': clock_source, 'canonical_world_roundtrip_max_m': roundtrip,
                      'floor_offset_m': off, 'canonical_rotation_world_from_window': rotation.tolist(),
                      'canonical_origin_world_yup': position.tolist()})
    metadata = json.loads((args.cloud_root / name / 'metadata.json').read_text())
    height = float(metadata.get('participant_height_cm', 0)) / 100
    if not .9 < height < 2.5:
        raise RuntimeError('Missing valid subject height for fixed capsule radii')
    radii = np.array([cap[3] for cap in CAPSULES]) * height / 1.75
    cloud_options = ({'max_inv_std': .0004, 'max_dist_std': .02, 'strict_confidence': True}
                     if args.cloud_filter == 'nymeria_legacy_sdk' else {})
    points, coverage = read_scene(slam / 'semidense_points.csv.gz', alignment['graph_uid'],
                                 rotation, position, off, canonical.world_positions, **cloud_options)
    metrics, point_ids, hits, depths = score(points, canonical.world_positions, radii,
        excluded_capsules=collision_exclusions(args.collision_body))
    lengths = np.array([len(v) for v in point_ids])
    offsets = np.r_[0, np.cumsum(lengths)]
    np.savez_compressed(dest / 'geometry.npz', points=points, joints=canonical.world_positions.astype(np.float32),
        head_rotation0=canonical.world_rotations[0, 6], radii=radii, hits=hits,
        point_ids=np.concatenate(point_ids), point_offsets=offsets,
        intrusion_depths=np.concatenate(depths), neutral=neutral,
        canonical_motion_normalized=(canonicalize_frame0(ground_features(
            np.load(case['motion_source'], allow_pickle=False)['features'][case['start']:case['start'] + 97], off)) - mean) / std)
    result = {'case': case['id'], 'uuid': case['uuid'], 'start': case['start'], 'model': 'GT',
              'cloud_filter': args.cloud_filter, 'physical_collision_validated': False,
              'collision_body': args.collision_body,
              'status': 'complete', 'alignment': alignment, 'coverage': coverage,
              'subject_height_m': height, 'capsule_radii_m': radii.tolist(), **metrics}
    save(dest / 'metrics.json', result)
    log('case_complete', case=case['id'], collision_frame_percent=metrics['collision_frame_percent'],
        scene_voxels=len(points), camera_error_m=alignment['camera_translation_error_m_max'])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['cohort', 'cloud-root', 'data-root', 'mean', 'std', 'aria-python', 'out']:
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--raw-root', dest='raw_roots', required=True, action='append', type=Path)
    p.add_argument('--notion-helper', type=Path)
    p.add_argument('--clock-manifest', dest='clock_manifests', action='append', type=Path, default=[])
    p.add_argument('--previews', type=int, default=4)
    p.add_argument('--cloud-filter', choices=['nymeria_legacy_sdk', 'historical'], default='nymeria_legacy_sdk')
    p.add_argument('--collision-body', choices=['exclude_thigh_shin_foot', 'full_body'], default='exclude_thigh_shin_foot')
    p.add_argument('--clock-workers', type=int, default=4)
    p.add_argument('--clock-range-mib', type=int, default=8)
    args = p.parse_args()
    host = socket.gethostname().split('.')[0]
    if 'login' in host.lower() or host in {'node37', 'cs-gpu-01'} or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Scene evaluation requires a permitted owned compute allocation')
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'configuration.json').exists():
        previous = json.loads((args.out / 'configuration.json').read_text())
        if previous['arguments'].get('cloud_filter', 'historical') != args.cloud_filter:
            raise RuntimeError('Output directory belongs to a different cloud filter; refusing to mix scores')
        if previous['arguments'].get('collision_body', 'full_body') != args.collision_body:
            raise RuntimeError('Output directory belongs to a different collision body policy')
    acceptance = geometry_acceptance(); save(args.out / 'geometry_acceptance.json', acceptance)
    args.mean_sha, args.std_sha = sha(args.mean), sha(args.std)
    mean, std = np.load(args.mean).astype(np.float32), np.load(args.std).astype(np.float32)
    cases = json.loads(args.cohort.read_text())['cases']
    if len(cases) != 71 or len({c['uuid'] for c in cases}) != 71:
        raise RuntimeError('Expected exactly the original 71 distinct fixed test windows')
    config = {'arguments': {k: ([str(x) for x in v] if isinstance(v, list) else str(v) if isinstance(v, Path) else v)
                            for k, v in vars(args).items()}, 'cohort_sha256': sha(args.cohort),
        'code_sha256': sha(__file__), 'capsules': CAPSULES, 'voxel_m': VOXEL,
        'margins_m': EPSILONS, 'radius_scales': SCALES, 'minimum_connected_voxels': MIN_CLUSTER,
        'frames': 97, 'fps': 20, 'model': 'GT', 'job': os.environ['SLURM_JOB_ID'], 'host': host}
    save(args.out / 'configuration.json', config)
    completed, failed, media = [], [], []
    pub = Publisher(args)
    def summary(state):
        value = {'state': state, 'model': 'GT', 'requested': 71,
                 'completed': len(completed), 'failed': len(failed), 'failures': failed,
                 'cloud_filter': args.cloud_filter, 'physical_collision_validated': False}
        value['collision_body'] = args.collision_body
        if completed:
            names = ['collision_frame_percent', 'colliding_body_part_frame_percent',
                     'intrusion_depth_mean_cm', 'intrusion_depth_p95_cm']
            value['aggregate'] = {k: float(np.mean([r[k] for r in completed])) for k in names}
            value['aggregate']['weighting'] = 'equal fixed window; depth summaries include zero-intrusion windows'
            value['aggregate']['collision_window_percent'] = float(np.mean([c['collision_frames'] > 0 for c in completed]) * 100)
            value['aggregate']['collision_frames'] = sum(c['collision_frames'] for c in completed)
            value['aggregate']['total_frames'] = sum(c['frames'] for c in completed)
            value['body_parts'] = {name: float(np.mean([c['body_parts'][name] for c in completed]))
                                   for name in completed[0]['body_parts']}
            value['sensitivity'] = [{**r, 'collision_frame_percent': float(np.mean([
                c['sensitivity'][i]['collision_frame_percent'] for c in completed]))}
                for i, r in enumerate(completed[0]['sensitivity'])]
        save(args.out / 'progress.json', value)
        return value
    pub.publish(summary('running'), media)
    # Fetch only missing headset clock members while scoring already-verified
    # recordings. Every case still waits for its own independent clock check.
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.clock_workers))
    clock_futures = {}
    for case in cases:
        name = case['uuid'].split('/')[-1]
        candidates = [root / 'recordings' / name / 'recording_head/data/motion.vrs' for root in args.raw_roots]
        if not any(path.is_file() for path in candidates):
            clock_futures[name] = pool.submit(fetch_head_clock, args, name)
    for index, case in enumerate(cases):
        try:
            name = case['uuid'].split('/')[-1]
            if name in clock_futures:
                clock_futures[name].result()
            result = run_case(args, case, mean, std); completed.append(result)
        except Exception as error:
            failure = {'case': case['id'], 'status': 'unscored', 'error': str(error),
                       'type': type(error).__name__}
            failed.append(failure); log('case_failed', **failure)
            save(args.out / 'cases' / case['id'] / 'failure.json', failure)
            continue
        summary('running')
        if index < args.previews:
            try:
                with np.load(args.out / 'cases' / case['id'] / 'geometry.npz') as z:
                    arrays = {k: z[k] for k in z.files}
                offsets = arrays['point_offsets']
                arrays['point_ids'] = [arrays['point_ids'][a:b] for a, b in zip(offsets[:-1], offsets[1:])]
                preview = render(args, case, arrays)
                media.append(preview); save(args.out / 'media.json', media)
                pub.publish(summary('running'), media)
            except Exception as error:
                log('preview_failed', case=case['id'], error=str(error))
                save(args.out / 'cases' / case['id'] / 'preview_failure.json', {'error': str(error)})
        if (index + 1) % 10 == 0:
            pub.publish(summary('running'), media)
    pool.shutdown(wait=True)
    state = 'complete' if len(completed) == 71 and len(media) >= min(args.previews, 71) else 'incomplete'
    final = summary(state); save(args.out / 'summary.json', final)
    save(args.out / 'per_case.json', completed)
    if completed:
        keys = ['case', 'model', 'collision_frame_percent', 'colliding_body_part_frame_percent',
                'intrusion_depth_mean_cm', 'intrusion_depth_p95_cm', 'near_body_scene_voxels_min']
        with (args.out / 'per_case.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore'); writer.writeheader(); writer.writerows(completed)
    pub.publish(final, media)
    log('finished', **final)
    raise SystemExit(0 if state == 'complete' else 1)


if __name__ == '__main__':
    main()
