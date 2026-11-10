"""Body23 position errors and DisCoRD jerk metrics, evaluated on decoded joints.

sJPE adapts the four HumanML3D end effectors to SOMA wrists and toe bases.
DisCoRD revision 782deffca1a263a2241b6321cfe6532a022f284e:
evaluation/eval_t2m.py calculate_finite_difference_jerk, calculate_area,
calculate_area_v2. Both native aggregations are retained explicitly. The
reported paper-style sJPE averages the native v1 result over its four joints.
"""
from __future__ import annotations

import numpy as np

BODY23_NAMES = (
    'Hips', 'Spine1', 'Spine2', 'Chest', 'Neck1', 'Neck2', 'Head',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
    'LeftLeg', 'LeftShin', 'LeftFoot', 'LeftToeBase',
    'RightLeg', 'RightShin', 'RightFoot', 'RightToeBase',
)
UPPER = tuple(range(1, 15))
LOWER = (0, *range(15, 23))
END_EFFECTORS = (18, 22, 10, 14)  # left/right toe, left/right wrist
DISCORD_REVISION = '782deffca1a263a2241b6321cfe6532a022f284e'
EPSILON = 1e-6


def protocol():
    return dict(schema='body23_position_discord_sjpe_v1', fps=20.,
                body23_joint_names=list(BODY23_NAMES), upper_joint_indices=list(UPPER),
                lower_joint_indices=list(LOWER), end_effector_indices=list(END_EFFECTORS),
                upper_joint_names=[BODY23_NAMES[i] for i in UPPER],
                lower_joint_names=[BODY23_NAMES[i] for i in LOWER],
                end_effector_names=[BODY23_NAMES[i] for i in END_EFFECTORS],
                position='Raw decoded canonical-world joints in meters; no new alignment or grounding. UPE14 upper joints; LPE9 pelvis+leg joints. These form a disjoint partition of all23 joints.',
                pa_mpjpe='Preserve existing whole-clip Umeyama similarity (scale, rotation, translation), not per-frame alignment.',
                root_ade='Mean absolute position error of Hips across all97 frames; existing root_err_m.',
                sjpe='Mean over94 valid third-difference frames and four end effectors of abs(predicted jerk magnitude - GT magnitude)/(predicted magnitude + GT magnitude + 1e-6). Dimensionless0..1; no extra factor2 or100.',
                sjpe_code='Exact calculate_area normalization sums four end effectors and divides only by94 frames; native joint-sum score equals4*reported paper-style mean. Native calculate_area_v2 ratio of sums is stored separately; it is not silently substituted.',
                jerk='Unaligned world positions; third forward difference * fps**3, then Euclidean magnitude. No smoothing, resampling, boundary padding, root centering or inpaint.',
                degeneracy='If both native v2 jerk sums are exactly zero, define its error as0 instead of upstream0/0; v1 retains native epsilon1e-6.',
                upstream_repository='https://github.com/whwjdqls/DisCoRD', upstream_commit=DISCORD_REVISION,
                upstream_path='evaluation/eval_t2m.py', mpjve_requested=False)


def jerk_magnitude(positions, fps=20.):
    p = np.asarray(positions)
    if p.ndim != 3 or p.shape[-1] != 3 or len(p) < 4 or not np.isfinite(p).all():
        raise ValueError('Finite [T>=4,J,3] positions required')
    jerk = (p[3:] - 3*p[2:-1] + 3*p[1:-2] - p[:-3]) * (fps**3)
    return np.linalg.norm(jerk, axis=-1)


def sjpe(prediction, target, fps=20.):
    jp = jerk_magnitude(prediction, fps)
    jg = jerk_magnitude(target, fps)
    assert jp.shape == jg.shape
    denominator = np.abs(jg) + np.abs(jp) + EPSILON
    static_raw = float(np.sum(np.maximum(0, jg-jp)/denominator)/len(jp))
    noise_raw = float(np.sum(np.maximum(0, jp-jg)/denominator)/len(jp))
    total = float(np.sum(jg+jp))
    static_v2 = float(np.sum(np.maximum(0, jg-jp))/total) if total else 0.
    noise_v2 = float(np.sum(np.maximum(0, jp-jg))/total) if total else 0.
    joints = jp.shape[1]
    return dict(sjpe=(static_raw+noise_raw)/joints,
                sjpe_static=static_raw/joints, sjpe_noise=noise_raw/joints,
                sjpe_discord_v1_jointsum=static_raw+noise_raw,
                sjpe_discord_v1_static_jointsum=static_raw,
                sjpe_discord_v1_noise_jointsum=noise_raw,
                sjpe_discord_v2_ratio_of_sums=static_v2+noise_v2,
                sjpe_discord_v2_static=static_v2, sjpe_discord_v2_noise=noise_v2,
                predicted_ee_jerk_m_s3=float(jp.mean()), gt_ee_jerk_m_s3=float(jg.mean()))


def extra_metrics(prediction, target, fps=20.):
    p, g = np.asarray(prediction), np.asarray(target)
    if p.shape != g.shape or p.ndim != 3 or p.shape[1:] != (23,3):
        raise ValueError(f'Matched [T,23,3] required: {p.shape}, {g.shape}')
    if not np.isfinite(p).all() or not np.isfinite(g).all():
        raise ValueError('Non-finite joints')
    distances = np.linalg.norm(p-g, axis=-1)
    values = dict(upe_m=float(distances[:,UPPER].mean()),
                  lpe_m=float(distances[:,LOWER].mean()))
    weighted = (len(UPPER)*values['upe_m']+len(LOWER)*values['lpe_m'])/23
    np.testing.assert_allclose(weighted, distances.mean(), rtol=2e-6, atol=1e-8)
    values.update(sjpe(p[:,END_EFFECTORS], g[:,END_EFFECTORS], fps))
    assert all(np.isfinite(v) for v in values.values())
    return values
