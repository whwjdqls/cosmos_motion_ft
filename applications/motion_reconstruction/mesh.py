"""SOMA standard-surface LBS preserving predicted global joint translations.

The stock 30-to77 visualization recomputes positions by standard-skeleton FK.
This path deliberately retains every controlled body joint's decoded position;
only unobserved face/finger/end bones follow the template's local bind offsets.
It is a visualization surface, not recovered subject-specific SOMA-X geometry.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np

BODY_NAMES = ('Hips','Spine1','Spine2','Chest','Neck1','Neck2','Head',
 'LeftShoulder','LeftArm','LeftForeArm','LeftHand',
 'RightShoulder','RightArm','RightForeArm','RightHand',
 'LeftLeg','LeftShin','LeftFoot','LeftToeBase','RightLeg','RightShin','RightFoot','RightToeBase')
BODY_IN30 = np.array([0,1,2,3,4,5,6,10,11,12,13,16,17,18,19,22,23,24,25,26,27,28,29])


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(4*1024*1024):h.update(b)
    return h.hexdigest()


@dataclass
class Skin:
    vertices: np.ndarray
    faces: np.ndarray
    bind: np.ndarray
    inverse_bind: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    parents: np.ndarray
    controls: np.ndarray
    local_bind: np.ndarray
    relaxed: np.ndarray
    vertex_joint_weights: np.ndarray
    source: str
    source_sha256: str


def load_skin(path):
    path=Path(path)
    with np.load(path,allow_pickle=False) as z:
        names=z['rig_joint_names'].tolist()
        vertices=z['bind_vertices'].astype(np.float64)
        faces=z['faces'].astype(np.int32)
        bind=z['bind_rig_transform'].astype(np.float64)
        indices=z['lbs_indices'].astype(np.int32)
        weights=z['lbs_weights'].astype(np.float64)
        edges=z['rig_joint_connections'].astype(int)
    parents=np.full(len(names),-1,dtype=int)
    for parent,child in edges:
        assert parents[child]==-1 and child>parent
        parents[child]=parent
    assert np.flatnonzero(parents<0).tolist()==[0]
    weights/=weights.sum(1,keepdims=True)
    assert np.isfinite(weights).all() and np.all(weights>=0)
    controls=np.array([names.index(name) for name in BODY_NAMES])
    local=bind.copy()
    for j,parent in enumerate(parents):
        if parent>=0:local[j]=np.linalg.inv(bind[parent])@bind[j]
    relaxed=np.load(path.with_name('relaxed_hands_rest_pose.npy')).astype(np.float64)
    owner=np.full(len(names),-1,dtype=int)
    for j in range(len(names)):
        hits=np.flatnonzero(controls==j)
        owner[j]=int(hits[0]) if len(hits) else owner[parents[j]]
        assert owner[j]>=0
    dense=np.zeros((len(vertices),len(controls)))
    for slot in range(indices.shape[1]):
        np.add.at(dense,(np.arange(len(vertices)),owner[indices[:,slot]]),weights[:,slot])
    np.testing.assert_allclose(dense.sum(1),1,atol=1e-12)
    return Skin(vertices,faces,bind,np.linalg.inv(bind),indices,weights,parents,controls,
        local,relaxed,dense,str(path),sha(path))


def expand_transforms(positions,rotations,skin,relaxed_hands=True):
    p=np.asarray(positions,dtype=np.float64);r=np.asarray(rotations,dtype=np.float64)
    if p.shape[1:]==(30,3):p=p[:,BODY_IN30];r=r[:,BODY_IN30]
    assert p.shape[1:]==(23,3) and r.shape==(*p.shape[:2],3,3)
    assert np.isfinite(p).all() and np.isfinite(r).all()
    full=np.broadcast_to(np.eye(4),(len(p),77,4,4)).copy()
    controlled={int(j):i for i,j in enumerate(skin.controls)}
    for j,parent in enumerate(skin.parents):
        if j in controlled:
            i=controlled[j];full[:,j,:3,:3]=r[:,i];full[:,j,:3,3]=p[:,i]
        else:
            local=skin.local_bind[j].copy()
            if relaxed_hands:
                local[:3,:3]=local[:3,:3]@skin.relaxed[j]
            full[:,j]=full[:,parent]@local
    selected=np.take(full,skin.controls,axis=1)
    np.testing.assert_array_equal(selected[...,:3,3],p)
    np.testing.assert_array_equal(selected[...,:3,:3],r)
    return full


def vertices_from_transforms(transforms,skin,chunk_frames=8):
    transforms=np.asarray(transforms,dtype=np.float64)
    assert transforms.shape[1:]==(77,4,4)
    output=np.empty((len(transforms),len(skin.vertices),3),dtype=np.float32)
    homogeneous=np.column_stack((skin.vertices,np.ones(len(skin.vertices))))
    for start in range(0,len(transforms),chunk_frames):
        affine=(transforms[start:start+chunk_frames]@skin.inverse_bind[None])[...,:3,:]
        selected=affine[:,skin.indices]
        points=np.einsum('tvwkl,vl->tvwk',selected,homogeneous,optimize=True)
        output[start:start+chunk_frames]=(points*skin.weights[None,:,:,None]).sum(2).astype(np.float32)
    assert np.isfinite(output).all()
    return output


def mesh_motion(positions,rotations,skin,relaxed_hands=True):
    return vertices_from_transforms(expand_transforms(positions,rotations,skin,relaxed_hands),skin)


def joint_error_surface(predicted,target,skin,reference='root_relative'):
    p,g=np.asarray(predicted),np.asarray(target)
    if p.shape[1:]==(30,3):p=p[:,BODY_IN30]
    if g.shape[1:]==(30,3):g=g[:,BODY_IN30]
    assert p.shape==g.shape and p.shape[1:]==(23,3)
    p=p.astype(np.float64);g=g.astype(np.float64)
    if reference=='root_relative':
        # Per-frame pelvis centering removes translation error only. Retain
        # orientation and scale; never modify the actual skinning inputs.
        p=p-p[:,:1];g=g-g[:,:1]
    elif reference!='world':
        raise ValueError('Error reference must be root_relative or world')
    errors=np.linalg.norm(p-g,axis=-1)
    return errors,errors@skin.vertex_joint_weights.T
