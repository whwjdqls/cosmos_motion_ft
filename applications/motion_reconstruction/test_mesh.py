"""Independent geometric/error invariants for the publication renderer."""
from pathlib import Path
import os
import numpy as np
from .mesh import load_skin,expand_transforms,vertices_from_transforms,joint_error_surface
from .colors import error_colors,hex_rgb,LOW,HIGH

ASSET=os.environ.get('SOMA_SKIN_PATH',str(Path(os.environ.get('KIMODO_ROOT',
    str(Path(__file__).resolve().parents[3]/'kimodo_open')))/'kimodo/assets/skeletons/somaskel77/skin_standard.npz'))


def test_bind_and_rigid_equivariance():
    s=load_skin(ASSET)
    r=s.bind[s.controls,:3,:3][None];p=s.bind[s.controls,:3,3][None]
    expanded=expand_transforms(p,r,s,relaxed_hands=False)
    np.testing.assert_allclose(expanded[0],s.bind,atol=1e-12)
    v=vertices_from_transforms(expanded,s)
    np.testing.assert_allclose(v[0],s.vertices,atol=1e-7)
    rotation=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    translation=np.array([.4,.7,-.2])
    moved=expand_transforms(p@rotation.T+translation,rotation@r,s,relaxed_hands=False)
    np.testing.assert_allclose(vertices_from_transforms(moved,s)[0],s.vertices@rotation.T+translation,atol=2e-7)


def test_controlled_positions_are_not_replaced_by_fk():
    s=load_skin(ASSET)
    r=s.bind[s.controls,:3,:3][None];p=s.bind[s.controls,:3,3][None].copy()
    p[:,17]+=[.3,-.2,.1]
    expanded=expand_transforms(p,r,s)
    np.testing.assert_array_equal(np.take(expanded,s.controls,axis=1)[...,:3,3],p)


def test_uniform_translation_joint_heat_and_palette():
    s=load_skin(ASSET)
    p=s.bind[s.controls,:3,3][None]
    joint,surface=joint_error_surface(p+[.1,0,0],p,s,reference='world')
    np.testing.assert_allclose(joint,.1,atol=1e-12)
    np.testing.assert_allclose(surface,.1,atol=1e-12)
    np.testing.assert_allclose(error_colors(np.array([0.,.3,.8])),[hex_rgb(LOW),hex_rgb(HIGH),hex_rgb(HIGH)],atol=3e-8)
    zeros=joint_error_surface(p,p,s)[1]
    assert np.all(zeros==0)
    colors=error_colors(np.linspace(0,.3,257))
    assert np.isfinite(colors).all() and colors.min()>=0 and colors.max()<=1


def test_local_errors_ignore_framewise_translation_but_retain_pose_error():
    s=load_skin(ASSET)
    gt=np.repeat(s.bind[s.controls,:3,3][None],3,axis=0)
    shifts=np.array([[.4,.2,-.8],[-.7,1.1,.3],[2.,-.5,.1]])[:,None]
    pred=gt+shifts
    local,surface=joint_error_surface(pred,gt,s)
    np.testing.assert_allclose(local,0,atol=5e-16)
    np.testing.assert_allclose(surface,0,atol=5e-16)
    pred[:,10]+=[.03,.04,0]
    local,surface=joint_error_surface(pred,gt,s)
    expected=np.zeros((3,23));expected[:,10]=.05
    np.testing.assert_allclose(local,expected,atol=5e-16)
    np.testing.assert_allclose(surface,expected@s.vertex_joint_weights.T,atol=5e-16)
    rotation=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    oriented,_=joint_error_surface(gt@rotation.T,gt,s)
    assert oriented.max()>.1  # No hidden heading/Procrustes alignment.
