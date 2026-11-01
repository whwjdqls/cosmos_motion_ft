"""CPU geometry and evaluation-input regression checks; run on owned compute."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from .camera import CV_FROM_GL, cube_lookup, cube_rotations, integrate_actions, scaled_intrinsics, sha256
from .profile import load_uniform, make_record, verify


class CameraContractTests(unittest.TestCase):
    def test_calibration_uses_actual_clockwise_pixels_not_stale_upright_K(self):
        from .calibrate import export
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp)/'camera.npz', Path(tmp)/'rays.npz'
            raw = np.array([[600.,0,710.],[0,600.,705.],[0,0,1.]])
            stale = np.array([[600.,0,705.],[0,600.,697.],[0,0,1.]])
            rotation = np.eye(4)
            rotation[:2,:2] = [[0,1],[-1,0]]
            np.savez(source,K_rgb_raw=raw,K_rgb_upright=stale,
                     image_size_hw=[1408,1408],T_rgb_upright=rotation,
                     T_device_rgb=np.eye(4),camera_model='FISHEYE624')
            meta = export(source,out)
            expected = scaled_intrinsics(np.array([[600.,0,702.],[0,600.,710.],[0,0,1.]]),[1408,1408])
            np.testing.assert_allclose(meta['K_256'],expected)
            self.assertGreater(meta['stored_upright_K_max_discrepancy_pixels'],2.)

    def test_gl_cv_is_an_involution_and_proper_rotation(self):
        np.testing.assert_array_equal(CV_FROM_GL @ CV_FROM_GL, np.eye(4))
        self.assertAlmostEqual(np.linalg.det(CV_FROM_GL[:3,:3]), 1.)
        np.testing.assert_array_equal(CV_FROM_GL[:3,:3] @ [0,0,1], [0,0,-1])

    def test_backward_framewise_composes_translation_in_rotated_frame(self):
        a = np.tile([0.,0.,0.,1.,0.,0.,0.,1.,0.],(96,1))
        # Turn optical forward+Z toward+X; the next forward meter must go+X.
        a[0,3:] = [0,0,-1,0,1,0]
        a[1,:3] = [0,0,1]
        start = np.eye(4)
        start[:3,3] = [7,2,8]
        poses = integrate_actions(a,start)
        self.assertEqual(poses.shape,(97,4,4))
        np.testing.assert_allclose(poses[-1,:3,3],[8,2,8])
        np.testing.assert_allclose(poses[0],start)

    def test_scene_scale_changes_translation_only(self):
        a = np.tile([0.,0.,1.,1.,0.,0.,0.,1.,0.],(96,1))
        p = integrate_actions(a,np.eye(4),2.)
        np.testing.assert_allclose(p[-1,:3,3],[0,0,192])
        np.testing.assert_allclose(p[-1,:3,:3],np.eye(3))

    def test_degenerate_camera_is_rejected(self):
        with self.assertRaises(ValueError):
            integrate_actions(np.zeros((96,9)),np.eye(4))
        with self.assertRaises(ValueError):
            integrate_actions(np.zeros((97,9)),np.eye(4))

    def test_cubemap_faces_project_optical_axes_to_image_centers(self):
        rotations = cube_rotations()
        for i, rotation in enumerate(rotations):
            np.testing.assert_allclose(rotation.T @ rotation,np.eye(3))
            self.assertAlmostEqual(np.linalg.det(rotation),1.)
            faces,uv,valid = cube_lookup(rotation[:,2].reshape(1,1,3),384)
            self.assertEqual(int(faces[0,0]),i)
            np.testing.assert_allclose(uv[0,0],[191.5,191.5])
            self.assertTrue(valid[0,0])

    def test_resize_preserves_pixel_centers(self):
        k = np.array([[704.,0,703.5],[0,704.,703.5],[0,0,1.]])
        np.testing.assert_allclose(scaled_intrinsics(k,(1408,1408)),
                                   [[128.,0,127.5],[0,128.,127.5],[0,0,1.]])


class ProfileTests(unittest.TestCase):
    def test_saved_contract_json_roundtrip_and_changed_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'model').mkdir()
            (root/'model/.metadata').write_bytes(b'checkpoint identity')
            source = root/'reference.jsonl'
            record = dict(model_mode='inverse_dynamics',num_steps=30)
            source.write_text(json.dumps(record)+'\n')
            contract = dict(active_modes=('image2video','inverse_dynamics'),training_prefix_lengths=(1,))
            profile = dict(checkpoint=str(root),contract=json.loads(json.dumps(contract)),
                checkpoint_metadata_sha256=sha256(root/'model/.metadata'),sampler='unipc',ema=True,
                templates={'inverse_dynamics':record},evaluation_inputs={'inverse_dynamics':
                    dict(path=str(source),sha256=sha256(source))})
            with patch('native_phase_training.run_contract.resolve_eval_contract',
                       return_value=(SimpleNamespace(to_dict=lambda:contract),None,root,None)):
                verify(profile)
                source.write_text(json.dumps(dict(record,num_steps=35))+'\n')
                with self.assertRaises(ValueError):
                    verify(profile)

    def test_record_preserves_evaluation_settings_and_empty_inverse_text(self):
        template = dict(model_mode='inverse_dynamics',num_frames=97,resolution='256',
            aspect_ratio='1,1',fps=20,shift=10.,num_steps=30,guidance=1.,
            action_chunk_size=96,image_size=256,domain_name='camera_pose',view_point='ego_view',
            seed=0,prompt='',name='reference_inverse_dynamics',vision_path='reference.mp4')
        profile = {'templates':{'inverse_dynamics':template}}
        record = make_record(profile,'inverse_dynamics','segment','generated.mp4','ignored caption',123)
        for key in set(template)-{'vision_path','name','seed'}:
            self.assertEqual(record[key],template[key])
        self.assertEqual(record['seed'],123)
        self.assertEqual(template['seed'],0)

    def test_mixed_sampling_profiles_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder)/'inputs.jsonl'
            file.write_text(json.dumps(dict(model_mode='image2video',guidance=6))+'\n'+
                            json.dumps(dict(model_mode='image2video',guidance=1))+'\n')
            with self.assertRaises(ValueError):
                load_uniform(file,'image2video')


if __name__ == '__main__':
    unittest.main()
