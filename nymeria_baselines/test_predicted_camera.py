"""Geometry tests: gauge invariance, metric scale and native conventions."""
import unittest
import numpy as np
from nymeria_baselines.predicted_camera import (
    remove_initial_gauge, egoego_condition, uem_condition, egoallo_poses,
    common_from_world, yaw_rotation, Y_FROM_Z, validate_poses)


class CameraContracts(unittest.TestCase):
    def setUp(self):
        self.pose = np.tile(np.eye(4), (97, 1, 1))
        self.pose[:, 2, 3] = np.linspace(0, 2, 97)
        self.pose[:, 1, 3] = np.linspace(0, -.4, 97)
        self.initial = np.diag([1., -1., -1.])

    def test_world_gauge_does_not_change_conditions(self):
        gauge = np.eye(4); gauge[:3, :3] = yaw_rotation(np.array([[0,0,1],[0,1,0],[-1,0,0]]))
        gauge[:3, 3] = [42, 9, -7]
        a = remove_initial_gauge(self.pose, self.initial, 1.6)
        b = remove_initial_gauge(gauge @ self.pose, self.initial, 1.6)
        np.testing.assert_allclose(a, b, atol=1e-12)
        np.testing.assert_allclose(egoego_condition(a), egoego_condition(b), atol=1e-6)

    def test_metric_scale_and_vertical_change_are_preserved(self):
        a = remove_initial_gauge(self.pose, np.eye(3), 1.6)
        self.assertAlmostEqual(a[-1, 2, 3], 2.)
        self.assertAlmostEqual(a[-1, 1, 3], 1.2)
        np.testing.assert_allclose(a[0, :3, 3], [0, 1.6, 0])

    def test_native_zup_and_previous_frame(self):
        a = remove_initial_gauge(self.pose, np.eye(3), 1.6)
        z = egoallo_poses(a)
        self.assertEqual(z.shape, (98, 4, 4))
        np.testing.assert_array_equal(z[0], z[1])
        np.testing.assert_allclose(Y_FROM_Z @ z[1:, :3, :3], a[:, :3, :3], atol=1e-6)
        np.testing.assert_allclose(z[1:, :3, 3] @ Y_FROM_Z.T, a[:, :3, 3], atol=1e-6)

    def test_height_survives_both_condition_encodings(self):
        a = remove_initial_gauge(self.pose, np.eye(3), 1.6)
        self.assertAlmostEqual(float(egoego_condition(a)[0, 1]), 1.6, places=6)
        self.assertAlmostEqual(float(uem_condition(a)[0, 7]), 1.6, places=6)
        np.testing.assert_array_equal(uem_condition(a)[0, 9:], [1,0,0,0,1,0,0,0,0])

    def test_camera_only_benchmark_anchor_keeps_floor(self):
        a = remove_initial_gauge(self.pose, np.eye(3), 1.6)
        c = common_from_world(a, dict(head_rotation_in_camera=np.eye(3),head_offset_camera=[.1,-.05,.2]))
        head = a[0, :3, 3] + [.1,-.05,.2]
        np.testing.assert_allclose(c[:3, :3] @ head + c[:3, 3], [0,1.55,0])

    def test_invalid_or_reflected_pose_rejected(self):
        a=self.pose.copy(); a[:, 0, 0]=-1
        with self.assertRaises(AssertionError): validate_poses(a)
        a=self.pose.copy(); a[0, 0, 0]=np.nan
        with self.assertRaises(ValueError): validate_poses(a)


if __name__ == '__main__': unittest.main()
