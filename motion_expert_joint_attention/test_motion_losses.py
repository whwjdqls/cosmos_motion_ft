"""Focused CPU contracts for joint-attention contact-aware losses."""
from __future__ import annotations

import unittest

import torch

from motion_losses import contact_aware_losses
from uniego_layout import FEAT_DIM, FOOT_JOINT_IDX, FOOT_SLICE, FOOT_Y_IDX


class ContactAwareLossTest(unittest.TestCase):
    def setUp(self):
        self.mean = torch.zeros(FEAT_DIM)
        self.std = torch.ones(FEAT_DIM)
        self.mean[FOOT_SLICE] = torch.tensor([0.70, 0.80, 0.70, 0.80])
        self.std[FOOT_SLICE] = torch.tensor([0.46, 0.40, 0.46, 0.40])
        self.valid = torch.ones(1, 3, dtype=torch.bool)

        raw = torch.zeros(1, 3, FEAT_DIM)
        raw[..., FOOT_SLICE] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.x0 = (raw - self.mean) / self.std
        self.joints_gt = torch.zeros(1, 3, 30, 3)

    def losses(self, x0_hat, joints_hat):
        return contact_aware_losses(
            x0_hat,
            self.x0,
            joints_hat,
            self.valid,
            self.mean,
            self.std,
            fps=20.0,
            contact_logit_scale=2.0,
        )

    def test_exact_contacts_and_stationary_feet_have_expected_losses(self):
        contact, velocity, height = self.losses(self.x0.clone(), self.joints_gt.clone())
        self.assertGreater(float(contact), 0.0)
        self.assertLess(float(contact), 0.5)
        self.assertEqual(float(velocity), 0.0)
        self.assertEqual(float(height), 0.0)

    def test_wrong_contacts_increase_balanced_bce(self):
        wrong = self.x0.clone()
        wrong_raw = torch.ones(1, 3, 4) - torch.tensor([1.0, 0.0, 0.0, 0.0])
        wrong[..., FOOT_SLICE] = (
            wrong_raw - self.mean[FOOT_SLICE]
        ) / self.std[FOOT_SLICE]
        correct_loss, _, _ = self.losses(self.x0.clone(), self.joints_gt.clone())
        wrong_loss, _, _ = self.losses(wrong, self.joints_gt.clone())
        self.assertGreater(float(wrong_loss), float(correct_loss))

    def test_physical_losses_only_use_contacting_foot(self):
        joints_hat = self.joints_gt.clone()
        x0_hat = self.x0.clone()
        contacting_joint = FOOT_JOINT_IDX[0]
        joints_hat[0, :, contacting_joint, 0] = torch.tensor([0.0, 0.1, 0.2])
        x0_hat[0, :, FOOT_Y_IDX[0]] = 0.2

        # A non-contacting foot can move without affecting either physical term.
        joints_hat[0, :, FOOT_JOINT_IDX[1], 0] = torch.tensor([0.0, 10.0, 20.0])
        _, velocity, height = self.losses(x0_hat, joints_hat)
        torch.testing.assert_close(velocity, torch.tensor(2.0))
        torch.testing.assert_close(height, torch.tensor(0.04))

    def test_padding_is_excluded(self):
        self.valid[:, 1:] = False
        joints_hat = self.joints_gt.clone()
        x0_hat = self.x0.clone()
        joints_hat[0, 1:, FOOT_JOINT_IDX[0], :] = 100.0
        x0_hat[0, 1:, FOOT_Y_IDX[0]] = 100.0
        _, velocity, height = self.losses(x0_hat, joints_hat)
        self.assertEqual(float(velocity), 0.0)
        self.assertEqual(float(height), 0.0)


if __name__ == "__main__":
    unittest.main()
