"""Check the changed loss and gradients against the actual weight-ten objective."""
import copy
from types import SimpleNamespace
import unittest
import torch
from motion_expert_phase3_edge_joint.settings import RECIPE
from motion_expert_phase3_edge_joint.objectives import specialist_loss as original_loss
from .objectives import specialist_loss
from .policy import configure, check_checkpoint

class VideoWeightTests(unittest.TestCase):
    def setUp(self):
        self.saved = copy.deepcopy(RECIPE)
        configure()

    def tearDown(self):
        RECIPE.clear(); RECIPE.update(self.saved)

    def test_recipe_differs_only_in_video_weight_and_experiment(self):
        from motion_expert_phase3_edge_image_joint_camera.policy import configure as base_configure
        RECIPE['video_weight'] = 10.0
        expected = copy.deepcopy(base_configure())
        actual = copy.deepcopy(configure())
        self.assertEqual({k:v for k,v in actual.items() if k not in {'experiment','video_weight'}},
                         {k:v for k,v in expected.items() if k not in {'experiment','video_weight'}})
        self.assertEqual(actual['video_weight'], 1.)
        self.assertEqual(actual['motion_weights']['w_foot_height'], 10.)
        self.assertEqual(actual['camera_prediction']['loss_weight'], 0.)
        check_checkpoint({'scheduler':{'recipe':actual}})
        with self.assertRaises(ValueError):
            check_checkpoint({'scheduler':{'recipe':expected}})

    def test_real_native_losses_and_gradients_all_task_partitions(self):
        torch.manual_seed(31)
        for modes in [['video2motion'],['motimg2video'],['joint'],['video2motion','motimg2video','joint']]:
            with self.subTest(modes=modes):
                batch = len(modes)
                motion = (torch.randn(batch,97,283)*.1).requires_grad_()
                video = torch.randn(batch,48,25,16,16,requires_grad=True)
                camera = torch.randn(batch,96,64,requires_grad=True)
                mask = torch.zeros(1,25,16,16); mask[:,0] = 1
                output = dict(motion=motion, video=video, camera_velocity=camera,
                    anchor=(motion.sum()+video.sum())*0,
                    packed=SimpleNamespace(vision=SimpleNamespace(condition_mask=[mask]*batch)))
                target = dict(motion=torch.randn_like(motion)*.1, velocity=torch.randn_like(video))
                inputs = dict(modes=modes, video_sigma=torch.full((batch,),.5))
                mean,std = torch.zeros(283),torch.ones(283)
                before,old_terms = original_loss(output,target,inputs,mean,std)
                after,terms = specialist_loss(output,target,inputs,mean,std)
                expected = old_terms['motion']*terms['motion_count']/batch + terms['video']*terms['video_count']/batch
                torch.testing.assert_close(after,expected,rtol=2e-6,atol=1e-6)
                torch.testing.assert_close(before-after,9*terms['video']*terms['video_count']/batch,rtol=1e-3,atol=1e-3)
                self.assertTrue(torch.equal(terms['video'],old_terms['video']))
                self.assertTrue(torch.equal(terms['motion'],old_terms['motion']))
                self.assertTrue(torch.equal(terms['video_weighted'],terms['video']))
                old_grad=torch.autograd.grad(before,(motion,video,camera),retain_graph=True,allow_unused=True)
                new_grad=torch.autograd.grad(after,(motion,video,camera),allow_unused=True)
                torch.testing.assert_close(new_grad[0],old_grad[0],rtol=0,atol=0)
                torch.testing.assert_close(new_grad[1]*10,old_grad[1],rtol=2e-6,atol=1e-9)
                self.assertIsNone(new_grad[2])
                self.assertTrue(torch.equal(new_grad[1][:,:,0],torch.zeros_like(new_grad[1][:,:,0])))

if __name__ == '__main__':
    unittest.main()
