"""Meaningful contracts: no loss, no RNG drift, same recipe and sampler states."""
import copy
from types import SimpleNamespace
import unittest

import torch

from motion_expert_phase3_edge_image_joint.policy import configure as base_configure
from motion_expert_phase3_edge_image_joint.objectives import noise_batch as base_noise
from motion_expert_phase3_edge_image_joint.sample import sample as base_sample
from .policy import configure
from .objectives import camera_clock, noise_batch, specialist_loss
from .sample import sample
from .execution import topology, require_acceptance


class CameraPredictionTests(unittest.TestCase):
    def setUp(self):
        from motion_expert_phase3_edge_joint.settings import RECIPE
        self.saved_recipe=copy.deepcopy(RECIPE)
        RECIPE.pop('camera_prediction',None)

    def tearDown(self):
        from motion_expert_phase3_edge_joint.settings import RECIPE
        RECIPE.clear();RECIPE.update(self.saved_recipe)

    def test_only_camera_policy_and_identity_change(self):
        baseline=copy.deepcopy(base_configure())
        actual=configure()
        self.assertEqual({k:v for k,v in actual.items() if k not in ['experiment','camera_prediction']},
                         {k:v for k,v in baseline.items() if k!='experiment'})
        self.assertEqual(actual['camera_prediction']['loss_weight'],0.)

    def test_single_gpu_retains_dual_rank_draw_groups_and_loss_averaging(self):
        from motion_expert_phase3_edge_joint.data import GlobalDrawSampler
        configure()
        serial=list(GlobalDrawSampler(37, 2, 5, 16, 0, 1))
        parallel=[list(GlobalDrawSampler(37, 2, 5, 16, r, 2)) for r in range(2)]
        self.assertEqual(topology(1,16)['accumulation'],2)
        self.assertEqual(topology(2,16)['accumulation'],1)
        for step in range(3):
            groups=[parallel[r][step] for r in range(2)]
            self.assertEqual(serial[2*step:2*step+2],groups)
            weights=torch.tensor([.3,-.2],dtype=torch.float64,requires_grad=True)
            def loss(group):
                x=torch.tensor([[draw%7,index%11] for draw,index in group],dtype=torch.float64)
                return ((x@weights).sin()-.1).square().mean()
            rank_gradients=[torch.autograd.grad(loss(group),weights)[0] for group in groups]
            for group in serial[2*step:2*step+2]:(loss(group)/2).backward()
            torch.testing.assert_close(weights.grad,torch.stack(rank_gradients).mean(0),rtol=1e-12,atol=1e-12)

    def test_diagnostic_or_other_topology_preflight_cannot_authorize_production(self):
        checks=['camera_predictions_all_three_tasks_no_clean_camera_tokens',
                'zero_camera_loss_and_no_camera_readout_gradient',
                'joint_caption_invariant_camera_motion_video_outputs']
        receipt=dict(passed=True,fingerprint='current',world_size=1,microbatch=16,checks=checks)
        require_acceptance(receipt,'current',1,16)
        for altered in [dict(microbatch=4),dict(world_size=2),dict(passed=False),
                        dict(fingerprint='old_source'),dict(checks=checks[:-1])]:
            with self.assertRaises(RuntimeError):
                require_acceptance({**receipt,**altered},'current',1,16)

    def test_target_clock_never_uses_the_clean_condition_clock(self):
        modes=['video2motion','motimg2video','joint']
        actual=camera_clock(modes,torch.tensor([.7,0.,.6]),torch.tensor([0.,.8,.3]))
        self.assertTrue(torch.equal(actual,torch.tensor([.7,.8,.3])))

    def test_original_noise_targets_and_global_rng_are_preserved(self):
        configure()
        raw=dict(motion=torch.zeros(3,97,283),video=torch.zeros(3,48,25,16,16),
                 neutral=torch.zeros(3,30,3),camera_action=torch.ones(3,96,64),
                 modes=['video2motion','motimg2video','joint'],captions=['a','b','private'],draws=[4,5,6])
        torch.manual_seed(131)
        start=torch.get_rng_state()
        expected,target=base_noise(raw,'cpu');after=torch.get_rng_state()
        torch.set_rng_state(start)
        actual,new_target=noise_batch(raw,'cpu')
        self.assertTrue(torch.equal(after,torch.get_rng_state()))
        self.assertEqual(set(new_target),{'motion','velocity'})
        for k in target:self.assertTrue(torch.equal(target[k],new_target[k]),k)
        for k,v in expected.items():
            if torch.is_tensor(v):self.assertTrue(torch.equal(v,actual[k]),k)
            else:self.assertEqual(v,actual[k])
        torch.set_rng_state(start);repeat,_=noise_batch(raw,'cpu')
        self.assertTrue(torch.equal(actual['camera_state'],repeat['camera_state']))
        self.assertFalse(torch.equal(actual['camera_state'],raw['camera_action']))

    def test_camera_values_and_predictions_have_no_native_video_loss_edge(self):
        configure()
        prediction=torch.randn(1,48,25,16,16,requires_grad=True)
        camera=torch.randn(1,96,64,requires_grad=True)
        mask=torch.zeros(1,25,16,16);mask[:,0]=1
        output=dict(video=prediction,motion=torch.zeros(1,97,283),camera_velocity=camera,
                    preds_action=[camera[0]],anchor=prediction.sum()*0,
                    packed=SimpleNamespace(vision=SimpleNamespace(condition_mask=[mask])))
        inputs=dict(modes=['motimg2video'],video_sigma=torch.tensor([.5]))
        target=dict(velocity=torch.zeros_like(prediction),motion=torch.zeros(1,97,283))
        mean,std=torch.zeros(283),torch.ones(283)
        loss,_=specialist_loss(output,target,inputs,mean,std)
        altered={**output,'camera_velocity':torch.full_like(camera,float('nan')),
                 'preds_action':[torch.full_like(camera[0],float('nan'))]}
        other,_=specialist_loss(altered,target,inputs,mean,std)
        self.assertTrue(torch.equal(loss,other))
        self.assertIsNone(torch.autograd.grad(loss,camera,allow_unused=True,retain_graph=True)[0])
        self.assertGreater(torch.autograd.grad(loss,prediction)[0].abs().max().item(),0)

    def test_added_camera_sampling_preserves_both_original_modality_clocks_and_draws(self):
        configure()
        class Model:
            def eval(self):return self
            def __call__(self,**x):
                return dict(motion=x['motion']*.3+.01,video=x['video']*.2,
                            camera_velocity=torch.zeros(1,96,64))
        values=dict(motion=torch.zeros(1,97,283),neutral=torch.zeros(1,30,3),video=torch.ones(1,48,25,16,16))
        for mode in ['video2motion','motimg2video','joint']:
            baseline=base_sample(Model(),mode=mode,caption='',**values,seed=17,steps=4,guidance=1.)
            actual=sample(Model(),mode=mode,caption='',**values,seed=17,steps=4,guidance=1.)
            if mode=='joint':
                for key in ['motion','video']:self.assertTrue(torch.equal(actual[key],baseline[key]),key)
            else:
                key='motion' if mode=='video2motion' else 'video'
                self.assertTrue(torch.equal(actual[key],baseline),mode)
            self.assertEqual(tuple(actual['camera'].shape),(1,96,64))


if __name__=='__main__':unittest.main()
