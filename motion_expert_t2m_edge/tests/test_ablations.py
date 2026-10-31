"""Objective/representation checks with physical semantics and legacy parity."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import ablation_settings as settings
from flow import add_noise_x0_masked, add_noise_velocity_masked, sample_x0_unipc
from feature_runtime import decode_joints
from kimodo_codec import cached_encode, encode_uniego
from losses import t2m_reconstruction_loss
from motion_losses import contact_aware_losses
from uniego_layout import FOOT_JOINT_IDX, FOOT_Y_IDX


def source_features(frames=12):
    raw=torch.zeros(frames,283)
    local=raw[:,:270].view(frames,30,9)
    local[...,0]=1; local[...,4]=1
    local[...,7]=torch.arange(30)*.02+1
    local[:,22,6]=-.15;local[:,26,6]=.15
    raw[:,270]=1;raw[:,274]=1
    raw[1:,276]=.03; raw[1:,278]=-.01
    raw[:,-4:]=torch.tensor([1.,0.,1.,0.])
    return raw


class AblationTest(unittest.TestCase):
    def test_limited_statistics_are_never_accepted_for_production(self):
        import json
        with tempfile.TemporaryDirectory() as root:
            (Path(root)/'summary.json').write_text(json.dumps(dict(codec='kimodo369_v1',
                split='train',complete=True,limited=True)))
            with mock.patch.object(settings,'CODEC','kimodo369_v1'),mock.patch.object(settings,'STATS_DIR',root):
                with self.assertRaisesRegex(RuntimeError,'Limited/test/smoke'):
                    settings.stats_manifest()

    def test_checkpoint_objective_resolves_before_construction_and_rejects_conflicts(self):
        with mock.patch.dict(os.environ,clear=False),mock.patch.object(settings,'CODEC','uniego283'), \
             mock.patch.object(settings,'OBJECTIVE','x0'),mock.patch.object(settings,'STATS_DIR','irrelevant'):
            for key in ('EDGE_PHASE2_FEATURE_CODEC','EDGE_PHASE2_OBJECTIVE'):
                os.environ.pop(key,None)
            os.environ['PHASE2_KIMODO_STATS_DIR']='unused_for_uniego'
            settings.configure_checkpoint(dict(version=1,feature_codec='uniego283',objective='velocity'))
            self.assertEqual(settings.OBJECTIVE,'velocity')
            os.environ['EDGE_PHASE2_OBJECTIVE']='velocity'
            with self.assertRaisesRegex(RuntimeError,'conflicts'):
                settings.configure_checkpoint(None)

    def test_a3_partition_includes_historical_double_dropout(self):
        value=settings.conditioning_partition()
        self.assertAlmostEqual(sum(value.values()),1)
        expected=dict(bones_text=.30375,bones_null=.07125,nymeria_text=.1125,
                      nymeria_null=.0125,nymeria_image_text=.35,nymeria_image_only=.15)
        for key,n in expected.items(): self.assertAlmostEqual(value[key],n)

    def test_per_task_dropout_and_legacy_rng_are_preserved(self):
        from data import EdgePhase2Dataset
        from nymeria_joint_dataset import NymeriaJointDataset
        import random
        row=object.__new__(EdgePhase2Dataset)
        row._train=True;row._cfg_dropout=.1;row._ti2m_cfg_dropout=.3
        row._active_rng=lambda: rng
        rng=random.Random(7)
        # Same input/random stream yields exactly the legacy T2M outcomes.
        state=rng.getstate()
        new=[row._apply_caption_policy('text2motion','C walks') for _ in range(100)]
        rng.setstate(state)
        old=[NymeriaJointDataset._apply_caption_policy(row,'text2motion','C walks') for _ in range(100)]
        self.assertEqual(new,old)
        for draw,expected in ((.2,''),(.4,'The camera wearer walks')):
            rng=mock.Mock();rng.random.return_value=draw
            self.assertEqual(row._apply_caption_policy('textimg2motion','C walks'),expected)

    def test_velocity_target_preserves_noising_rng_masks_and_clean_pose(self):
        x=torch.randn(3,7,283)
        pad=torch.tensor([[False]*5+[True]*2,[False]*7,[False]*7])
        sigma=torch.tensor([0.,.2,1.])
        torch.manual_seed(9)
        old=add_noise_x0_masked(x,pad,sigma)
        state=torch.random.get_rng_state()
        torch.manual_seed(9)
        noisy,s,v,valid=add_noise_velocity_masked(x,pad,sigma)
        self.assertTrue(torch.equal(old[0],noisy))
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        self.assertTrue(torch.equal(old[3],valid))
        self.assertTrue(torch.equal(noisy[pad],x[pad]))
        restored=noisy-s[:,None,None]*v
        torch.testing.assert_close(restored[valid],x[valid],rtol=2e-5,atol=5e-7)

    def test_kimodo_roundtrip_contacts_and_cache(self):
        source=source_features()
        actual=encode_uniego(source)
        self.assertEqual(tuple(actual.shape),(len(source),369))
        torch.testing.assert_close(decode_joints(actual[None]),decode_joints(source[None]),rtol=0,atol=2e-7)
        self.assertTrue(torch.equal(actual[:,-4:],source[:,-4:]))
        self.assertTrue(torch.equal(actual[:,1],decode_joints(source[None])[0,:,0,1]))
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ,{'PHASE2_KIMODO_CACHE_ROOT':root}):
            first=cached_encode(source)
            with mock.patch('kimodo_codec.encode_uniego',side_effect=AssertionError('cache miss')):
                second=cached_encode(source)
            self.assertTrue(torch.equal(first,second))
            self.assertEqual(len(list(Path(root).rglob('*.npy'))),1)

    def test_tiny_windows_are_preserved_and_velocity_uses_actual_length(self):
        for n in (1,2,3,12):
            features=source_features(n)
            converted=encode_uniego(features)
            positions=decode_joints(features[None])[0]
            expected=(positions[1:]-positions[:-1])*20
            velocity=converted[:,275:365].reshape(n,30,3)
            if n>1:
                torch.testing.assert_close(velocity[:-1],expected)
                torch.testing.assert_close(velocity[-1],expected[-1])
            else:
                self.assertEqual(float(velocity.abs().sum()),0)
            torch.testing.assert_close(decode_joints(converted[None]),positions[None],rtol=0,atol=2e-7)

    def test_representation_collation_keeps_padding_and_shape(self):
        import config
        from data import collate_joint
        rows=[]
        for n in (8,12):
            rows.append(dict(mode='text2motion',source='bones',caption='walk',
                domain_id=torch.tensor(2),motion=encode_uniego(source_features(n)),
                motion_pad_mask=torch.zeros(n,dtype=torch.bool),neutral_joints=torch.zeros(30,3)))
        with mock.patch.object(config,'MOTION_DIM',369): value=collate_joint(rows)
        self.assertEqual(tuple(value['motion'].shape),(2,12,369))
        self.assertTrue(value['motion_pad_mask'][0,8:].all())
        self.assertTrue(torch.equal(value['motion'][0,:8],rows[0]['motion']))
        self.assertTrue(torch.equal(value['neutral_joints'][0],rows[0]['neutral_joints']))

    def test_legacy_contract_is_identical_and_cannot_silently_become_velocity(self):
        import config
        import subprocess
        import types
        from checkpoint import validate_checkpoint_contract
        baseline=types.ModuleType('_pre_ablation_config')
        baseline.__file__=config.__file__
        source=subprocess.check_output(['git','-C',str(config.REPO_ROOT),'show',
                                       'f3a7a89:motion_expert_t2m_edge/config.py'],text=True)
        exec(compile(source,config.__file__,'exec'),baseline.__dict__)
        with mock.patch.object(settings,'CODEC','uniego283'),mock.patch.object(settings,'OBJECTIVE','x0'):
            hashes=config.validate_artifacts()
            baseline.validate_artifacts=lambda: hashes
            expected=baseline.architecture_contract()
            self.assertEqual(config.architecture_contract(),expected)
            validate_checkpoint_contract({'contract':expected})
            with mock.patch.object(settings,'OBJECTIVE','velocity'):
                with self.assertRaisesRegex(RuntimeError,'codec/objective'):
                    validate_checkpoint_contract({'contract':expected})

    def test_kimodo_physical_losses_and_finite_backward(self):
        source=source_features()[None]
        kimodo=encode_uniego(source[0])[None]
        raw=kimodo.clone().requires_grad_(True)
        with torch.no_grad():
            raw[...,5:95]+=.01
            raw[0,2,5]+=.02
        valid=torch.ones(1,len(source[0]),dtype=torch.bool)
        mean=torch.zeros(369);mean[-4:]=.5
        std=torch.ones(369)
        pred=(raw-mean)/std;target=(kimodo-mean)/std
        loss,terms=t2m_reconstruction_loss(pred,target,valid,mean,std,w_joint=1,w_smooth=5,
            w_contact=.05,w_foot_vel=1,w_foot_height=10)
        loss.backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertGreater(float(raw.grad.abs().sum()),0)
        # Equivalent physical motion has equivalent contact/height/velocity losses in either layout.
        a=contact_aware_losses(source,source,decode_joints(source),valid,torch.zeros(283),torch.ones(283))
        b=contact_aware_losses(kimodo,kimodo,decode_joints(kimodo),valid,torch.zeros(369),torch.ones(369),
            contact_slice=slice(365,369),foot_y_indices=tuple(6+3*j for j in FOOT_JOINT_IDX))
        for old,new in zip(a,b): torch.testing.assert_close(old,new,rtol=1e-5,atol=1e-6)

    def test_velocity_feature_loss_is_direct_but_physical_losses_decode_clean(self):
        x=source_features()[None]
        torch.manual_seed(8)
        pad=torch.zeros(x.shape[:2],dtype=torch.bool)
        noisy,sigma,v,valid=add_noise_velocity_masked(x,pad,torch.tensor([.6]))
        output=v.clone().requires_grad_(True)
        clean=noisy-sigma[:,None,None]*output
        loss,terms=t2m_reconstruction_loss(clean,x,valid,torch.zeros(283),torch.ones(283),
            feature_prediction=output,feature_target=v,w_joint=1,w_smooth=5)
        self.assertEqual(float(terms['feature'].detach()),0)
        self.assertLess(float(loss.detach()),1e-10)
        (loss+output.square().mean()).backward()
        self.assertTrue(torch.isfinite(output.grad).all())

    def test_sampler_passes_native_velocity_once_and_preserves_legacy_x0(self):
        from cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler
        calls=[]
        original=FlowUniPCMultistepScheduler.step
        def capture(scheduler,**kwargs):
            calls.append(kwargs['model_output'].clone())
            return original(scheduler,**kwargs)
        noise=torch.randn(1,8,3)
        with mock.patch.object(FlowUniPCMultistepScheduler,'step',capture):
            result=sample_x0_unipc(lambda x,s:torch.full_like(x,.3),predict_null=lambda x,s:torch.full_like(x,.1),
                T=8,motion_dim=3,steps=5,guidance=2,device='cpu',initial_noise=noise,prediction_type='velocity')
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(len(calls),5)
        for value in calls: torch.testing.assert_close(value,torch.full_like(value,.5))
        with mock.patch.object(settings,'OBJECTIVE','velocity'):
            auto=sample_x0_unipc(lambda x,s:torch.full_like(x,.3),predict_null=lambda x,s:torch.full_like(x,.1),
                T=8,motion_dim=3,steps=5,guidance=2,device='cpu',initial_noise=noise)
        self.assertTrue(torch.equal(auto,result))
        baseline=sample_x0_unipc(lambda x,s:torch.zeros_like(x),T=8,motion_dim=3,steps=5,device='cpu',initial_noise=noise)
        explicit=sample_x0_unipc(lambda x,s:torch.zeros_like(x),T=8,motion_dim=3,steps=5,device='cpu',initial_noise=noise,prediction_type='x0')
        self.assertTrue(torch.equal(baseline,explicit))


if __name__=='__main__': unittest.main()
