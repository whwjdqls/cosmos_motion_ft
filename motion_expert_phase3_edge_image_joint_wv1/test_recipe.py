import copy
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from motion_expert_phase3_edge_joint.settings import RECIPE
from motion_expert_phase3_edge_joint.objectives import specialist_loss
from motion_expert_phase3_edge_joint.site_paths import storage_path
from .policy import configure, check_checkpoint


@pytest.fixture(autouse=True)
def restore_recipe():
    saved = copy.deepcopy(RECIPE)
    yield
    RECIPE.clear()
    RECIPE.update(saved)


def test_joint_recipe_and_checkpoint_rejection():
    recipe = copy.deepcopy(configure())
    assert recipe['video_weight'] == 1
    assert recipe['tasks'] == {'video2motion': .4, 'motimg2video': .4, 'joint': .2}
    assert recipe['conditioning_attention'] == 'bidirectional'
    check_checkpoint({'scheduler': {'recipe': recipe}})
    recipe['video_weight'] = 10
    with pytest.raises(ValueError, match='video_weight'):
        check_checkpoint({'scheduler': {'recipe': recipe}})


def test_actual_video_loss_and_gradient_scale_without_changing_motion():
    configure()
    def run(weight):
        RECIPE['video_weight'] = weight
        video = torch.full((2, 2, 3, 2, 2), .7, requires_grad=True)
        motion = torch.full((2, 7, 3), .3, requires_grad=True)
        output = dict(video=video, motion=motion, anchor=video.sum()*0+motion.sum()*0,
                      packed=SimpleNamespace(vision=SimpleNamespace(condition_mask=[
                          torch.tensor([1.,0.,0.])[:,None,None]]*2)))
        target = dict(velocity=torch.zeros_like(video), motion=torch.zeros_like(motion))
        inputs = dict(modes=['motimg2video','joint'], video_sigma=torch.ones(2)*.5)
        # Isolate video weighting while retaining the real native video-loss implementation.
        with patch('motion_expert_phase3_edge_joint.objectives.t2m_reconstruction_loss',
                   side_effect=lambda pred,*args,**kwargs: (pred.square().mean(), {})):
            loss, metrics = specialist_loss(output,target,inputs,torch.zeros(3),torch.ones(3))
        loss.backward()
        return loss.detach(), metrics, video.grad.clone(), motion.grad.clone()
    one, m1, gv1, gm1 = run(1)
    ten, m10, gv10, gm10 = run(10)
    torch.testing.assert_close(gv10,gv1*10)
    torch.testing.assert_close(gm10,gm1)
    torch.testing.assert_close(ten-one,9*m1['video'])
    torch.testing.assert_close(m1['video_weighted'],m1['video'])
    torch.testing.assert_close(m10['video_weighted'],10*m10['video'])


def test_site_boundaries_and_native_cache_mapping():
    from motion_expert_phase3_edge_image_joint.train import configure_local_paths
    with patch.dict(os.environ, {'COSMOS_SITE':'yonsei','COSMOS_STORAGE_ROOT':'/lustre/whwjdqls99/cosmos'}):
        os.environ.pop('PHASE3_IMAGE_JOINT_PATH_MAP',None)
        assert configure_local_paths() == {}
        assert storage_path('/scratch2/whwjdqls99/cosmos/test',must_exist=False).is_absolute()
        with pytest.raises(ValueError):
            storage_path('/mnt/local/another_site',must_exist=False)
        with pytest.raises(ValueError):
            storage_path('/scratch2/whwjdqls99/cosmos/../../../outside',must_exist=False)
    with patch.dict(os.environ, {'COSMOS_SITE':'custom'}):
        assert storage_path('/mnt/local/test',must_exist=False) == Path('/mnt/local/test')
        with pytest.raises(ValueError):
            storage_path('/scratch2/whwjdqls99/cosmos/test',must_exist=False)


def checked_collate(rows):
    from motion_expert_phase3_edge_joint.data import collate
    assert RECIPE['video_weight'] == 1
    assert RECIPE['tasks'] == {'video2motion': .4, 'motimg2video': .4, 'joint': .2}
    assert RECIPE['conditioning_attention'] == 'bidirectional'
    return collate(rows)


def test_spawned_worker_preserves_weight_one_and_joint_caption_policy(tmp_path):
    import json
    import numpy as np
    from motion_expert_phase3_edge_joint.data import identity
    from .data import Phase3Dataset
    latent = tmp_path/'latent.npz'
    np.savez(latent,latents=np.zeros((48,25,16,16),np.float16))
    np.save(tmp_path/'sample.motion.npy',np.zeros((1,97,283),np.float32))
    np.save(tmp_path/'sample.neutral.npy',np.zeros((30,3),np.float32))
    row=dict(latent=str(latent),latent_identity=identity(latent),array='sample',row=0,
             caption='Private target caption',uuid='test',start=0)
    (tmp_path/'index.json').write_text(json.dumps(dict(rows=[row])))
    configure()
    with patch.dict(os.environ, {'PHASE3_IMAGE_JOINT_PATH_MAP':'{}'}):
        batches=list(torch.utils.data.DataLoader(Phase3Dataset(tmp_path),batch_size=8,
            sampler=[(i,0) for i in range(32)],num_workers=1,multiprocessing_context='spawn',
            collate_fn=checked_collate))
    modes=[mode for batch in batches for mode in batch['modes']]
    assert set(modes) == {'video2motion','motimg2video','joint'}
    for batch in batches:
        assert all(caption == '' for caption,mode in zip(batch['captions'],batch['modes']) if mode != 'motimg2video')
