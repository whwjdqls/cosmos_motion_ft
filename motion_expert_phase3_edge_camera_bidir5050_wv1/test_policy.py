"""Disabled joint task never enters the sampler; all other recipe fields match."""
import copy
import random
import unittest
from collections import Counter
from motion_expert_phase3_edge_joint.settings import RECIPE, sample_task
from motion_expert_phase3_edge_image_joint_camera_wv1.policy import configure as base_configure
from .policy import configure, check_checkpoint

class TwoTaskTests(unittest.TestCase):
    def setUp(self):
        self.saved = copy.deepcopy(RECIPE)

    def tearDown(self):
        RECIPE.clear(); RECIPE.update(self.saved)

    def test_only_tasks_and_experiment_change(self):
        original = copy.deepcopy(base_configure())
        actual = copy.deepcopy(configure())
        self.assertEqual({k:v for k,v in original.items() if k not in {'experiment','tasks'}},
                         {k:v for k,v in actual.items() if k not in {'experiment','tasks'}})
        self.assertEqual(actual['tasks'], {'video2motion':.5,'motimg2video':.5,'joint':0.})
        self.assertEqual(actual['video_weight'],1.)
        self.assertEqual(actual['conditioning_attention'],'bidirectional')
        self.assertEqual(actual['camera_prediction']['loss_weight'],0.)
        check_checkpoint({'scheduler':{'recipe':actual}})
        with self.assertRaises(ValueError):
            check_checkpoint({'scheduler':{'recipe':original}})

    def test_actual_per_draw_sampler_never_selects_joint(self):
        configure()
        counts=Counter(sample_task(random.Random(RECIPE['seed']+1000003*(draw+1))) for draw in range(20000))
        self.assertEqual(set(counts),{'video2motion','motimg2video'})
        self.assertLess(abs(counts['video2motion']/20000-.5),.015)
        class Boundary:
            def __init__(self,x):self.x=x
            def random(self):return self.x
        for value,expected in [(0.,'video2motion'),(.499999999,'video2motion'),(.5,'motimg2video'),(.9999999999999999,'motimg2video')]:
            self.assertEqual(sample_task(Boundary(value)),expected)

if __name__ == '__main__':
    unittest.main()
