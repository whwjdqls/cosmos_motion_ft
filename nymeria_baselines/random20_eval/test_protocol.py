import unittest
from .protocol import check_training_recipe


class ProtocolTests(unittest.TestCase):
    def recipe(self, tasks):
        return dict(bridge_parameterization='mmdit', tasks=tasks,
                    caption_policy={k: 'empty' for k in tasks})

    def test_default_preserves_two_task_guard(self):
        check_training_recipe(self.recipe(dict(video2motion=.5, motimg2video=.5, joint=0)), {})
        with self.assertRaises(ValueError):
            check_training_recipe(self.recipe(dict(video2motion=.4, motimg2video=.4, joint=.2)), {})

    def test_explicit_joint_checkpoint_mix_is_required(self):
        tasks = dict(video2motion=.4, motimg2video=.4, joint=.2)
        check_training_recipe(self.recipe(tasks), dict(training_tasks=tasks))
        with self.assertRaises(ValueError):
            check_training_recipe(self.recipe(tasks), dict(training_tasks=dict(video2motion=.5, motimg2video=.5)))

    def test_wrong_architecture_or_captions_are_rejected(self):
        r = self.recipe(dict(video2motion=.5, motimg2video=.5))
        r['bridge_parameterization'] = 'shared'
        with self.assertRaises(ValueError):check_training_recipe(r, {})
        r['bridge_parameterization'] = 'mmdit';r['caption_policy']['motimg2video'] = 'caption_dropout_0.1'
        with self.assertRaises(ValueError):check_training_recipe(r, {})
