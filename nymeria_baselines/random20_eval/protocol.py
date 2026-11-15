"""Validate the explicitly selected no-caption MMDiT training recipe."""


def check_training_recipe(recipe, run):
    if recipe.get('bridge_parameterization') != 'mmdit':
        raise ValueError('Only MMDiT checkpoints are selected')
    if not recipe.get('caption_policy') or any(value != 'empty' for value in recipe['caption_policy'].values()):
        raise ValueError('Expected caption-free training')
    expected = run.get('training_tasks', {'video2motion': .5, 'motimg2video': .5, 'joint': 0.})
    actual = recipe['tasks']
    keys = {'video2motion', 'motimg2video', 'joint'}
    if set(actual) - keys or set(expected) - keys:
        raise ValueError('Unknown training task')
    if any(actual.get(k, 0.) != expected.get(k, 0.) for k in keys):
        raise ValueError('Selected checkpoint training task mix differs')
    if any(actual.get(k, 0.) < 0 for k in keys) or abs(sum(actual.values()) - 1.) > 1e-8:
        raise ValueError('Invalid training task probabilities')
