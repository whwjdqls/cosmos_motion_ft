"""Reuse native formulas, changing the two video multipliers explicitly."""
import inspect
from motion_expert_phase3_edge_joint.objectives import specialist_loss as original_loss
from motion_expert_phase3_edge_image_joint_camera.objectives import noise_batch

source = inspect.getsource(original_loss)
if source.count('10.0 *') != 2:
    raise RuntimeError('Upstream video weighting changed; review the ablation adapter')
namespace = original_loss.__globals__.copy()
exec(compile(source.replace('10.0 *', 'RECIPE["video_weight"] *'),
             __file__ + ':native_loss', 'exec'), namespace)
specialist_loss = namespace['specialist_loss']
