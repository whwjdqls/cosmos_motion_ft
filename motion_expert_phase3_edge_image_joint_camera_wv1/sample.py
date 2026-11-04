"""Same camera sampler, with strict weight-one checkpoint identity."""
import inspect
from motion_expert_phase3_edge_image_joint_camera import sample as original
from .policy import configure, check_checkpoint

sample = original.sample

def main():
    source = inspect.getsource(original.main)
    marker = "    if payload['scheduler']['recipe'].get('camera_prediction') != CAMERA_POLICY:"
    if source.count(marker) != 1:
        raise RuntimeError('Upstream checkpoint loader changed')
    source = source.replace(marker, '    check_checkpoint(payload)\n' + marker)
    namespace = original.__dict__.copy()
    namespace.update(configure=configure, check_checkpoint=check_checkpoint)
    exec(compile(source, __file__ + ':loader', 'exec'), namespace)
    namespace['main']()

if __name__ == '__main__':
    main()
