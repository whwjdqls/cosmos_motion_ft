"""Load both trained motion and bridge state, then use the same camera sampler."""
from motion_expert_phase3_edge_image_joint_camera import sample as camera_sample
from .model import EdgePhase3
from .overlays import function_source, replace_once
from .policy import configure


def main():
    source = function_source(camera_sample, 'main')
    source = replace_once(source, "    model.bridges.load_state_dict(payload['bridges'],strict=True)",
        "    model.bridges.load_state_dict(payload['bridges'],strict=True)\n"
        "    model.motion.load_motion_state_dict(payload['motion_state'],strict=True)")
    namespace = dict(vars(camera_sample), EdgePhase3=EdgePhase3, configure=configure,
                     CAMERA_POLICY=configure()['camera_prediction'])
    exec(compile(source, __file__+':same_camera_inference_full_motion_state', 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
