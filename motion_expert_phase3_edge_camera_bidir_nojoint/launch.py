"""Same guarded fresh launch and source freeze; change only conditioning policy."""
import inspect
from motion_expert_phase3_edge_camera_directional_nojoint import launch as original
from .policy import configure

def main():
    source=inspect.getsource(original.main)
    old="original.main(experiment_package='motion_expert_phase3_edge_camera_directional_nojoint')"
    assert source.count(old)==1
    source=source.replace(old,"""from motion_expert_phase3_edge_camera_bidir_nojoint.launch import launch_native
    launch_native(original)""")
    source=source.replace("('conditioning-attention','directional')","('conditioning-attention','bidirectional')")
    old="['motion_expert_phase3_edge_image_joint_camera','motion_expert_phase3_edge_camera_directional_nojoint']"
    assert source.count(old)==1
    source=source.replace(old,"['motion_expert_phase3_edge_image_joint_camera','motion_expert_phase3_edge_camera_directional_nojoint','motion_expert_phase3_edge_camera_bidir_nojoint']")
    namespace=original.__dict__.copy();namespace['configure']=configure
    exec(compile(source,__file__+':guarded_launcher','exec'),namespace)
    namespace['main']()

def launch_native(launcher):
    # Existing directional replacement already holds the original handover
    # lease. A distinct lease permits this explicitly coordinated second
    # overlap; resident UID/start checks and the64GiB resource gate stay intact.
    source=inspect.getsource(launcher.main)
    old='suffix = ".handover.lock" if residents else ".lock"'
    assert source.count(old)==1
    source=source.replace(old,'suffix = ".handover.bidir_camera_nojoint.lock" if residents else ".lock"')
    namespace=launcher.__dict__.copy()
    exec(compile(source,__file__+':coordinated_overlap','exec'),namespace)
    namespace['main'](experiment_package='motion_expert_phase3_edge_camera_bidir_nojoint')

if __name__=='__main__':main()
