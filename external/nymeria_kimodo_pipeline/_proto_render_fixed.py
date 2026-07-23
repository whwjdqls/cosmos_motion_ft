import numpy as np, torch, imageio.v3 as iio
from kimodo.skeleton import SOMASkeleton77
from kimodo.skeleton.transforms import global_rots_to_local_rots
from kimodo.scripts.render_soma import render_single

s77 = SOMASkeleton77(); par = s77.joint_parents.tolist()
names = list(s77.bone_order_names)
lf, rf = names.index("LeftFoot"), names.index("RightFoot")
ltoe = names.index("LeftToeBase") if "LeftToeBase" in names else lf
rtoe = names.index("RightToeBase") if "RightToeBase" in names else rf

d = np.load("/home/jungbin_cho/_proto_fix_arrays.npz")
G = torch.from_numpy(d["G_rel_y"]); root = torch.from_numpy(d["root_y"]); neutral = torch.from_numpy(d["neutral"])
local = global_rots_to_local_rots(G, s77); B = local.shape[0]
_, posed, _ = s77.fk(local, root, neutral_joints=neutral.unsqueeze(0).expand(B, -1, -1))
p = posed.numpy()

# CONSECUTIVE window for real-time 20fps (not subsampled across the whole clip)
START, NF = 4000, 200            # 10 s at 20 fps
seg = p[START:START + NF].copy()  # (NF,77,3)

# GROUND it: shift Y so the lowest foot in the window sits on the floor (y=0)
foot_y_min = seg[:, [lf, rf, ltoe, rtoe], 1].min()
seg[..., 1] -= foot_y_min
print(f"window {START}:{START+NF}  foot_y_min shifted by {foot_y_min:.3f} m")

skip = [s77.bone_index[n] for n in ("LeftHandThumbEnd", "LeftHandMiddleEnd",
        "RightHandThumbEnd", "RightHandMiddleEnd") if n in s77.bone_index]
frames = render_single(seg, par, caption="jeffery CORRECTED (real-time 20fps, grounded)",
                       max_frames=NF, frame_stride=1, skip_joints=skip, camera="follow")
iio.imwrite("/home/jungbin_cho/_proto_fixed_realtime.mp4", frames, fps=20, codec="libx264")
iio.imwrite("/home/jungbin_cho/_proto_fixed_realtime_mid.png", frames[NF // 2])
print("wrote /home/jungbin_cho/_proto_fixed_realtime.mp4  (%d frames, %.1fs)" % (NF, NF / 20))
