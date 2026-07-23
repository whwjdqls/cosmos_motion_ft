import numpy as np, torch, imageio.v3 as iio
from kimodo.skeleton import SOMASkeleton77
from kimodo.skeleton.transforms import global_rots_to_local_rots
from kimodo.scripts.render_soma import render_single

s77 = SOMASkeleton77()
par = s77.joint_parents.tolist()
names = list(s77.bone_order_names)
h, lf, rf = names.index("Head"), names.index("LeftFoot"), names.index("RightFoot")

d = np.load("/home/jungbin_cho/_proto_fix_arrays.npz")
G = torch.from_numpy(d["G_rel_y"])          # (B,77,3,3) per-joint global (rest->posed, Y-up)
root = torch.from_numpy(d["root_y"])        # (B,3)
neutral = torch.from_numpy(d["neutral"])    # (77,3) Y-up rest, centered

# kimodo local rotations from global, then FK with the rest neutrals
local = global_rots_to_local_rots(G, s77)
B = local.shape[0]
_, posed, _ = s77.fk(local, root, neutral_joints=neutral.unsqueeze(0).expand(B, -1, -1))
p = posed.numpy()

sel = np.linspace(0, B - 1, 100).astype(int)
hmf = (p[sel, h] - 0.5 * (p[sel, lf] + p[sel, rf])).mean(0)
print("CORRECTED kimodo FK head-foot mean (xyz):", hmf.round(3),
      "-> up", "xyz"[int(np.argmax(np.abs(hmf)))], "dist=%.2f" % np.linalg.norm(hmf))

# render a clip with the follow camera (drop fingertip ends like training viz)
skip = [s77.bone_index[n] for n in ("LeftHandThumbEnd", "LeftHandMiddleEnd",
        "RightHandThumbEnd", "RightHandMiddleEnd") if n in s77.bone_index]
vsel = np.linspace(0, B - 1, 150).astype(int)
frames = render_single(p[vsel], par, caption="jeffery CORRECTED (Y-up, kimodo FK)",
                       max_frames=150, skip_joints=skip, camera="follow")
iio.imwrite("/home/jungbin_cho/_proto_fix_jeffery.mp4", frames, fps=20, codec="libx264")
# also one mid still
iio.imwrite("/home/jungbin_cho/_proto_fix_jeffery_mid.png", frames[len(frames) // 2])
print("wrote /home/jungbin_cho/_proto_fix_jeffery.mp4 and _mid.png")
