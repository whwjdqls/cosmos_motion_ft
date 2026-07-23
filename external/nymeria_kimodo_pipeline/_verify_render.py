import numpy as np, torch, imageio.v3 as iio, sys
from kimodo.skeleton import SOMASkeleton77
from kimodo.scripts.render_soma import render_single
s = SOMASkeleton77(); par = s.joint_parents.tolist(); names = list(s.bone_order_names)
lf, rf = names.index("LeftFoot"), names.index("RightFoot")
f = sys.argv[1]; start = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
d = np.load(f, allow_pickle=True)
lrm = torch.from_numpy(d["local_rot_mats"].astype(np.float32))
root = torch.from_numpy(d["root_positions"].astype(np.float32))
nj = torch.from_numpy(d["neutral_joints"].astype(np.float32))
B = lrm.shape[0]
_, posed, _ = s.fk(lrm, root, neutral_joints=nj.unsqueeze(0).expand(B, -1, -1))
p = posed.numpy()
print("foot Y over clip: min=%.3f mean=%.3f  (should be ~0, grounded)" % (p[:, [lf, rf], 1].min(), p[:, [lf, rf], 1].mean()))
NF = min(200, B - start)
seg = p[start:start + NF]   # consecutive, NO render-side grounding (data is grounded)
skip = [s.bone_index[n] for n in ("LeftHandThumbEnd","LeftHandMiddleEnd","RightHandThumbEnd","RightHandMiddleEnd") if n in s.bone_index]
frames = render_single(seg, par, caption="from converted NPZ (data grounded, 20fps)", max_frames=NF, skip_joints=skip, camera="follow")
out = "/home/jungbin_cho/_verify_" + f.split("/")[-1].replace(".npz", "") + ".mp4"
iio.imwrite(out, frames, fps=20, codec="libx264")
iio.imwrite(out.replace(".mp4", "_mid.png"), frames[NF // 2])
print("wrote", out)
