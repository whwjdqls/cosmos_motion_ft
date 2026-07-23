"""Prototype the corrected conversion (SOMA side): extract TRUE world joint
rotations, build rest->posed relative rotations in Y-up, save for the kimodo step.

G_rel[f,j]   = R_posed[f,j] @ R_rest[j]^T           # rest->posed rotation (world)
G_rel_y[f,j] = R_z2y @ G_rel[f,j]                   # rotate posed side Z-up->Y-up
neutral[j]   = rest world positions (pelvis-centered)
root_y[f]    = R_z2y @ hips_world_pos[f]
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
sys.path.insert(0, "/home/jungbin_cho/SOMA-X")
from soma.soma import SOMALayer
from soma.geometry.transforms import rotvec_to_matrix

SEQ = "/weka/jungbin/nymeriaplus/S04/20230816_s1_jeffery_bryant_act0_p5w199/body/xdata_soma.npz"
OUT = "/home/jungbin_cho/_proto_fix_arrays.npz"
R_Z2Y = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)  # (x,y,z)->(x,z,-y)

z = np.load(SEQ, allow_pickle=True)
poses = z["poses"].astype(np.float32); transl = z["transl"].astype(np.float32)
ic = z["identity_coeffs"].astype(np.float32); ts = z["timestamps_us"].astype(np.int64)

# 20fps subsample
t0, t1 = int(ts[0]), int(ts[-1]); nout = int((t1 - t0) / 1e6 * 20)
q = np.linspace(t0, t1, nout).astype(np.int64)
idx = np.unique(np.clip(np.searchsorted(ts, q), 0, len(ts) - 1))

layer = SOMALayer("/home/jungbin_cho/SOMA-X/assets", identity_model_type="smpl", device="cpu", mode="warp")
layer.prepare_identity(torch.from_numpy(ic).float())

# capture T_world from the skinning pose
cap = {}
orig = layer.batched_skinning.pose
def patched(*a, **k):
    k["return_transforms"] = True
    out = orig(*a, **k)
    cap["T"] = out[1].detach()
    return out
layer.batched_skinning.pose = patched

with torch.no_grad():
    layer.pose(torch.from_numpy(poses[idx]), transl=torch.from_numpy(transl[idx]), pose2rot=True)
    Tw_posed = cap["T"].numpy()                                  # (B,78,4,4)
    layer.pose(torch.zeros(1, 77, 3), transl=torch.zeros(1, 3), pose2rot=True)
    Tw_rest = cap["T"].numpy()                                   # (1,78,4,4)

R_posed = Tw_posed[:, 1:, :3, :3].astype(np.float64)             # (B,77,3,3) drop Root
R_rest = Tw_rest[0, 1:, :3, :3].astype(np.float64)               # (77,3,3)
N = Tw_rest[0, 1:, :3, 3].astype(np.float64)                     # (77,3) rest positions
hips_world = Tw_posed[:, 1, :3, 3].astype(np.float64)            # (B,3) hips = joint idx1

G_rel = np.einsum("bjmn,jno->bjmo", R_posed, np.transpose(R_rest, (0, 2, 1)))  # R_posed @ R_rest^T
G_rel_y = np.einsum("mn,bjno->bjmo", R_Z2Y, G_rel)               # R_z2y @ G_rel
root_y = hips_world @ R_Z2Y.T                                    # (B,3)
N_centered = N - N[0:1]                                          # pelvis-center (idx0 = Hips after drop)

np.savez(OUT, G_rel_y=G_rel_y.astype(np.float32), root_y=root_y.astype(np.float32),
         neutral=N_centered.astype(np.float32))
# sanity: rest head-foot of neutral
print("saved", OUT, "G_rel_y", G_rel_y.shape, "root_y", root_y.shape)
n77 = list(layer.rig_data["joint_names"])[1:]
h, lf, rf = n77.index("Head"), n77.index("LeftFoot"), n77.index("RightFoot")
print("neutral head-foot:", (N_centered[h] - 0.5 * (N_centered[lf] + N_centered[rf])).round(3))
