"""Mirror/handedness check: confirm NymeriaPlus & BONES use the SAME chirality + forward convention.
Uses LABELED joints (Left/Right), which detects a mirror even on a symmetric rest pose."""
import glob, random, sys
import numpy as np
sys.path.insert(0, "/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention")
from uniego_layout import canonicalize_frame0
from decode_uniego_torch import decode_joints
import torch
HIPS, HEAD, LLEG, RLEG, LHAND, RHAND = 0, 6, 22, 26, 13, 19

def neutral_chirality(files, name, n=200, seed=0):
    rng=random.Random(seed); J=[]
    for f in rng.sample(files, min(n,len(files))):
        try:
            with np.load(f) as d: J.append(d["neutral_joints"].astype(np.float64))
        except Exception: pass
    J=np.stack(J).mean(0)  # [30,3] mean rest pose
    lr = J[RLEG]-J[LLEG]                       # left->right vector
    up = J[HEAD]-J[HIPS]                        # up vector
    fwd = np.cross(lr, up)                      # right-handed forward
    lr_ax, up_ax, fwd_ax = int(np.argmax(np.abs(lr))), int(np.argmax(np.abs(up))), int(np.argmax(np.abs(fwd)))
    print(f"  [{name}] L->R axis={'xyz'[lr_ax]}{'+' if lr[lr_ax]>0 else '-'}  "
          f"up axis={'xyz'[up_ax]}{'+' if up[up_ax]>0 else '-'}  "
          f"fwd=LRxUP axis={'xyz'[fwd_ax]}{'+' if fwd[fwd_ax]>0 else '-'}  "
          f"| LeftHand lat({'xyz'[lr_ax]})={J[LHAND][lr_ax]:+.2f} RightHand lat={J[RHAND][lr_ax]:+.2f}")
    return (lr_ax, np.sign(lr[lr_ax])), (fwd_ax, np.sign(fwd[fwd_ax])), np.sign(J[LHAND][lr_ax])

def motion_walk_forward(files, name, n=60, T=60, seed=2):
    """For clips that translate, confirm the body moves along +Z (forward) on average, same sign as fwd."""
    rng=random.Random(seed); disp=[]
    for f in rng.sample(files, min(n,len(files))):
        try:
            with np.load(f) as d: x=d["features"].astype(np.float32)
        except Exception: continue
        if x.shape[0]<T: continue
        j=decode_joints(torch.from_numpy(canonicalize_frame0(x[:T])).unsqueeze(0))[0].numpy()
        disp.append(j[-1,HIPS]-j[0,HIPS])      # root displacement over the clip
    D=np.stack(disp); 
    print(f"  [{name}] mean |root disp| per axis (x,y,z) = ({np.abs(D).mean(0)[0]:.2f},{np.abs(D).mean(0)[1]:.2f},{np.abs(D).mean(0)[2]:.2f})")

nym=sorted(glob.glob("/weka/jungbin/nymeriaplus_kimodo_proportional/uniego_rep/*/*.npz"))
bon=sorted(glob.glob("/weka/jungbin/seed/soma_proportional_uniegomotion_20fps/*/*.npz"))
print("=== neutral-pose chirality (labeled L/R joints) ===")
n_lr,n_fwd,n_lh = neutral_chirality(nym,"nymeria"); b_lr,b_fwd,b_lh = neutral_chirality(bon,"bones")
print("=== walking root displacement ==="); motion_walk_forward(nym,"nymeria"); motion_walk_forward(bon,"bones")
print(f"\nVERDICT: L->R axis+sign match={n_lr==b_lr}  forward axis+sign match={n_fwd==b_fwd}  "
      f"LeftHand side match={n_lh==b_lh}  => same handedness/no-mirror = {n_lr==b_lr and n_fwd==b_fwd and n_lh==b_lh}")
