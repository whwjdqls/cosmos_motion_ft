import sys, json, torch
sys.path.insert(0, "/home/jungbin_cho/cosmos_motion_ft")
sys.path.insert(0, "/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention")
from safetensors.torch import load_file
import train_motion_ft as T

# 1) remap coverage on the full key list (no model load needed)
snap = T.NANO_SNAPSHOT
km = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
tkeys = [k for k, v in km.items() if "transformer/" in v]
plain = [k for k in tkeys if ".self_attn.to_q." in k or ".self_attn.to_out." in k or "action_proj" in k]
for k in plain[:4] + [k for k in tkeys if "norm_q" in k and "added" not in k][:1]:
    print(f"  {k}  ->  {T._diffusers_to_net_key(k)}")

# 2) full load with the fixed remap
from train_motion_ft import build_network
out = build_network(tiny=False, dtype=torch.bfloat16, action_gen=True)
net = out[0] if isinstance(out, tuple) else out
print("net built; materializing + loading...")
T.materialize(net)
n_loaded = T.load_gen_weights(net, verbose=True)

# 3) numeric identity: reasoner q_proj layer 0 + action2llm.fc vs snapshot
sd = dict(net.named_parameters()); sd.update(dict(net.named_buffers()))
def snap_tensor(key):
    return load_file(f"{snap}/{km[key]}")[key]
for skey, nkey in [
    ("layers.0.self_attn.to_q.weight", None),
    ("layers.10.self_attn.to_out.weight", None),
    ("action_proj_in.fc.weight", None),
]:
    nkey = T._diffusers_to_net_key(skey)
    if nkey not in sd:
        cands = [n for n in sd if nkey.split(".")[-2] in n and ("layers.0." in n or "layers.10." in n or "action" in n)][:2]
        print(f"  MISS {skey} -> {nkey}; candidates: {cands}"); continue
    a = snap_tensor(skey).float(); b = sd[nkey].detach().float().cpu()
    print(f"  {skey} -> {nkey}: shape {tuple(a.shape)}=={tuple(b.shape)} maxdiff={(a-b).abs().max().item():.2e}")
