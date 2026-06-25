"""Profile a training step: reasoner H_R (64 seq forwards) vs MotionExpert fwd vs backward."""
import time, torch
from torch.utils.data import DataLoader
import flow
from motion_expert import MotionExpert
from reasoner import D_REASONER, FrozenReasoner
from uniego_dataset import UniegoTextMotionDataset, collate
from uniego_layout import FEAT_DIM

dev = "cuda"
reasoner = FrozenReasoner(dtype=torch.bfloat16, device=dev)
model = MotionExpert(kv_dim=D_REASONER, motion_dim=FEAT_DIM).to(dev)
ds = UniegoTextMotionDataset("/home/jungbin_cho/cosmos_motion_ft/motion_expert/pairs_train.jsonl", T=96, train=True)
dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4, collate_fn=collate, drop_last=True)
opt = torch.optim.AdamW(model.parameters(), lr=2e-4)


def sync(): torch.cuda.synchronize()


it = iter(dl)
for w in range(2):  # warmup
    b = next(it)
for trial in range(4):
    b = next(it)
    motion = b["motion"].to(dev); nj = b["neutral_joints"].to(dev); mpad = b["motion_pad_mask"].to(dev)
    sync(); t0 = time.time()
    with torch.no_grad():
        H_R, h_pad = reasoner.encode_text(b["caption"])
    sync(); t1 = time.time()
    H_R = H_R.float()
    sigma = flow.sample_sigma_logitnormal(64, dev)
    x_sigma, v_t, _ = flow.add_noise(motion, sigma)
    v_hat = model(x_sigma, sigma, H_R, h_pad, nj, motion_pad_mask=mpad)
    loss = ((v_hat - v_t) ** 2).mean()
    sync(); t2 = time.time()
    opt.zero_grad(); loss.backward(); opt.step()
    sync(); t3 = time.time()
    print(f"[{trial}] reasoner_H_R={t1-t0:.3f}s  expert_fwd={t2-t1:.3f}s  backward={t3-t2:.3f}s  "
          f"TOTAL={t3-t0:.3f}s  (reasoner={100*(t1-t0)/(t3-t0):.0f}%)")
