"""CPU unit test for flow.add_noise_x0_masked (per-modality x0 objective). DIAGNOSTIC.

Checks (vs the spec + parity with add_noise_velocity_masked):
  1. tuple structure / shape / dtype parity with the velocity twin
  2. clean tokens (condition_mask=True) pass through EXACTLY (x_t == x0)
  3. TARGET is x0 itself (bit-identical), NOT eps - x0
  4. noised_mask == ~condition_mask
  5. explicit sigma respected (sigma=0 -> x_t==x0 everywhere; sigma=1 -> noised rows pure eps)
  6. default sigma ~ logit-normal(m=0, s=1) (the bs_train recipe)

Run (any torch env, CPU): python _verify_x0_noiser_cpu.py
"""
import torch

import flow

torch.manual_seed(0)
B, N, D = 4, 20, 7
x0 = torch.randn(B, N, D)
cm = torch.zeros(B, N, dtype=torch.bool)
cm[:, :5] = True                    # first 5 tokens clean everywhere
cm[2] = True                        # sample 2 fully clean (condition-only)

# --- 1. structure parity ---
out_v = flow.add_noise_velocity_masked(x0, cm)
out_x = flow.add_noise_x0_masked(x0, cm)
assert len(out_v) == len(out_x) == 4
for a, b, nm in zip(out_v, out_x, ("x_t", "t/sigma", "target", "noised_mask")):
    assert a.shape == b.shape, (nm, a.shape, b.shape)
    assert a.dtype == b.dtype, (nm, a.dtype, b.dtype)
print("[1] tuple/shape/dtype parity with add_noise_velocity_masked: OK")

x_t, sigma, target, noised = out_x
assert torch.equal(x_t[cm], x0[cm]), "clean tokens must pass through EXACTLY"
assert torch.equal(x_t[2], x0[2]), "fully-clean sample must be untouched"
print("[2] clean tokens pass through exactly (x_t == x0 on condition_mask): OK")

assert torch.equal(target, x0), "TARGET must be x0 itself"
# and provably NOT the velocity target: recover eps on the noised rows and compare.
sb = sigma.view(-1, 1, 1)
eps_rec = (x_t - (1.0 - sb) * x0) / sb.clamp(min=1e-6)
v_tgt = eps_rec - x0
diff_v = (target[~cm] - v_tgt[~cm]).abs().mean().item()
assert diff_v > 0.5, f"target suspiciously close to eps-x0 (mean|diff|={diff_v})"
print(f"[3] target is x0 (bit-identical); mean|x0 - (eps-x0)| on noised rows = {diff_v:.3f} "
      f"(clearly not the velocity target): OK")

assert torch.equal(noised, ~cm)
print("[4] noised_mask == ~condition_mask: OK")

s0 = torch.zeros(B)
xt0, s_out, tgt0, _ = flow.add_noise_x0_masked(x0, cm, s0)
assert torch.equal(xt0, x0) and torch.equal(s_out, s0) and torch.equal(tgt0, x0)
s1 = torch.ones(B)
xt1, _, _, _ = flow.add_noise_x0_masked(x0, cm, s1)
nz = xt1[~cm]
assert abs(nz.mean().item()) < 0.15 and abs(nz.std().item() - 1.0) < 0.15
print(f"[5] explicit sigma respected: sigma=0 -> x_t==x0; sigma=1 -> noised rows pure eps "
      f"(mean={nz.mean():.3f} std={nz.std():.3f}): OK")

big = 20000
sig = flow.add_noise_x0_masked(torch.randn(big, 1, 1),
                               torch.zeros(big, 1, dtype=torch.bool))[1]
lo = torch.logit(sig.clamp(1e-6, 1 - 1e-6))
assert abs(lo.mean().item()) < 0.05 and abs(lo.std().item() - 1.0) < 0.05
print(f"[6] default sigma ~ logit-normal: mean(logit)={lo.mean():.4f} std(logit)={lo.std():.4f} "
      f"(expect ~0 / ~1): OK")

print("\nALL CPU UNIT CHECKS PASS (add_noise_x0_masked)")
