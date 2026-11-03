# Phase2 timestep-zero embedding diagnostic — 2026-11-02

Measured on sep-2 at **2026-11-02 00:47 UTC**, using the selected **Phase2 200k** specialist loaded by the current Phase3 experiments. The requested relative difference is **0.00947585959 = 0.947585959%**.

$$
\frac{\|E_M(0)-E_M(0.01)\|_2}{\|E_M(0.01)\|_2}
= \frac{1.451958229}{153.227072972}
= 0.00947585959.
$$

Here $E_M(t)$ means the **2048-D learned timestep embedding**, `heads.time_embedder(t)`: sinusoidal features followed by the checkpoint's trained SiLU MLP. It is not the full motion encoder or denoiser output. The input is normalized noise-time/sigma; pass 0, 0.01, and 0.05 directly, with no additional division by 1000.

| t | L2 norm of E_M(t) | Relative L2 distance from E_M(0), divided by norm of E_M(t) | Cosine with E_M(0) |
| --- | --- | --- | --- |
| 0 | 154.588566425 | 0 | 1 |
| 0.01 | 153.227072972 | 0.947585959% | 0.999994627504 |
| 0.05 | 147.760867746 | 4.930339719% | 0.999858695412 |

The 0.01→0.05 L2 distance is 5.833181078; divided by the norm at 0.05 it is 3.947717123%, with cosine 0.999908427007. Full vectors are saved in `embeddings_fp32.npz`.

**Interpretation:** the learned timestep embeddings at 0 and 0.01 are close in relative L2 and direction. This measurement alone does not establish equivalent denoiser predictions, reconstruction errors, or generation quality at those timesteps.

**Important for motion→video:** the current original, joint, control/FiLM, and video-projection Phase3 paths mask the timestep embedding out on clean conditioning motion. Their input follows:

$$
h_M(x,t,m)=W_Mx+b_M+e_{mod}+mE_M(t).
$$

For clean M2V condition frames, $m=0$. Therefore their motion input omits the timestep embedding; it does **not** add $E_M(0)$. The learned $E_M(0)$ is nonzero (norm 154.588566425). Removing it and evaluating it at zero are different operations. Whether that masking choice affects M2V quality requires a separate matched experiment; no training or inference behavior was changed here.

**Training support:** Phase2 samples `sigma = shift3(1 - sigmoid(Z)), Z ~ N(0,1)` and adds the time bias on valid target-motion frames. The continuous distribution has no point mass at zero. Also, 0.01 is an extremely rare tail reference: analytically P(sigma ≤ 0.01) = 6.214601436e-9 and P(sigma ≤ 0.05) = 2.638003929e-5. These are distribution probabilities, not observed counts from historical training.

**Checkpoint and verification**

- Checkpoint: `/mnt/local/jinhyung.kim/cosmos_storage/restored/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/motion_expert_t2m_edge/edge_7layer_self_cross_final_hidden_knorm_captiononly_nymeria_bones50_t2m_ti2m_nanoparity_gbs256_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260921/checkpoints/step_000200000.pt`.
- SHA256: `6a8d58d746560c0af3d340f8d711dcfe23ce163bb741912afcffa8ebfdbadc3a`.
- All 16 motion-head tensors loaded strictly; the four time-MLP parameters retain their checkpoint FP32 values.
- Native framework TimestepEmbedder class matches the checkpoint-pinned commit `d4599e2e43fbd06168e9884205b9b66c3902d8f6` exactly at the AST level.
- CPU FP32 native forward; scalar diagnostics accumulated in FP64. A hook on the actual `MotionHeads.encode_motion` call matched these embeddings bitwise. The clean mask suppresses the time contribution.
- Independent FP64 arithmetic gives relative L2 0.009475872860 for 0 vs 0.01; maximum FP32/FP64 embedding difference is 1.90e-6.
- Operation directory: `/mnt/local/jinhyung.kim/cosmos_storage/operations/phase2_timestep_embedding_20261102`. Contains `measure.py`, `run.sh`, `results.json`, `measurement.log`, `embeddings_fp32.npz`, and this report.
- Reproduce with `bash /mnt/local/jinhyung.kim/cosmos_storage/operations/phase2_timestep_embedding_20261102/run.sh`. CPU only; no GPU allocation, dataset scan, checkpoint modification, or training change.

Notion: https://app.notion.com/p/3fbd2a44f44b811a9d14e00c7b8bddc5
