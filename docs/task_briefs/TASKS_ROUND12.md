# TASKS_ROUND12 — the U-Net backbone (the organizers' own best model on this dataset)

**Author: Claude (design + gates). Executor: Gemini 3.1 Pro (labour only).**
**Date: 7 Sep 2026. Banked: 79.462591. NO SUBMISSION THIS ROUND.**

---

## 0. RULES

1. **Report raw numbers. Do not interpret, do not decide, do not build a submission zip.**
2. **Never write a live score into `project_memory.md` unless it appears in the Codabench feed.**
3. Any failure or missing file: **stop and report the exact error.** Never substitute a different
   model, split, config, loss or metric.
4. Scratch only under `/SML_DISK_24TB/rajeshr/Aryamann/UGP/`. Never `/tmp`, never the VM root.
5. ⛔ **NO OUTSIDE WEIGHTS.** Port the *code* from `code/RealPDEBench/` freely — it is the official
   benchmark repo and is permitted. **Never download or load any checkpoint from HuggingFace**
   (`hzk17/DPOT`, `RealPDEBench-models`, anything else). That is disqualification, not a low score.
   Every weight must be trained by us, from the competition's own data release.
6. ⚠️ **ONLY GPUs 2 AND 3 ARE FREE** (checked 7 Sep: GPU 0 runs another user's
   `pravah2dgpu_serial.exe`, GPU 1 runs their `underPINN` job). **Use only
   `CUDA_VISIBLE_DEVICES=2` and `=3`.** Re-check `nvidia-smi` before starting and never kill or
   crowd another user's process. ⛔ Never `pkill -f <pattern>` where the pattern could match your
   own ssh command.
7. Dependencies `einops`, `einops_exts`, `rotary_embedding_torch` are ALREADY INSTALLED in
   `/SML_DISK_24TB/rajeshr/Aryamann/env` (done 7 Sep). Do not reinstall.

**DIVISION OF WORK — READ THIS CAREFULLY:**
* ⛔ **Claude is running Arm D himself on GPU 3. DO NOT RUN ARM D, and DO NOT TOUCH GPU 3.**
  Arm D is kept in this document only so you can see the full design.
* ✅ **You run Arms A, B, C — on GPU 2 ONLY**, sequentially in that order.
* **Report Arm A the moment it finishes; do not wait for B or C.** Arm A plus Claude's Arm D
  together already answer the question that matters (does the U-Net beat FNO on OUR airfoil).

---

## 1. Why (context, not opinion to act on)

RealPDEBench Table 1, Foil / Real-world Finetuning — the organizers' own benchmark, same rig and
task as this competition:

| model | params | Rel L2 | KE (= our tke) |
|---|---:|---:|---:|
| **U-Net (`Unet3d`)** | 23.0 M | **0.0145** | **0.00007** |
| **FNO (what we ship)** | 50.4 M | 0.0206 | 0.00009 |

−29.6% rel_l2 and −22% KE at half the parameters. Our rel_l2 is rank 50-of-50 in the top-50 room,
so this is aimed squarely at our binding constraint. **Their foil is NACA0025 (symmetric); ours is
NACA4418 (cambered) — so the ranking may or may not transfer. Arm D exists to test exactly that.**

---

## 2. Shared setup (identical across all arms — do not vary)

* **Data / split / windows:** exactly as `ftmv.py` — `local_harness/tr_frames.npy`, `tr_meta.json`,
  `split=re_lohi` (hold out Re 3750/5025/25425/26700), 3 channels (u, v, p≡0), 32×64.
* **Evaluation set:** the same 900-window `va_sub` (`rng = np.random.default_rng(0)`), so every
  number is directly comparable to §80–§82.
* **Normalisation:** the `ftmv.py` constants `MI/SI/MT/ST` (lines 31–33), unchanged.
* **Metrics:** the kit's `scoring.py` — report `rel_l2`, `tke`, `mvpe` as `score_error(...)`, plus
  `d_acc = 0.669*Δrel_l2 + 0.157*Δtke + 0.170*Δmvpe` measured **against the kit base**
  (rel_l2 **95.4738**, tke **75.8957**, mvpe **96.0945** — reproduce this to 0.001 as a harness check).
* **Evaluate every 500 updates** and print the full curve. I need the SHAPE, not just the best row.
* Keep the best checkpoint by `d_acc`, and separately report the best-by-RMSE step (their criterion).

⚠️ **THE `dim` TRAP — the single most likely way to waste this round.** `load_model.py:52` passes
`dim=input_shape[1]`, which at our 32×64 silently builds a ~6M model instead of the 23M one.
**Construct `Unet3d` directly and pass `dim=64` explicitly.** After building, **print the parameter
count and assert it is between 20M and 26M.** If it is ~6M, you have hit the trap — stop and report.

```python
from realpdebench.model.unet import Unet3d
model = Unet3d(dim=64, out_channels=3, dim_mults=[1,2,4], channels=3, in_time=20, out_time=20)
```
`einops`, `einops_exts` and `rotary_embedding_torch` are already installed in
`/SML_DISK_24TB/rajeshr/Aryamann/env` — do not reinstall. Run everything with that interpreter:
`/SML_DISK_24TB/rajeshr/Aryamann/env/bin/python`.

---

## 3. THE FOUR ARMS

### Arm A (GPU 2, FIRST) — U-Net, real data only
Organizers' foil U-Net recipe: **plain Adam (no weight decay), lr 1e-4, cosine to
`T_max=num_update`, batch 12, `num_update=10000`, MSE loss, `clip_grad_norm=0`.** Trained from
random init on real data only. Tag `r12_unet_real`.

### Arm B (GPU 3, SECOND) — U-Net, real only, + EMA
Identical to Arm A plus an EMA of the weights, decay **0.999**, evaluated from the EMA weights.
(`unet.py:120` already contains an unused `EMA` class — use it or a 5-line equivalent.)
Tag `r12_unet_ema`.

### Arm C (GPU 2, SECOND) — U-Net, SIM-pretrain then real fine-tune
The organizers' actual protocol, and the row that produced 0.0145.
1. **Sim pretrain** on `data/comp_sim/train_sim` with their domain randomisation:
   * multiplicative noise on **both input and target**: `x += x * randn_like(x) * 0.1`
   * **`mask_prob = 0.5`**: with prob 0.5 zero the `p` channel (this is what teaches the model to
     run with p≡0, which is the real-data condition — and the phase we have never run)
   * `sub_s_numerical = 2` to land on our 32×64 grid. 30,000 updates, batch 12, lr 1e-4, cosine.
2. **Fine-tune** the result on real data with the Arm A recipe (no noise, no masking — real `p` is
   already 0). Tag `r12_unet_sim2real`.
⚠️ `train_sim` is ~18 GB — **memory-map it (`mmap_mode="r"`)**. Loading it whole OOM-killed every
python on this box once before (§65.4).

### Arm D (GPU 3, FIRST) — CONTROL: FNO at the organizers' own settings
**This arm decides whether the benchmark's ranking transfers to NACA4418, and it is as important
as the U-Net arms.** Fine-tune from the kit checkpoint `data/comp_real/sim_real_fno.pth` using
their FNO config exactly: **`num_update=4000`, batch 32, lr 1e-4, cosine, plain Adam, MSE loss**
(NOT our `wtke` composite, NOT 30,000 steps). Tag `r12_fno_official`.
⇒ If Arm A/C beats Arm D on `rel_l2`, the U-Net finding transfers. If Arm D wins, it does not, and
we have instead learned our own FNO recipe is over-trained (§83.2).

---

## 4. What to report

For each arm: the full every-500-updates curve (`rel_l2`, `tke`, `mvpe`, `d_acc`, train loss), the
best row and its step, parameter count, and wall-clock.

Then a summary table against these existing honest reference points:
| reference | rel_l2 | tke | mvpe | d_acc |
|---|---:|---:|---:|---:|
| kit base (no fine-tune) | 95.4738 | 75.8957 | 96.0945 | 0.0000 |
| best single FNO member | 95.1541 | 77.7077 | 96.0153 | +0.0572 |
| `soup_v3` (4 members, honest) | 95.4027 | 80.6442 | 96.2280 | +0.7207 |
| `soup_v2` (SHIPPED, leaky on re_lohi) | 95.4385 | 81.4441 | 96.2948 | +0.8815 |

**Also measure inference time**: ms/window for the best U-Net vs the shipped FNO, batch 48,
interleaved, ≥5 repeats each, same GPU. The U-Net has temporal attention and may be slower;
`time` is a scored channel (≈0.032 final per 0.6 ms), so I need this to price it.

## 5. Gates — report, do not act on them

* **G1 (harness):** the kit base must reproduce 95.4738 / 75.8957 / 96.0945 to 0.001. If not, stop.
* **G2 (the trap):** U-Net parameter count in [20M, 26M]. If ~6M, stop and report.
* **G3 (the headline):** does any U-Net arm beat **rel_l2 95.4738** (the best rel_l2 anything of ours
  has ever achieved)? Report the margin.
* **G4 (transfer):** Arm A/C vs Arm D on `rel_l2` — does the benchmark's U-Net > FNO ranking hold
  on NACA4418?

## 6. What NOT to do

* ⛔ No submission zip, no live submission.
* ⛔ No downloaded checkpoints of any kind. Every weight trained by us.
* ⛔ Do not substitute `dim=input_shape[1]`. Do not change the split, the 900-window sample, the
  metrics, or the normalisation constants.
* ⛔ Do not add our `wtke` composite loss to any arm — all four arms use plain MSE, as the
  organizers do. (`wtke` is our invention and §68.4 shows it is already swept out.)
* ⛔ Append **only** raw tables to `project_memory.md` under
  `## 84. ROUND 12 RAW RESULTS (Gemini, executed)` — no conclusions, no projected live scores.
