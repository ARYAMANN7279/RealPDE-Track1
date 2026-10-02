# SHARED CONTEXT — RealPDE Track 1 research agents (written 11 Sep 2026 by the coordinator)

**Read this whole file before doing anything. It is short on purpose; every rule in §4–§6 has
already cost us a live submission slot or produced a silently wrong number.**

## 0. Goal and authority
* NeurIPS 2026 RealPDE Competition, Track 1 (Sim2Real airfoil PIV), Codabench #17363, team `aryamannsr`.
* **Banked live score: 79.484440** (`submissions/submission_SCREEN.zip`). **Goal: cross 79.5, then 80.**
  Leaderboard top ≈ 81.82. Phase closes 27 Sep 2026. One live submission per UTC day.
* ⛔ **Agents NEVER submit.** Aryamann submits. The coordinator verifies every candidate first.
* Full decision record: `$B/project_memory.md` (~9,500 lines) — **READ-ONLY for you.** Grep it for the
  sections your brief names; do not read it end to end. **Never edit it** (parallel agents would
  corrupt it). The coordinator consolidates your reports into it.

## 1. Scoring (weights solved from the board, max residual 0.005)
```
final = 0.46743*rel_l2 + 0.10027*tke + 0.09420*mvpe + 0.09689*time + 0.24737*sps + 0.9119
rel_l2 / tke / mvpe score = 100 / (1 + 0.5*err)          (u, v only)
time  = 100 / (1 + sqrt(t / 0.72896))   t = mean neural inference wall time per sample, seconds
sps   = 100 * sum_b w_b * mean[(1 - pm_b) * exp(-nil) * inside]      w = 0.5 DM, 0.3 TKE, 0.2 MVPE
        pm = e/(0.5+e) per window per branch;  nil = (upper-lower)/0.0563870;  inside = lower<=target<=upper
        default band if no bounds: pred +/- 0.05|pred|.  Non-finite or reversed bounds => ALL scores 0.
```
`scoring.py` in the starting kit reproduces the five subscores exactly. 5-minute wall-clock run limit.

## 2. Where the gap is — live leaderboard, 11 Sep
| | final | rel_l2 | tke | mvpe | time | sps |
|---|---:|---:|---:|---:|---:|---:|
| **us (banked)** | **79.484** | 94.13 | **76.43** | 93.16 | 90.46 | **37.92** |
| rank 1 skabob | 81.824 | 94.79 | 79.19 | 93.86 | 92.29 | 44.00 |
| rank 10 csasaa | 81.430 | 94.64 | 79.55 | 93.81 | 90.25 | 43.34 |
| rank 50 amalss | 79.886 | 94.38 | 79.37 | 93.60 | 85.88 | 39.48 |
* Gap to rank 10 = +1.94 final: **sps 69%**, tke 16%, rel_l2 12%, mvpe 3%, time −1%.
* **Our sps AND our tke are the worst in the entire top 50.** Every top-50 team has sps ≥ 38.76, tke ≥ 76.82.
* sps ≈ 100·W·E. W = accuracy part `(1−pm)`: us 0.689 vs skabob 0.715. **E = bounds part
  `inside·exp(−nil)`: us 0.550 vs skabob 0.616 — ~75% of our sps gap to skabob is E.**
* **Time:** our ≈8.1 ms/sample (90.46); zhoubojian 93.33 ≈ 3.7 ms/sample. A true 2× speedup ≈ +0.25 final.

## 3. The banked artifact — fully decomposed (§133/§134, verified bit-exact)
```
banked backbone = 0.125*ft_all_w15_lr1 + 0.125*ft_all_w15_lr3 + 0.125*ft_all_w33_lr1 + 0.125*ft_all_w33_lr3
                + 0.5*sv3_3e5_10
```
* `ft_all_*` = `local_harness/finetune_all.py`: bs 8, OneCycle pct_start 0.1, 6000 steps, ALL 81 real
  trajectories, fixed `rng(0)`. Naming: `wNN` = wtke 0.NN, `lrN` = N×1e-5.
* `sv3_3e5_10` = `train_es/ftaug_sv3_3e5_10.pth`: lr 3e-5, wtke 0.10, 16000 steps, split none, seed 206.
* All start from the organizers' kit FNO `data/comp_real/sim_real_fno.pth` (modes 4/12/16, 4 layers, **width 64**, padding 6 — the '100.7M params' figure counts real+imag of 50.33M complex elements; §146.3).
* Bounds: ONE small U-Net emits lower/upper from the input window (`bounds_assets.npz`, 186 keys).
  Inference code: `submission.py` inside the zip. Template for any backbone swap: `submission_SV2.zip`;
  exact packing code (complex via `view_as_real`, fp16): `$B/r50_build.py`.
* Harness: `$B/r12_eval.py` (`selftest()` must PASS: kit base = rel_l2 95.4738 / tke 75.8957 /
  mvpe 96.0945). Trainers: `$B/r45_train.py` (splits re_lohi/every5/none…), `local_harness/finetune_all.py`.

## 4. ⛔⛔ RULES THAT COST US LIVE SLOTS — non-negotiable
1. **The local ruler is LEAKY.** It scores 900 `re_lohi` windows; for any 100%-data model those windows
   are TRAINING data.
2. **Within-class comparisons only.** A 100%-data candidate vs the 100%-data banked artifact: the local
   SIGN has been right 2/2 (SCREEN local +0.027 → live +0.022; RECIPE2 local −0.140 → live −0.064,
   magnitude ≈0.5×). **Blending any model trained on a different split (every5 / every9 / re_lohi) into
   the banked artifact is INVALID** — `W73`: local +0.215 → live −0.043.
3. **If the local monitor says WORSE, it IS worse.** Never argue past a local-negative sign (§136).
4. **Magnitude law (§130):** local-predicted gains ≥ 0.20 have ALWAYS realised NEGATIVE; ≤ 0.07 realised
   positive at 0.3–0.8×. All 7 forecasts ever made were optimistic. **Report local Δ; do not forecast live.**
5. **cos0 gate:** any blend/soup ingredient whose TKE-map cosine `cos0` > 0.8734 is a memoriser — reject.
6. **Honest-holdout gains do NOT survive promotion to 100% data** (§136: lr 3e-6 won honestly, lost live).
7. **A concern you write down blocks the candidate until a MEASUREMENT resolves it — not an argument.**
8. Pricing a local Δ: effective final per LOCAL point: rel_l2 **0.6133**, tke **0.1628**, mvpe **0.1700**;
   sps local→live slope 1.4385 (× weight 0.24737).

## 5. Technical traps (each already produced a plausible-looking wrong number)
* FNO spectral weights are **complex**. `.float()` silently drops the imaginary part (kit scores 86.39
  instead of 95.47). Average/scale complex tensors as complex.
* `bns.*.num_batches_tracked` are **float32** BN counters: COPY, never average. Match on NAME, not dtype.
* The fp32 SpectralConv wrap lives inside `predict()`; driving the model directly can raise cuFFT errors.
* `OneCycleLR(total_steps=--steps)`: a different `--steps` is a different schedule. A run killed mid-schedule
  is unusable.

## 6. Operations — VM `ssh vm`, `$B=/SML_DISK_24TB/rajeshr/Aryamann/UGP`
* Python: `/SML_DISK_24TB/rajeshr/Aryamann/env/bin/python`
* Launch ONLY as: `setsid nohup env CUDA_VISIBLE_DEVICES=<g> $P script.py … < /dev/null > log 2>&1 & disown`
* Liveness: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` or `ps -p <PID>`.
  ⛔ NEVER `pgrep -f` / `pkill -f` — they match your own ssh command and have killed our sessions.
* ⛔ NEVER write to `/tmp` or the VM root partition (90% full; it once truncated a checkpoint that still
  passed `unzip -t`). Your scratch: `$B/agents/<your_name>/`.
* Shared box: other users' jobs appear on GPUs. Never touch another user's process. Never reboot.
* **THERMAL:** before every launch run
  `nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,clocks_throttle_reasons.sw_thermal_slowdown --format=csv,noheader`.
  If your GPU is ≥ 90 °C or SW slowdown is Active, do not launch; wait. **Use ONLY your assigned GPU(s).**
  If another user occupies your GPU, stop and report — do not move to someone else's.
* ssh drops happen. "PID gone + `[done]` in the log" = finished. "PID gone, no `[done]`" = killed.
* Quote safety: write Python locally and `rsync` it to the VM; never embed quoted Python in `ssh '…'`.
* ⛔ **No outside pretrained weights** (HuggingFace model checkpoints etc.) = disqualification. Official data
  and the RealPDEBench *code* are fine.

## 7. CLOSED — do not re-propose without NEW evidence
| area | verdict | § |
|---|---|---|
| bound-width tightening / reallocation | 4 slots lost; live transfer ≈0; reallocation ceiling +2.2% | §53 §73 §79 §94 |
| every knob in the zip (alpha, mh_alpha, cw, LUT) | at optimum | §80 §95 |
| blends with leaky partners / composition screens | screen = leakage meter, r²=0.99 with cos0 | §127 §128 |
| backbone swaps (U-Net, F-FNO) | pretraining moat + time | §88 §90 |
| 2nd full fp16 backbone (MoE/ensemble) | 167% of the 256 MiB cap | §101.1 |
| EMA / dropout / weight decay | gain < sps disturbed | §107 §112 |
| spectral-power (BSP), AMSE, amplitude fixes, DMD | closed (closed-form / dose-response / oracle) | §49 §114 §117 |
| post-hoc residual correctors | ≤5% of oracle | §61 |
| mirror TTA, Lie augmentation, "more data" | — | §65.3 §83.5 |
| lower-lr recipe on 100% data (RECIPE, RECIPE2) | live −0.064 | §136 |

## 8. Deliverables
* ⚠️ The harness BLOCKS subagents from writing report files. **Deliver your report as your FINAL MESSAGE TEXT** (scripts/logs on the VM via ssh are fine). It must contain: what you tried, **raw numbers**, local Δ per channel
  vs banked, verdict. Honest negatives are valuable — say "this failed" plainly.
* You may build a candidate zip **only** if it is within-class and local-POSITIVE vs banked on the validated
  harness. Name it `submissions/submission_<AGENT>_<tag>.zip`. Never overwrite or modify an existing zip;
  never delete checkpoints.
* End your final message with: the single best candidate (or "none"), its zip md5 and backbone md5, and its
  per-channel local Δ vs banked.
