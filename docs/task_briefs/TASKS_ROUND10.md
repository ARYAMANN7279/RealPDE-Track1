# TASKS_ROUND10 — rebuild the per-element error-magnitude predictor

**Author: Claude (design + gates). Executor: Gemini 3.1 Pro (labour only).**
**Date: 7 Sep 2026. Banked: 79.462591. NO SUBMISSION THIS ROUND.**

---

## 0. RULES

1. **Report raw numbers. Do not interpret, do not decide, do not build a submission zip.**
2. **Never write a live score into `project_memory.md` unless it appears in the Codabench feed.**
3. Any failure or missing file: **stop and report the exact error.** Never substitute a
   different file, split, model, loss or metric.
4. Scratch only under `/SML_DISK_24TB/rajeshr/Aryamann/UGP/`. Never `/tmp`, never the VM root.
5. Check `nvidia-smi` and pick an idle GPU with `CUDA_VISIBLE_DEVICES=N`.
6. Do not modify `submission_SV2.zip` or the shipped `bounds_assets.npz`. Work on copies.

---

## 1. What this round does, and the ONE rule that makes it safe

§71: our error predictor captures only **29.5%** of the floor→oracle range; **26.9% of `g`
headroom remains**. Capturing half the range is worth **+0.735 final (+0.367 at a 50% haircut)**
— more than the +0.29 top-50 gap.

**Four live slots (LUTFIX, WIDE125, arcsinh, LUTCAL) were lost by changing the width POLICY —
the global scale/shape, which §53.3 proved unidentifiable without live anchors.**

★★ **THE INVARIANT FOR THIS ROUND: the width DISTRIBUTION must stay exactly the shipped one.**
We only change *which element gets which width*. The global calibration is therefore untouched
**by construction**, not by argument. Every number below is computed under that constraint.
⛔ **If any step wants to make widths globally narrower or wider, that step is wrong.** The local
surface will ask for this (§71.5: it says "tighten u by 44%"). It is the §69.2 illusion. Refuse it.

---

## 2. Definitions (use these exactly)

σ = `SIGMA_GLOBAL` = 0.0563870259. Per element, per channel ch ∈ {u, v}:
* `c_i` = interval centre = `(lower_i + upper_i)/2` from the shipped artifact
* `h_i` = shipped half-width = `(upper_i - lower_i)/2`
* `d_i` = `|target_i - c_i|`  ← **distance to the interval CENTRE, not to the prediction**
* `g = mean[ exp(-2h/σ) · 1{d ≤ h} ]` over scored elements (`scored = target != 0.0`)

**Quantile matching (the invariant).** Given any new per-element score `s_i`, define
`h_new` by: sort `s` ascending, sort the shipped multiset `{h_i}` ascending, assign in order
(largest `s` gets largest `h`). The resulting width distribution is **identical** to the shipped
one by construction. Intervals are `[c_i - h_new_i, c_i + h_new_i]` — **centres are never changed.**

---

## 3. Setup

* Data / split / windows: exactly as `TASKS_ROUND9.md §2.1` (`ftmv.py` loading, `re_lohi` split,
  `rng=np.random.default_rng(0)`, `va_sub` 900 windows for evaluation).
* **Training windows = trajectories NOT in `re_lohi`. Evaluation windows = `re_lohi` only.**
  Never train on an evaluation trajectory.
* **Predictions come from `train_es/soup_v3_fp16.pth`** (md5 `b3255e22…`), the honest soup that
  genuinely held out `re_lohi`. Not `soup_v2` — its `re_lohi` residuals are leaky-small.
* Reuse `r9_probe.py`'s harness for `g`, including **Trap A (pad a zero 3rd channel)** and
  **Trap B (chunks ≤250, assert the `[0.0129, 0.0098]` constant-width fallback never fires)**.

⚠️ **Known mismatch, stated so you do not "fix" it:** on training trajectories `soup_v3`'s
residuals are in-sample and therefore smaller than on `re_lohi`. This is tolerable **only
because we quantile-match** — the head needs its *ranking* to transfer, not its scale. Do not
attempt to rescale, reweight or otherwise correct for it.

---

## 4. TASK A — retrain the width head only (zero time cost, low capacity)

The shipped bounds net computes `o = net.out(net.trunk(ui))`, then `c = o[:, :40]` (centre) and
`w = o[:, 40:]` (width score → LUT). In `bounds_assets.npz` that is `w_out.weight (80,96,1,1)`
and `w_out.bias (80,)`. **Only rows 40:80 feed the width.** The `e0_*`/`e1_*` nets feed the
centre only and must not be touched.

* **Freeze everything except `w_out.weight[40:80]` and `w_out.bias[40:80]`.**
* Train those to predict `log(|target - prediction| + 1e-6)` per element, per channel,
  **Huber loss** (delta = 1.0), AdamW, lr 1e-3, weight decay 1e-6, batch 16 windows,
  8000 steps, cosine schedule, seed 1234.
* **Architecture, parameter count and inference path are unchanged ⇒ `time` is invariant by
  construction.** Do not add, remove or resize any module.

## 5. TASK B — a dedicated width U-Net (higher capacity, small time cost)

Train a NEW U-Net with **the same architecture as the existing `w_*` trunk** (same channel
widths: 96/192/384, `pe1..pe3`, `pb`, `pd3..pd1`, `pout`), input `ui` (80 channels), output
**40 channels** (width score only). Same optimiser/loss/schedule as Task A but train all its
weights, 20000 steps.

The shipped net stays in place for centres; this net replaces the width computation.
**This adds one U-Net forward to the inference path.** Report its measured cost (see §7.3).

## 6. TASK C — reporting only, no training

Report, for each of `train_es/soup_honest.pth`, `train_es/soup_lohi_honest.pth`,
`train_es/joint_soup_aoa15.pth`, `train_es/soup_v3.pth`: **which trajectories/conditions it held
out** (read the training script or its `.json` sidecar; if it cannot be determined, say so).
I need this to design a fold scheme that gives genuinely out-of-sample training residuals.
**Do not train anything for this task.**

---

## 7. What to compute and report

### 7.1 The `g` table, on the `re_lohi` evaluation windows, per channel and equal-weight
For each of: **shipped** (baseline), **Task A**, **Task B** — all under quantile matching:

| quantity | note |
|---|---|
| `g` | the headline |
| `g / g_shipped` | the ratio that decides everything |
| `spearman(h_new, d)` | ranking quality, Spearman **not** Pearson (kurtosis 26–34) |
| `coverage` | mean `1{d ≤ h}` |
| KS distance between `h_new` and shipped `h` distributions | **must be ~0 by construction — if it is not, quantile matching is broken; stop and report** |

### 7.2 The corrected fixed-budget ceiling (fixes §71.1's invalid `g_ranked`)
Recompute `g_ranked` **ranking by `d`, not by `|target - prediction|`**: sort `d` ascending,
sort shipped `h` ascending, assign in order. Report per channel.
**Sanity: `g_ranked ≥ g_shipped` must now hold in BOTH channels** (in §70 it failed on u,
because the wrong variable was used). If it still fails, stop and report — do not reorder.
Note this is a heuristic assignment, so report it as a *ceiling estimate*, not a proven bound.

### 7.3 Deployable form and its cost
* Refit the **24-bin LUT** to map the new score to a half-width such that the output width
  distribution reproduces the shipped marginal (match the 24 bin quantiles). Report `g` under
  this deployable LUT alongside the exact-quantile `g` — the gap is the deployment loss.
* **Task B only:** time the inference path with and without the extra U-Net, interleaved,
  ≥5 repeats each, same GPU, report median ms/window both ways.

### 7.4 Output
`/SML_DISK_24TB/rajeshr/Aryamann/UGP/r10_results.json` + a printed table. Save checkpoints as
`train_es/r10_taskA.pth` and `train_es/r10_taskB.pth`. Scripts `r10_train.py`, `r10_eval.py`.

---

## 8. Gates — report, do not act on them

Let `R = g_TaskX / g_shipped` on the `re_lohi` evaluation set under exact quantile matching.

* `R < 1.02` → the route is closed.
* `1.02 ≤ R < 1.05` → marginal.
* **`R ≥ 1.05` → interesting. `R ≥ 1.08` → this clears the top-50 gap at a 50% haircut.**

Also report `R` under the deployable LUT (7.3), which is the number that would actually ship.

---

## 9. What NOT to do

* ⛔ No submission zip. No live submission.
* ⛔ **Do not change the width distribution, the LUT scale, `alpha`, `cw`, or any centre.**
* ⛔ Do not touch `e0_*` / `e1_*` / `mh_*` weights.
* ⛔ Do not train or evaluate on `re_lohi` trajectories.
* ⛔ Do not use `soup_v2` predictions for training or for the headline evaluation.
* ⛔ Do not "improve" the metric, the split, the loss or the window sample.
* ⛔ In `project_memory.md`, append **only** the raw tables under
  `## 72. ROUND 10 RAW RESULTS (Gemini, executed)` — no conclusions, no recommendations,
  no projected live scores.
