# TASKS_ROUND9 — locate our error predictor on the information ladder

**Author: Claude (design + gates). Executor: Gemini 3.1 Pro (labour only).**
**Date: 7 Sep 2026. Banked: 79.462591 (`submission_SV2.zip`). No submission this round.**

---

## 0. RULES

1. **Report raw numbers. Do not interpret, do not decide, do not build a zip.**
2. **Never write a live score into `project_memory.md` unless it appears in the Codabench feed.**
3. Any step that fails or any file that is missing: **stop and report the exact error.** Do not
   substitute a different file, split, model or metric.
4. Scratch only under `/SML_DISK_24TB/rajeshr/Aryamann/UGP/`. Never `/tmp`, never the VM root.
5. Do not modify `submission_SV2.zip` or anything inside it. Extract to a working copy.

---

## 1. The one question this round answers

Our bounds half-widths come from a U-Net that predicts a per-element error magnitude, mapped
through a 24-bin LUT. **We do not know how much that predictor actually contributes.**

Measured already (§69.4), on honest `re_lohi` residuals, equal channel weight, for the
per-element quantity `g = mean[ exp(-(upper-lower)/σ) · 1{target ∈ [lower,upper]} ]`,
σ = `SIGMA_GLOBAL` = 0.0563870259:

| policy | g |
|---|---:|
| constant width at our shipped median | 0.5706 |
| best single global constant width | 0.6160 |
| **ORACLE per-element (h = \|e\|)** | **0.8455** |

**Where our ACTUAL per-element policy sits between those is unknown, and it decides the next
round.** Near 0.616 ⇒ the predictor contributes nothing and is worth rebuilding. Near 0.75 ⇒
the route is mostly spent.

---

## 2. Setup

### 2.1 Data — build from source, NOT from a residual cache
Use exactly `ftmv.py`'s loading so results are comparable to everything else in the record:
* `ftmv.py:34` → `local_harness/tr_meta.json` (`off`, `lens`, `names`)
* `ftmv.py:45` → `local_harness/tr_frames.npy` (`mmap_mode="r"`)
* `ftmv.py:38` → split `re_lohi`: hold out trajectories with Re ∈ {3750, 5025, 25425, 26700}
* `ftmv.py:46-52` → window starts; validation starts; then
  `rng = np.random.default_rng(0); va_sub = rng.choice(starts_va, size=900, replace=False)`

Each window is 40 frames: `[:20]` = input, `[20:]` = **target**. Channels are (u, v).

### 2.2 Model — the shipped artifact, unmodified
Extract `submissions/submission_SV2.zip` to `_r9/` and import its `submission.py`.
Call its `predict()`. Do not reimplement the bounds stack.

### 2.3 ⚠️ TWO TRAPS — both will silently corrupt the result

**Trap A — channel count.** `predict()` does `yb[..., 2] = 0.0`, so the input MUST have
**3 channels**. The frames have 2. **Pad a zero third channel** to shape
`(N, 20, 32, 64, 3)` before calling, exactly as `ftmv.py:63` does
(`z = np.zeros(w.shape[:-1]+(1,)); w = np.concatenate([w, z], -1)`).

**Trap B — the silent time-budget fallback.** `submission.py:67` sets `_TIME_BUDGET = 145.0`.
On timeout it **silently** replaces predictions with a repeat of the last input frame and sets
bounds to a CONSTANT half-width `[0.0129, 0.0098]`. This does not raise.
* Call `predict()` in **chunks of at most 250 windows** (each call resets its own clock).
* **After every call, assert the fallback did NOT fire:** the returned half-widths
  `(upper-lower)/2` must NOT be uniformly equal to 0.0129 (u) / 0.0098 (v). If more than 1% of
  a chunk's elements sit at exactly those values, **stop and report** — the numbers are void.

---

## 3. Compute these, on the scored element set only

`scored = (target != 0.0)` — same rule as `scoring.py:210`. All quantities below are means over
scored elements, computed per channel (u, v) and then reported both per channel and equally
weighted.

Let `h = (upper - lower)/2` (per element), `e = |target - prediction|` (per element),
`nil = (upper - lower)/σ`.

### 3.1 Our actual policy
* `g_ship = mean[ exp(-nil) · 1{lower ≤ target ≤ upper} ]`
* `coverage = mean[ 1{lower ≤ target ≤ upper} ]`
* mean and median `h`, per channel
* **`spearman(h, e)`** per channel — the direct quality measure of the predictor. Use Spearman,
  not Pearson: these residuals have kurtosis 26–34 and Pearson is meaningless on them.

### 3.2 The information ladder — the point of this round
All four use the **same elements**; the middle two use **our exact multiset of widths**, so the
width distribution is held fixed and only the *allocation* changes.

| name | how to build it |
|---|---|
| `g_shuffled` | randomly permute the `h` values across elements (within channel, seed 0), keep centres. **Destroys the predictor's information, preserves the width distribution.** |
| `g_ship` | as computed in 3.1 |
| `g_ranked` | sort elements by `e`, sort `h` ascending, assign largest `h` to largest `e` (within channel). **Perfect ranking with our exact widths.** |
| `g_oracle` | `mean[ exp(-2e/σ) ]` — h set to `e` exactly |

Report all four plus, for context, the best single constant width per channel
(`g_const` and its `h`), found by sweeping `h ∈ linspace(0.0005, 0.06, 3000)`.

★ **`g_shuffled` is the control that matters.** `(g_ship − g_shuffled)` is exactly what the
predictor buys. `(g_ranked − g_ship)` is what a perfect predictor would add at the same width
budget.

### 3.3 The real sps, for calibration
Also compute the true blended subscore via the kit, so the per-element numbers can be tied to a
real score: import `scoring.py` from `starting_kit_v9/realpde_t1_starting_kit_v9/` and call
`aggregate_sps(pred, target, c=2, lower=..., upper=...)`. Report `weighted`, `coverage`, and
`score_sps(weighted)`.

**Centres:** our intervals are deliberately NOT centred on the prediction (§30.11), so
`h` is a half-width but the interval is not `pred ± h`. Compute `inside` from the **actual
`lower`/`upper`**, never from `pred ± h`. Only `g_shuffled`/`g_ranked` re-place widths, and
they must keep each element's original interval **centre** `(lower+upper)/2`.

---

## 4. Output

Write `/SML_DISK_24TB/rajeshr/Aryamann/UGP/r9_results.json` and print a table:

```json
{"n_windows": 900, "n_scored_u": ..., "n_scored_v": ...,
 "per_channel": {"u": {"g_ship":..., "g_shuffled":..., "g_ranked":..., "g_oracle":...,
                       "g_const":..., "h_const":..., "coverage":..., "h_mean":..., "h_median":...,
                       "spearman_h_e":...}, "v": {...}},
 "equal_weight": {"g_ship":..., "g_shuffled":..., "g_ranked":..., "g_oracle":...},
 "kit_sps": {"weighted":..., "coverage":..., "score_sps":...}}
```

Save the script as `r9_probe.py`. Do not save any checkpoint or array.

---

## 5. Sanity checks to run and report (all cheap)

1. `g_shuffled ≤ g_ship ≤ g_ranked ≤ g_oracle` must hold. **If it does not, the harness is
   wrong — stop and report**, do not "fix" it by reordering.
2. `coverage` from 3.1 must match `kit_sps.coverage` to ~1e-6.
3. Report how many chunks were processed and confirm the Trap-B assertion passed on every one.

---

## 6. What NOT to do

* ⛔ No submission zip. No training. No retuning of the LUT, widths, or alpha.
* ⛔ Do not change `submission.py`, `_TIME_BUDGET`, `_BATCH`, or `_CHUNK_*`.
* ⛔ Do not substitute a residual cache for the source frames.
* ⛔ Do not delete any existing file.
* ⛔ In `project_memory.md`, append **only** the raw table under
  `## 70. ROUND 9 RAW RESULTS (Gemini, executed)` — no conclusions, no recommendations,
  no projected scores.
