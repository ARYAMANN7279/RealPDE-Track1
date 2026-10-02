# TASKS_ROUND11 — refit the LUT on live-emulated residuals (the 3rd-anchor probe)

**Author: Claude (design + gates). Executor: Gemini 3.1 Pro (labour only).**
**Date: 7 Sep 2026. Banked: 79.462591. BUILD A ZIP, DO NOT SUBMIT.**

---

## 0. RULES

1. **Report raw numbers. Do not interpret, do not decide whether to submit.** Aryamann spends
   the slot; I gate the decision. Build the artifact, hand back the tables.
2. **Never write a live score into `project_memory.md` unless it appears in the Codabench feed.**
3. Any failure, missing file, or gate breach in §6: **stop and report the exact error.** Never
   substitute a different file, model, split, map or metric.
4. Scratch only under `/SML_DISK_24TB/rajeshr/Aryamann/UGP/`. Never `/tmp`, never the VM root.
5. Do not modify `submission_SV2.zip` or its `bounds_assets.npz`. Work on copies.

---

## 1. Why this round exists

§73.2: **91.7% of the remaining sps headroom is in the width DISTRIBUTION**, not in reallocating
widths (that ceiling is +2.2% and we hold 97.8% of it). The distribution channel has cost four
live slots — LUTFIX, WIDE125, arcsinh, LUTCAL — every one by trusting a **local** optimum.

§53.3 fitted a heavy-tail error map from the two anchors those slots bought:

```
e_real = a · med · (e_local / med)^b        a = 1.1446,  b = 1.80
```

then shelved it as untestable (2 params, 2 anchors, 0 DOF). **It has never been used to build a
policy.** §69.1 independently corroborates its qualitative form from raw residuals (kurtosis
26.6 / 34.0), which it did not have when it was shelved.

**This round refits the LUT for the true SPS objective on residuals pushed through that map.**
The whole point is that the refit is done on a LIVE-EMULATED error distribution, not a local one.
That single change is what separates this from LUTCAL, which refit on local residuals and lost.

⛔ **`ED` (bin edges) stays EXACTLY as shipped. Only the 24×2 `LUT` half-widths change.**
Nothing else in the artifact changes ⇒ `time` is invariant by construction and the centre policy,
`alpha`, `cw` and `mh` are untouched.

---

## 2. Definitions

σ = `SIGMA_GLOBAL` = 0.0563870259. Per element, per channel ch ∈ {u, v}:
* `c` = interval centre = `(lower + upper)/2` from the shipped artifact
* `d` = `|target − c|` — **distance to the interval CENTRE** (not to the prediction)
* objective for a bin B at half-width h: `G_B(h) = mean_{i∈B}[ exp(−2h/σ) · 1{d_i ≤ h} ]`

**The emulation map**, applied to `d` per channel (`med` = median of `d` in that channel):

```
d_emul = a · med · (d / med) ** b            a = 1.1446
```

---

## 3. Setup

* Data / split / windows / harness: exactly as `TASKS_ROUND9.md §2.1` and `r9_probe.py`,
  including **Trap A** (pad a zero 3rd channel) and **Trap B** (chunks ≤ 250; assert the
  `[0.0129, 0.0098]` constant-width fallback never fires).
* Predictions from `train_es/soup_v3_fp16.pth` (md5 `b3255e22…`) — the honest `re_lohi` holdout
  model. Evaluate on `re_lohi` windows only.
* Read shipped `LUT (24,2)` and `ED (2,23)` from `submission_SV2.zip`'s `bounds_assets.npz`
  (md5 `6e7a6290…`).
* Also read LUTCAL's LUT from `submissions/submission_LUTCAL.zip` (`bounds_assets.npz`
  md5 `70ee0010…`) — needed for gate **G1**.

---

## 4. TASK 1 — build the refit LUTs

Assign every scored element to its bin with the **shipped** `ED`, via
`np.digitize(w, ED[ch], right=True)` (equivalently `torch.bucketize(..., right=True)`).

For each channel and each of the 24 bins, and for each `b` in **{1.0, 1.4, 1.8, 2.2}**:

* map `d → d_emul` using §2 with that `b` (b = 1.0 is the pure-scaling control, i.e. the
  LUTCAL-style local refit)
* choose `h_bin = argmax_h G_B(h)` over `h` on a dense grid `np.linspace(1e-4, 0.12, 4000)`
* record `h_bin`, the bin count `n`, and `G_B(h_bin)`

⚠️ Bins with `n < 200`: **do not refit them — copy the shipped value** and say which bins those
were. A 24-bin argmax on a handful of elements is noise.

Produce four LUTs: `LUT_b100`, `LUT_b140`, `LUT_b180`, `LUT_b220`.

## 5. TASK 2 — evaluate and report

### 5.1 The LUT table (the thing I most need to see)
Per channel, one row per bin: `n`, shipped `h`, LUTCAL `h`, and `h` for b = 1.0 / 1.4 / 1.8 / 2.2,
plus the ratio `h_b180 / h_shipped`.

### 5.2 Sensitivity to `b` — the parameter the change actually moves
Report `median_bins(h_b220 / h_b140)` per channel. **§53.1's lesson: sweeping a fitted nuisance
parameter proves nothing; sweep the parameter the change moves.** `b` is that parameter, and it
was pegged at its search boundary, so this number decides whether the construction is fragile.

### 5.3 Predicted effect — state it BEFORE anything is submitted
For each of the four LUTs, on the `re_lohi` evaluation set, report:
* `g` and `g / g_shipped` under the **un-emulated** (raw local) residuals
* `g` and `g / g_shipped` under the **b = 1.8 emulated** residuals
* mean coverage, mean half-width per channel

Also report, for the shipped LUT, `g` under b = 1.8 emulation — this is the baseline the
prediction is made against.

## 6. GATES — stop and report if any fails

* **G1 (the theory's own falsification test).** The b=1.8 refit must be **WIDER than the b=1.0
  refit** in the majority of bins (`median(h_b180 / h_b100) > 1.0`). The whole hypothesis is that
  inflating to live scale corrects LUTCAL's tightening. **If b=1.8 comes out narrower than b=1.0,
  the theory is wrong — stop, report, build nothing.**
* **G2.** If `median_bins(h_b220 / h_b140) > 1.25` in either channel, the construction is fragile
  to an unidentified parameter — **report it and still build, but flag it prominently.**
* **G3.** `ED` must be byte-identical to shipped, and every asset other than `LUT` must be
  byte-identical to shipped. Verify by md5 per array and report.

## 7. TASK 3 — build the artifact (only if G1 passes)

Copy `submission_SV2.zip`, replace **only** the `LUT` array in `bounds_assets.npz` with
`LUT_b180`, repack as `submissions/submission_LUTEMU.zip`.

Then run the **zip's own `predict()`** as the final gate (§12 discipline — structural checks pass
on broken artifacts):
* extracted size < 268,435,456 B; `unzip -t` clean; no `.pyc`; entry list identical to SV2
* on ≥100 real windows: all finite, `lower <= upper` everywhere, `p == 0`
* bounds actually vary (report std of `h_u`, `h_v`)
* **the `[0.0129, 0.0098]` constant-width fallback must NOT fire** — assert 0 elements
* md5 of the built zip, and confirm `sim_real_fno_fp16.pth` inside is still `571062ce…`
* interleaved timing vs `submission_SV2.zip`, ≥5 repeats each, report median ms/window both ways
  (expected identical — a LUT is a lookup)

⛔ **Do not submit it.**

## 8. Output

`/SML_DISK_24TB/rajeshr/Aryamann/UGP/r11_results.json` + printed tables. Script `r11_lut.py`.
Append raw tables only to `project_memory.md` under
`## 74. ROUND 11 RAW RESULTS (Gemini, executed)` — no conclusions, no recommendation, no
projected live scores.

## 9. What NOT to do

* ⛔ Do not submit. Do not change `ED`, `alpha`, `cw`, `mh`, any centre, or the backbone.
* ⛔ Do not refit on raw local residuals and ship that — b = 1.0 is a **control**, not a candidate.
* ⛔ Do not "improve" the map, the objective, the split or the grid.
* ⛔ Do not refit bins with `n < 200`.
