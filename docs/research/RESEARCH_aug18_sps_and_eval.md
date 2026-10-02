# Research note — Aug 18, 2026: the SPS objective, and why the local proxy is broken

Derived from the scoring source (`starting_kit_v9/.../scoring.py`) and the public
Codabench API. Nothing here relies on the local proxy's *rankings*.

---

## 1. The exact SPS objective (previously unstated in memory)

```
SPS = 100 * (1/n_scored) * SUM_over_scored_elements [ W_s * exp(-width_e/sigma) * 1[ target_e in [lower_e, upper_e] ] ]

sigma   = SIGMA_GLOBAL = 0.0563870259      (frozen: mean of u,v train_real stds)
width_e = upper_e - lower_e
scored  = (target != 0.0)                  (airfoil body / outside PIV FOV excluded)
W_s     = 0.5*(1-n(dm)) + 0.3*(1-n(tke)) + 0.2*(1-n(mvpe)),   n(x) = x/(0.5+x)
```
`W_s` is **per-window**, `exp(-width/sigma)*1[inside]` is **per-element**.

### Consequences

1. **The objective is separable over elements.** Each element's bound affects only
   its own term => the optimum is genuinely per-element. Constant bounds are
   provably suboptimal.

2. **Optimal half-width h solves  f(h) / F(h) = 2/sigma = 35.47**, where F is the
   CDF of |error| for that element and f its density. Accurate elements should get
   intervals that are *wide relative to their own error* (near-certain coverage,
   cheap because absolute width is small); inaccurate elements should get
   relatively narrow ones (you give up on them). This is the opposite of a
   uniform "err wide" rule applied globally.

3. **Accuracy multiplies SPS.** Improving tke raises `W_s`, which raises SPS
   *proportionally*. tke therefore pays twice: 0.163/pt directly on the final
   score, plus ~0.217 * (dSPS) indirectly.

### Decomposition of where we stand

`SPS = 100 * W * E`, with `E` = mean over scored elements of `exp(-width/sigma)*1[inside]`.

| | rel_l2 | tke | mvpe | W | SPS | => E |
|---|---:|---:|---:|---:|---:|---:|
| us (`fno_plain_sps`) | 94.17 | 74.03 | 92.84 | 0.6784 | 29.84 | **0.440** |
| doomduke2 (LB #1) | 94.78 | 78.03 | 94.05 | 0.7105 | 42.84 | **0.603** |

The entire SPS gap is `E`: 0.440 -> 0.603. `E` depends **only** on the bounds and
the true error distribution — no model change required.

### Quantified target
- tke 74.03 -> 79.0 : W 0.6784 -> 0.6990 => +0.81 final directly, +0.20 via SPS
- E 0.440 -> 0.603  : SPS -> ~42 => +2.67 final
- Both => **~80.7**, i.e. top-15. No new checkpoint needed.

---

## 2. Leaderboard facts (re-derived from the API, Aug 18)

The public feed exposes **one row per user** — the leaderboard rule is
`submission_rule: Force_Best`. It is NOT a full submission history.

- Our leaderboard row is still `submission_fno_plain_sps.zip` @ 77.20 => **nothing
  we have ever submitted scored above 77.20**, and 77.20 cannot be lost.
- Rank **105 / 153**. Top is doomduke2 **81.48**.
- **rel_l2 ceiling across all 153 entries is exactly 95.01; zero above.**
- **SPS ceiling is 42.84; zero entries above 45, zero above 50.**

### CORRECTION to memory §4
Two rows in the "current-eval leaders" table were pre-Aug-5 **scrapped-eval**
numbers — the exact contamination §3 warns about:

| | §4 claimed | actual current-eval |
|---|---|---|
| dentist23 | 82.44, sps 54.44 | **74.06, sps 24.95** |
| anaelle_haomiao | 81.70, sps 58.51 | **77.83, sps 32.86** |

So "reaching 54 => +5.3" targets a number nobody has ever achieved. The real
reachable band is **sps 41–43**, worth **+2.6**. The top-10 cluster very tightly
(sps 41.6–42.8, tke 78–80, rel_l2 94.4–94.8), which looks like one shared recipe.

---

## 3. ROOT CAUSE: why the local proxy has been wrong 5 times

### 3a. The local tke is structurally broken
Local tke score = **14.33** vs real **74.03**. `kinetic_energy()` is the *temporal*
variance within the 20-frame window. On our eval set:

- target KE = 7.83e-06, FNO-predicted KE = 1.17e-04 => the FNO injects **15x**
  more temporal fluctuation than the targets contain.
- No window selection fixes it: keeping only the loudest 10% of windows still
  gives tke **37.41**, nowhere near 74.03.

### 3b. The real reason: the FNO's error is 6.8x the signal on this data
On `real_eval_v3`:

| channel | mean abs err | target within-window std | error / signal |
|---|---:|---:|---:|
| u | 0.02818 | 0.00412 | **6.8x** |
| v | 0.00540 | 0.00768 | 0.7x |

`rel_l2` normalizes by `||target||`, which is dominated by the large **constant
mean** (u ~ 0.296). So a "94.17 rel_l2" hides the fact that on u the FNO is ~7x
worse than simply predicting the window mean. That is precisely why persistence
and other do-nothing predictors look brilliant locally (99.09) and terrible for
real (91.87) — **memory §2's symptom, now with its mechanism.**

### 3c. The data is real, but its statistics contradict the official constants
The downloaded HF shards are genuine PIV: the temporal-std map shows a clear wake
band (0.19-0.53 of max) against a 0.02-0.04 freestream, dt = 0.005 s uniform
(20-frame window = 0.10 s, ~14% of domain width of advection), FOV 0.213 x 0.106 m,
native **128x256** (the kit README's "64x128" is wrong; both prep scripts
correctly stride by 4).

But pooled over all 38 downloaded trajectories the statistics are incompatible
with the official normalization constants:

| | our 38 shards | official train_real |
|---|---:|---:|
| u mean | 0.33532 | **0.154961** |
| u std | 0.13031 | **0.096810** |
| v mean | 0.00725 | **-0.000518** |
| v std | 0.03582 | **0.015964** |
| u coefficient of variation *within* a field | **~1.3%** | **~62%** |
| fraction of exactly-zero targets | **0.0000** | nonzero by construction |

Two robust discrepancies:

1. **Within-field structure.** Every trajectory has u nearly uniform
   (u_mean ~0.28-0.31, u_std ~0.002-0.011, i.e. CV ~1-2%). The official
   u_mean=0.155 / u_std=0.0968 implies CV ~62% — a field varying by tens of
   percent. Our pooled u_std only reaches 0.13 because of cross-trajectory mean
   spread, not within-field structure.
2. **No masked elements.** The scorer excludes `target == 0.0` ("outside the PIV
   field of view or inside the airfoil body"). Our eval set has **none**
   (`scored fraction = 1.0000`).

Masking is the only mechanism that simultaneously lowers the mean and raises the
std in the observed direction. Solving both official moments gives a consistent
fit at **flow-region mean u = 0.2154 with 28% masked zeros**. It does not close
completely (our flow-region mean is 0.298, not 0.215), so treat this as the
leading hypothesis, not a settled fact. **`vo` is NULL in all 38 shards**, so the
mask is not directly recoverable from the data we have.

*Correction to an intermediate conclusion:* an earlier pooled-statistics check
suggested "no u/v channel issue". That was misleading — the pooled u_std was
inflated by cross-trajectory mean spread. Within every field, **v fluctuates more
than u** (v_std ~2x u_std), whereas official has u_std ~6x v_std. That inversion
is unexplained. (v_std > u_std is physically normal in a vortex street, so this is
suggestive, not conclusive.)

**Dataset heterogeneity worth flagging:** 3 of 38 trajectories — `8437_0.0`,
`10000_0.0`, `13125_0.0`, all AoA 0.0 — are gross outliers (u_mean 0.67-0.89 vs
~0.30 elsewhere; v_std 0.106-0.140 vs ~0.009). Other AoA-0.0 trajectories are
normal, so this is not an angle effect. Any rebuilt eval set should quarantine
these until understood.

### 3d. What the proxy can and cannot do (revised)
- CANNOT rank bound *policies* (constant vs per-element vs proportional): the
  anchors prove the error-distribution *shape* is wrong, not just its scale.
- CANNOT evaluate tke at all.
- CAN still be used for format/shape/runtime validation.

---

## 4. Next steps implied

1. **Rebuild the eval set with the official mask.** The arrow rows carry unused
   columns `x`, `y` (128x256 float64 coordinate grids), `t` (timestamps), and `vo`
   (null in the shards checked). `prep_real.py` reads only `u`,`v`,`shape_*`.
   Recovering the mask is the single highest-value fix: it makes tke measurable
   and SPS calibratable, which unlocks both remaining levers.
2. **Validate any rebuilt eval set against all four real anchors** for our own
   FNO: rel_l2 94.17, tke 74.03, mvpe 92.84, and sps 29.84 @ half-widths
   [0.030, 0.010]. Four constraints — a construction matching all four is
   trustworthy; anything less is not.
3. Only then optimize per-element bounds against the condition in §1.2.

---

## 5. VALIDATED ON REAL DATA: what bounds policy actually reaches the leaders

### 5a. The decomposition holds across the whole leaderboard
Computing `E = SPS / (100*W)` for all 152 scored current-eval entries:

- `E` ranges 0.023 - **0.603**. **Zero entries exceed 1.0**, which would have
  falsified the formula. Zero exceed 0.75.
- **Top-12 E = 0.5883 +/- 0.0065** (range 0.578-0.603). Twelve independent teams
  landing within ~1% of each other is not coincidence — it is a shared recipe
  sitting at a well-defined optimum.
- Our E = **0.4399**. `corr(E, W) = +0.54`, so a little E comes free with accuracy,
  but most of it is the bounds policy.

At our current W = 0.6784, reaching E = 0.5883 gives **sps 39.91, i.e. +2.19 final**.

### 5b. Constant bounds are nearly exhausted — confirmed analytically, proxy-free
Calibrating a half-normal error model to the **real** anchor (sps 29.84 at
half-widths [0.030, 0.010] => E = 0.4399 => implied coverage 0.841, s_u = 0.0213,
s_v = 0.0071), then maximizing `exp(-2h/sigma)*erf(h/(s*sqrt2))`:

| policy | half-widths | E | sps @ our W |
|---|---|---:|---:|
| current | [0.030, 0.010] | 0.4399 | 29.84 |
| **best constant** | **[0.0205, 0.0113]** | 0.4579 | **31.06** |
| `submission_fno_bopt.zip` (already built) | [0.016, 0.009] | ~0.455 | ~30.9 |
| leaders | per-element | 0.5883 | 39.91 |

**Best-possible constant bounds are worth only ~+1.2 sps (+0.26 final).** This
independently confirms memory §4's "constant caps at 31-35" and shows the existing
`bopt` zip already captures nearly all of it. Constant bounds cannot reach 0.588.

### 5c. Per-element bounds: how much heterogeneity is required
Modelling the per-element error scale as lognormal with spread `sd(log s)`:

| sd(log s) | E best-constant | E per-element | gain |
|---:|---:|---:|---:|
| 0.0 | 0.4579 | 0.4579 | +0.0000 |
| 0.6 | 0.4742 | 0.4981 | +0.0239 |
| 0.9 | 0.4898 | 0.5359 | +0.0462 |
| 1.2 | 0.5328 | 0.5996 | +0.0669 |
| 1.5 | 0.5788 | 0.6615 | +0.0826 |

Reaching E ~ 0.59 requires the per-element error scale to vary by roughly
**3-4x (one sd) across the field** — entirely plausible for wake vs freestream.

**=> memory §8.2 (a quantile regressor for per-element |error|) is confirmed as
the correct and the *only* sufficient lever, now with a quantified target
(E 0.44 -> 0.588, +2.19 final) and an exact design rule: choose each element's
half-width h_e to solve `f(h)/F(h) = 2/sigma = 35.47` for that element's
predicted error distribution.**

### 5d. How to build it despite the broken eval set
The predictor does not need the local set to be distributionally correct in
absolute terms — only its *relative spatial structure* must be right. So:

1. Predict a **relative** per-element error scale `s_hat_e` from
   inference-available features (local gradient magnitude, vorticity, KE,
   distance from wake centreline, timestep index).
2. Set `h_e = c * h_opt(s_hat_e)` from the rule above.
3. Choose the single global multiplier `c` so the **predicted mean coverage
   matches the real anchor's 0.841** — this pins the absolute scale to real
   leaderboard data rather than to the local proxy.
4. Bias `c` upward. Memory §4's asymmetry is real and severe: bounds slightly too
   tight score **exactly zero** for that element; too wide decays only as
   `exp(-2h/sigma)`.

This is bounds-only, so it cannot touch rel_l2/tke/mvpe — worst case is a wasted
submission slot, and Force_Best means the banked 77.20 is not at risk.

---

## 6. RESOLVED Aug 18: `submission_hf_fno.zip` real result — the checkpoint is DEAD

**Real: rel_l2 85.95 | tke 69.37 | mvpe 73.15 | time 91.10 | sps 9.33 | final 64.76**

This settles memory §5's fork in the worst direction. Do not rebuild on the HF
checkpoint. Do not revisit checkpoint swaps without an independent real-data test.

| quantity | local proxy said | REAL | error |
|---|---:|---:|---:|
| rel_l2 | 96.61 | **85.95** | proxy over by **10.66 pts** |
| interval efficiency | 0.518 | **E = 0.1898** | proxy over by **2.73x** |

Compare the same two quantities for the kit FNO (`submission_fno_plain_sps`):
rel_l2 94.29 -> 94.17 (over by 0.12), local E 0.4078 -> real 0.4399 (**under** by 7%).

**This is the 6th proxy failure and it obeys memory §2's rule exactly:** the proxy
is reliable for the kit FNO and worthless for a structurally different model. The
§5 contamination test passed and the checkpoint still failed — so passing a
falsification test is necessary, not sufficient.

It also corroborates §1: the decomposition `SPS = 100*W*E` gives W = 0.4918 and
E = 0.1898, a valid value; and the solved final-score formula predicts **64.69 vs
actual 64.76** (residual -0.07).

### The local->real E ratio is policy-dependent (why §5d pins to coverage, not to E)
For the kit FNO: constant bounds ratio **1.08**, proportional-to-|pred| bounds
ratio **1.61**. A single rescaling cannot serve both, so local E must never be used
to rank bound *policies*. For a per-element policy the ratio is unknown — hence
calibrate the global multiplier `c` to the real anchor's implied **coverage 0.841**,
not to any local efficiency number.

### Real SPS anchors (the only trustworthy calibration data)
| model | half-widths | real sps | real E |
|---|---|---:|---:|
| kit FNO | 0.05*abs(pred) (default) | 14.08 | 0.2076 |
| kit FNO | [0.030, 0.010] | 29.84 | 0.4399 |
| HF ckpt | [0.025, 0.007] | 9.33 | 0.1898 |

---

## 7. ROOT CAUSE FOUND: the entire project was validating on the WRONG DATASET

`docs/competition_spec.md` line 112 names the competition's public data: Google
Drive folder `1Cg23DoTuSvWXR3Mm1uRfmMNAbkyaIhrQ`, containing `train_real.tar.gz`
and `example_data/3750_0.h5`. Line 160: *sigma_global = 0.0563870 (frozen — mean
of u,v std on **train_real**)*. We had never downloaded it. Every local number in
this project came from the HF RealPDEBench `foil` arrow shards instead.

They are different datasets:

| | competition `train_real` (H5) | HF RealPDEBench `foil` (arrow) |
|---|---|---|
| shape | **(868, 64, 128)** float64 | (3990, 128, 256) float32 |
| u mean / std | 0.0464 / **0.0217** | 0.2967 / 0.0040 |
| u range | **-0.021 .. 0.073** (negative = recirculation) | 0.272 .. 0.303 |
| within-field CV | **47%** | **1.3%** |
| masked zeros | **6-13%** | **0.0%** |
| Reynolds values | 3750, 5025, 6300, 7575 ... 26700 | 2968, 3750, 4531 ... 17812 |
| naming | `3750_0.h5` | `3750_0.0.h5` |

**Clincher:** competition `v_mean = -0.000514` vs the frozen official constant
**`-0.000518`** (4-decimal match). The arrow data gives `+0.00725`. The
normalization constants and SIGMA were computed on the H5 data.

The FNO normalizes with `(x - 0.155)/0.0968`. Fed the arrow data it saw a
near-constant `+1.5` everywhere — permanently out of distribution. That single
fact explains all six proxy failures, the tke break, the un-calibratable SPS,
"fine-tuning degrades the FNO", and why persistence looked brilliant locally.

### Anchor test on the correct data (82 trajectories, 1693 windows)

| | rel_l2 | tke | mvpe | coverage |
|---|---:|---:|---:|---:|
| **measured on `train_real`** | 96.61 | **77.89** | 97.13 | 0.969 |
| real leaderboard anchor | 94.17 | **74.03** | 92.84 | 0.841 |
| delta | +2.44 | **+3.86** | +4.29 | +0.128 |

Every metric is now in the right regime, uniformly optimistic by a small,
consistent margin — exactly the signature of evaluating on the model's own
fine-tuning data. Compare tke: **12.92 before, 77.89 now, anchor 74.03.**

**This is a usable proxy for the first time in the project.**

### Caveats and what is now unblocked
- `train_real` IS the FNO's fine-tuning data, so absolute scores are optimistic.
  Use it for *ranking* changes, not for predicting leaderboard values.
- Coverage 0.969 vs real 0.841 means local errors are smaller than real, so
  locally-optimal bounds would come out **too tight** — the dangerous direction
  (too tight scores exactly zero). Always calibrate the global bound multiplier
  so predicted coverage matches the real anchor's **0.841**, per section 5d.
- Now testable for the first time: tke improvements (leaders gain +4 to +5.8),
  per-element SPS bounds, and fine-tuning (the earlier "degrades" verdict was
  measured through the broken input path and should be revisited).

---

## 8. RESULTS ON THE CORRECT DATA: fine-tuning + per-location bounds

All numbers below are on **17 trajectories the fine-tune never trained on**,
split again into a fit half and a score half. The pipeline carries a built-in
validation: the *original model at current bounds must project to +0.00*, since
lambda is calibrated so its E equals the real anchor 0.4399. It does.

### 8a. Fine-tuning works once the input path is correct
Memory §6 recorded "fine-tuning degrades the FNO (94.11 -> 89.44 in 2 steps)".
That was measured through the broken input path, so the gradients were
meaningless. On `train_real`, fine-tuning improves the model immediately.

Loss `L = rel_l2 + w * tke_rel_l2`, weight chosen from the score's own
sensitivity (d(final)/d(err_rel) = -13.6 vs d(final)/d(err_tke) = -4.47).

| w_tke | rel_l2 | tke | mvpe | + per-location bounds |
|---:|---:|---:|---:|---:|
| 0.00 | — | — | — | **degrades** (d_acc -0.181) |
| 0.05 | 94.04 | 76.93 | 92.68 | +1.15 |
| **0.10** | 93.93 | 77.83 | 92.69 | **+1.18** |
| 0.15 | 93.85 | 78.16 | 92.69 | +1.17 |
| 0.33 | 93.70 | 78.40 | 92.73 | +1.08 |
| 0.60 | 93.58 | 78.54 | 92.70 | +0.95 |

Pure rel_l2 (w=0) is *worse than not training* — the tke term is essential.
The optimum is a plateau over w in [0.05, 0.15]; the 0.01 spread there is noise.

### 8b. Bounds
| policy | real sps | d final |
|---|---:|---:|
| current [0.030, 0.010] | 29.84 | +0.00 (validation) |
| best constant [0.0155, 0.0090] | 32.49 | +0.58 |
| per-timestep | 32.52 | +0.58 |
| **per-location map** | 33.13 | **+0.71** |
| per-timestep x location | 33.14 | +0.72 (overfits) |
| ORACLE ceiling | 49.55 | +4.28 |

Practical policies capture ~67% of the oracle; agent33 (real, identical model)
reaches 71%. Error does **not** grow with rollout step -- the FNO emits all 20
frames in one shot -- so per-timestep bounds add nothing over constant.

### 8c. The candidate: `submissions/submission_ft_perloc.zip`
- fine-tuned FNO (w_tke 0.15, lr 1e-5, best of 8000 steps), fp16-packed
- per-location bound map with a **x1.15 safety margin** (costs ~1% of E, lifts
  coverage 0.747 -> 0.791; too-tight bounds score exactly zero, too-wide decay
  only as exp(-2h/sigma), so margin is cheap insurance)
- 201.9 MB extracted (cap 256); fp16 vs fp32 max abs diff 2.5e-04, rel_l2 identical
- end-to-end tested through the zip's own `predict()`: prediction/lower/upper
  shapes, finiteness and lower<=upper all verified
- official format validator: **all checks pass** at the eval geometry 32x64
  (note its default is 64x128, the raw PIV resolution, which no fixed-shape FNO
  can accept -- run it with `--height 32 --width 64`)

**Projected: 77.20 -> ~78.4**, uncertainty +-0.74 from the calibration residual.
For reference agent33 scored 78.64 on the identical base model with bounds alone,
so this is a plausible, not an extraordinary, claim.

---

## 9. MODEL SOUP + the complex-tensor trap

Six fine-tunes all started from `sim_real_fno.pth`, so their weights can be averaged.

| model | rel_l2 | tke | mvpe | E | d final |
|---|---:|---:|---:|---:|---:|
| single w0.15 | 96.59 | 80.68 | 97.33 | 0.4741 | +1.17 |
| soup w005+w010+w015 | 96.71 | 80.17 | 97.33 | 0.4796 | +1.20 |
| soup w010+w015 | 96.64 | 80.52 | 97.33 | 0.4760 | +1.18 |
| **soup all 6** | 96.61 | 80.74 | 97.35 | 0.4750 | **+1.21** |

+0.04 over the best single run is noise on its own, but weight averaging also reduces
variance under distribution shift, and the hidden eval *is* a shift. Shipped.

**Trap worth recording:** the FNO has 16 **complex** spectral weight tensors.
- `sd[k].float()` silently discards the imaginary part. The first soup attempt did this and
  scored **-7.34** while looking like a legitimate result; only the
  `UserWarning: Casting complex values to real` gave it away.
- `v.half() if v.is_floating_point()` skips complex tensors entirely, leaving the checkpoint
  at 403 MB instead of 201 MB (over the 256 MB cap).

Both need `v.is_complex() or torch.is_floating_point(v)`. The kit's `pack_ckpt_fp16.py`
already handles this correctly via `torch.view_as_real(t).half()` — use it rather than
rolling your own.

### Bounds: practical ceiling reached
Multipliers keyed to the INPUT's temporal variance (legal at inference):

| policy | E | coverage | % of oracle |
|---|---:|---:|---:|
| per-location | 0.4706 | 0.727 | 66% |
| + window-level input-tv multiplier | 0.4747 | 0.754 | 67% |
| + per-element local input-tv multiplier | 0.4724 | 0.743 | 66% |
| both | 0.4747 | 0.771 | 67% |
| ORACLE | 0.7138 | 1.000 | 100% |

Best addition is +0.0041 in E (~+0.07 final) — not worth the extra inference code path.
Bounds are done until someone builds a materially better per-element error predictor.
