# Aug 20 — Full-day research: where the remaining points actually are

## 1. THE STRATEGIC PICTURE (from 166 scored leaderboard entries)

Gap from our **78.07** to doomduke2's **81.64** = 3.58:

| subscore | us | #1 | delta | worth |
|---|---:|---:|---:|---:|
| rel_l2 | 94.17 | 94.90 | +0.73 | +0.22 |
| tke | 74.03 | 78.17 | +4.15 | **+0.68** |
| mvpe | 92.84 | 94.13 | +1.29 | +0.28 |
| time | 91.36 | 92.44 | +1.08 | +0.11 |
| **sps** | **33.08** | **43.31** | **+10.23** | **+2.22** |

## 2. ⛔ BOUNDS ALONE CANNOT WIN ⛔
Writing `SPS = 100*W*E`, our W = 0.6784 is fixed by accuracy. To reach 81.5 on our
current model we would need **E = 0.7207** — higher than anyone has ever achieved
(max 0.6073) and essentially oracle-level.

| | E | % of our oracle (0.7775) |
|---|---:|---:|
| our banked (constant [0.0129,0.0098]) | 0.4876 | 63% |
| best possible CONSTANT (real) | ~0.489 | 63% |
| **learned uncertainty head (this work)** | **~0.528** | **68%** |
| top-20 mean | 0.5896 | 76% |
| doomduke2 | 0.6073 | 78% |
| oracle (perfect per-element) | 0.7775 | 100% |

**Conclusion: the remaining gap is ACCURACY, not bounds.** The top teams have both
better models AND better bounds; we can close part of the bounds gap but not enough.

## 3. LEARNED UNCERTAINTY HEAD — three formulations, one ceiling
Trained on `train_real`, validated on disjoint held-out trajectories.

| variant | approach | E_test (model scale) |
|---|---|---:|
| constant | init | 0.5132 |
| **v2** | maximise `exp(-2h/s)*sigmoid((h-e)/tau)` directly | **0.5586** |
| v3 | regress log\|err\|, then h = k*exp(mu) | 0.5401 |
| v4 | regress log\|err\|, then per-quantile optimal-h LUT | 0.5561 |

All converge to **E ~ 0.556 model / ~0.528 real -> sps ~35.9 -> final ~78.67 (+0.60)**.

### What each attempt taught
- **v2** learned the right h *mapping* but almost no spatial structure
  (`sd(log h) = 0.36`, where theory wants >~1.2). The step-like surrogate has
  vanishing gradients at small tau.
- **v3** learned real structure (`sd(mu) = 0.87-1.02`) but `h = k*exp(mu)` is the
  wrong mapping and scored *worse*.
- **v4** combined them via a non-parametric LUT. The optimal-h curve **grows**
  with predicted error (0.0020 -> 0.0270) but **sub-linearly** — so "give up on hard
  elements" was wrong; the truth is "widen, but less than proportionally".
- Train/test gaps are tiny throughout (0.5599/0.5561), so this is **underfitting,
  not overfitting** — yet more capacity did not break the ~0.556 ceiling. The limit
  is the *features*, i.e. how much of the error is predictable at all from
  inference-available signals.

## 4. WHAT THIS MEANS FOR STRATEGY
1. **Ship the learned head**: +0.60 -> ~78.67. Bounds-only, so accuracy cannot regress.
2. **Then the model is the only remaining lever.** Worth ~1.2 directly (rel_l2+tke+mvpe)
   PLUS a better error distribution, which raises the achievable E on top.
3. Every top-20 team sits at tke 78-80 while stock FNO is 74.03. That is the single
   most consistent signature on the board — whatever they do, it moves tke ~+4.
   `benslash2` is the exception (tke 74.70, near-stock) and reaches 80.74 via
   rel_l2 95.01, the highest on the board.
