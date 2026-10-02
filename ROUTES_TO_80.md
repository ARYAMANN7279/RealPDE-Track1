# Every Route to 80+ — Complete Enumeration

**Current: 79.342** (`submission_ENSEMBLE_FAST_v3.zip`). **Top-50 cutoff: 80.401.**
**Gap: +1.06.** Rank 1 (`np-user`): 81.761. Phase closes 27 Sep 2026 (~26 slots).

`final = 0.46743·rel_l2 + 0.10027·tke + 0.09420·mvpe + 0.09689·time + 0.24737·sps + 0.9119`

Marginal value of one subscore point, **including the W→sps channel**:
**rel_l2 +0.669 · mvpe +0.170 · tke +0.157 · time +0.097**

---

## 1. Where the gap actually is

| | rel_l2 | tke | mvpe | time | E |
|---|---:|---:|---:|---:|---:|
| top-21 mean | 94.597 | 79.028 | 93.816 | ~90.9 | 0.598 |
| **us** | **94.045** | **76.00** | **92.874** | 90.663 | **0.551** |
| deficit | −0.552 | **−3.028** | −0.942 | −0.24 | −0.047 |

Priced: accuracy **+1.004**, bounds **+0.700**, total **+1.70**.

**★ ~70% of the bounds deficit is mechanically downstream of the accuracy deficit.** Our
rel_l2 error is 9.8% larger than the cluster's, so our bounds must be ~9.8% wider at equal
coverage; `nil` 0.596→0.538 alone recovers E 0.551→0.584. **Fix accuracy and most of the
bounds gap closes itself.** There is no independent second lever.

---

## 2. ★★ THE ROOT-CAUSE HYPOTHESIS (new, 1 Sep) ★★

`load_baseline.py:61` — the kit FNO is `modes1=4, modes2=12, modes3=16`.

| axis | modes kept | unique rfft modes | **bandwidth** |
|---|---:|---:|---:|
| **time (T=20)** | **4** | **11** | **36.4%** |
| height (H=32) | 12 | 17 | 70.6% |
| width (W=64) | 16 | 33 | 48.5% |

**TKE is computed from `u − mean_t(u)` — purely temporal fluctuation energy.** An operator
that can represent only 36% of temporal modes **structurally cannot carry** the fluctuation
content TKE scores. This single fact explains, in one stroke:

* why our tke (76.00) trails the leaders (78–80);
* why the early per-frequency spectral gain bought tke +0.94 — it was patching exactly this;
* why fine-tuning plateaus at ~78.3 local tke across **16+ configs** — the ceiling is
  architectural, not optimisational;
* why every tke remedy (noise injection, phase augmentation, wtke weighting, TKE
  amplification, divergence-free) failed — all post-hoc patches on a bandwidth limit.

**⇒ The predicted fix is more temporal modes.** Cost in a dense FNO is `O(k1·k2·k3)`, which
is why the kit truncates. **A factorised spectral layer costs `O(k1+k2+k3)`** — full
bandwidth on all three axes for ~22× fewer parameters (measured: 201 MB → 9 MB).

**Status: the single most promising untested idea in the project.** Falsifiable cheaply —
see §5.1.

---

## 3. ⛔ CLOSED — measured, not assumed

### Bounds / SPS (all closed; ceiling ≈ +0.04 total)
| route | verdict |
|---|---|
| post-hoc LUT recalibration / local→real scale fit | family misspecified; best resid 0.0128; **cost a live slot** (LUTFIX 78.34) |
| LUT width scaling, wider | measured live: **78.30** (WIDE125) |
| LUT width scaling, tighter | measured live: **76.70** (arcsinh) — `h_u` ratio **anti-correlates** with live score |
| width head (5 archs, 3 objectives, 4× capacity) | caps at 22–23% of headroom; in-sample probe 21.5% ⇒ **information limit** |
| centre predictor (2.8× params, 3 losses, spectral operator) | β = 0.12–0.17 of oracle; 4× short |
| centre ensembling (same architecture) | +0.05…+0.08; members correlate ~0.99 |
| α (centre strength) sweep | α = 0.85/0.95 already optimal, flat top |
| asymmetric / offset global bounds | +0.0004 |
| per-bin centre offset | nothing |
| target quantisation | targets continuous float32; no atoms |
| time-mean bias correction | **f = 0.05** measured on two splits (§36.1) |

### Accuracy
| route | verdict |
|---|---|
| fine-tuning recipe: lr, wd (4 orders), steps, wtke, data, init, soup size — **16+ configs** | all tie; plateau on both easy and honest splits |
| wtke weighting | **saturated**: 0.15→0.60 buys +0.32 tke for −0.27 rel_l2, net negative |
| model soup / weight averaging | works, but +0.38 total and exhausted |
| prediction ensembling, ensemble distillation | +0.02 |
| EMA | ties |
| `soup_v3` (strict holdout) | +0.07 tke ⇒ illusion. `soup_v2` leaks (100% of data) |
| divergence-free constraint | **70.99** live — far worse |
| mvpe in the loss | gradient touches 1.2% of elements |
| phase augmentation (4 subsample phases) | **hurts**: no-aug control beats both arms |
| noise injection | negative (tested inside the phase-aug arms) |
| TTA | closed by construction |
| per-location TKE amplification | biggest oracle in the project; does not transfer |
| **alternative architectures (official FT checkpoints)** | **NEW, 1 Sep — both WORSE than our soup:** |

| model | re_lohi acc-proxy | aoa15 | size |
|---|---:|---:|---:|
| **SOUP (ours)** | **62.227** | **62.335** | 201 MB |
| kit FNO | 61.783 | 61.949 | 201 MB |
| CNO-FT (official) | 60.879 | 61.083 | 32 MB |
| Transolver-FT (official) | 59.089 | 59.270 | 50 MB |

| **cross-architecture ensemble (FNO+CNO)** | **NEW — negative.** Error correlation 0.727 (vs 0.99 for soup members), so genuinely decorrelated, but every blend loses: 20% CNO gives rel_l2 **+0.086** and tke **−1.539** ⇒ net **−0.196**. Averaging smooths away fluctuation energy — the same bandwidth story as §2. |

### Inference / time (ceiling ≈ +0.04)
* **FNO is 88.5% of runtime** (6.84 of 7.73 ms); everything else totals 0.03 ms.
* fp16/bf16 **unavailable** — cuFFT rejects the 20×32×64 transform (20 not a power of 2).
* CUDA-stream fan-out of the 3 bounds nets: **2.02×, bit-identical** (+0.02). ✓ available.
* Distillation to 1 net: +0.069 time but −0.0596 sps ⇒ **net +0.010**. Not worth it.
* Naive F-FNO is **5× SLOWER** (48 transforms/forward vs 8) ⇒ −0.33…−0.67 on `time`.

---

## 4. ⚠️ Traps that have each cost a submission
1. **Local optima do not transfer.** Three slots lost (78.34, 78.30, 76.70) to candidates
   that were provably optimal on the local holdout. `h_u` ratio anti-correlates with live.
2. **Prefer moves whose SIGN is calibration-independent.** The centre shift works because at
   fixed widths only the coverage indicator moves.
3. **Check what a checkpoint was TRAINED on before scoring it on a split.**
   `finetune.py:32` holds out `every5`, so `re_lohi`/`aoa15` are 81–100% *training* data for
   every existing fine-tune. This invalidated an entire line of work.
4. **Silent failures pass structural checks.** `ENS_IMPROVED_v3` shipped constant bounds
   because a 120-vs-80 channel mismatch was swallowed by `except Exception: net = None`.
   Only running the zip's own `predict()` caught it.
5. **Never scratch to `/tmp`** — the 47 GB root partition truncated a checkpoint mid-unzip,
   and the archive still passed `unzip -t`.

---

## 5. ★ WHAT REMAINS OPEN — ranked

### 5.1 ★★ Fused-domain F-FNO at full temporal bandwidth — THE candidate
**Theory (§2):** tke is capped by 36% temporal bandwidth. Factorisation gives full
bandwidth at `O(k1+k2+k3)`.
**Blocker:** naive F-FNO is 5× slower. **Fix identified, not yet implemented** — a 1-D
multiplier along T is, in the 3-D Fourier domain, `W1(k_T)` broadcast over `(k_H,k_W)`, so
all three factors become einsums over **one** rfft/irfft pair (2 transforms/layer, not 6).
Never materialise the sum — that rebuilds the dense tensor and destroys the size win.
**Falsification (cheap, do first):** implement, benchmark ms/sample vs kit FNO. If it is not
within ~1.3× of kit speed, the route costs more in `time` than it can win. Then train briefly
on `train_sim` + fine-tune and check tke on the honest split.
**Prize if the theory holds:** tke 76→79 = +0.47, and the accuracy→E channel carries most of
the remaining +0.70. **This is the only idea that plausibly reaches +1.06.**

### 5.2 Sim-mixing during fine-tuning (`psim` dose-response)
`sim_frames.npy` (1.33 GB, moment-matched, mask-transplanted) is prepared and **never used**.
Dose-response at psim 0.15/0.35/0.50 vs a 0.0 control was launched but never completed.
Cheap, independent of architecture. The RealPDEBench paper states sim pretraining
"consistently improves accuracy and convergence". **Finish this.**

### 5.3 Longer training
Max attempted is 12,500 steps. Nobody has tried 50–100k. Cheap to test, low prior.

### 5.4 Loss formulation
Every run used `rel_l2 + w·tke`. Untried: spectral loss (penalise the energy spectrum
directly — the natural fit for §2), multi-step/rollout, distributional/adversarial.
**A spectral loss is the natural partner to 5.1 and should be tested with it.**

### 5.5 Residual items
* CUDA-stream fan-out: **+0.02**, bit-identical, ready to build.
* Time-mean correction at f=0.05: **+0.094**, but needs 15.3 MB against 5.5 MB free —
  distil to W32/W48 or drop an ensemble member.
* Combined these give **≈ 79.46**. Real, but not a route to 80.

---

## 6. Honest position

**Everything except §5.1 is measured and near-exhausted.** The realistic stack from
already-validated components is **≈ 79.46**.

**+1.06 requires the accuracy deficit to close, and the only theory that explains that
deficit — and predicts a fix — is the temporal-bandwidth limit in §2.** If a fused-domain
F-FNO reaches kit speed at full bandwidth and lifts tke toward 79, top 50 is live. If it
does not, this codebase tops out around **79.5**, and that should be said plainly rather
than chased.

**Next action: implement and benchmark the fused-domain F-FNO (§5.1). It is a few hours of
engineering and it decides the question.**
