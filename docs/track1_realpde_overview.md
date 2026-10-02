# REALPDE_T1 Bundle — Overview

**Source:** `~/Downloads/REALPDE_T1.zip` extracted to `~/Desktop/IMPORTANT/UGP/code/track1_realpde/`
**Date reviewed:** 2026-08-05
**What's in it:** A complete, working, *experimentally validated* Track 1 sim-to-real training
pipeline. This is a competitor-quality codebase — not a stub. A previous team has already run 14
controlled experimental rounds, reached **final_score ≈ 91.9** with a U-Net, and documented
everything in a 50 KB LaTeX/PDF report.

This bundle was **not** in `project_memory.md` (v1.5, 2026-08-03). It supersedes a lot of the
"to do" items in §9 of the memory.

---

## 1. Layout (relative to `code/track1_realpde/`)

```
realpde/
  __init__.py          FIELDS = (u, v, p); REAL_FIELDS = (u, v); NUMERICAL_FIELDS = (u, v, p)
  data.py              TrajectoryStore + FoilWindowDataset (sliding windows)
  device.py            Auto device picker: cuda → mps (with spectral probe) → cpu
  model.py             5 architectures: FNO2dTime, FNO3dTime, UNetTime, UFNOTime, +
  metrics.py           All 5 Track 1 metrics + leaderboard_score() composite
  normalize.py         Per-channel mean/std normaliser; decode_logvar() for the bug fix
  train.py             3 stages: sim → finetune, real, finetune; FiLM LR split
  evaluate.py          Per-checkpoint eval, 5 test_modes, leaderboard JSON output

configs/
  foil_fno.yaml        FNO2dTime, width=48, modes=16/24, 4 layers — ~7.1M params, ~54MB
  foil_fno_smoke.yaml  Tiny config (8 frames) for pipeline plumbing check
  foil_unet.yaml       UNetTime, base_width=64, n_levels=3 — ~7.4M params, current best
  foil_unet_stats.yaml Ablation: unet + use_input_stats=true
  foil_ufno_everylevel.yaml  Hybrid U-Net with FNO at every level (single seed)

scripts/
  fetch_data.py        Streaming downloader — sub_s=2 to 64x128, deletes raw shards,
                       resumable. Default 4 real + 3 numerical.
  make_synthetic.py    Fake NACA-shaped wake data — pipeline smoke test, no download
  make_submission.py   Packages submission.zip with submission.py + _model.py + model.pth
  validate_submission.py  Format-checks a zip: 256MB limit, blocked imports (scipy etc),
                       runs submission in subprocess, calls predict(), times it
  visualize.py         Plots target/pred/error/sigma

submission/
  submission.py        Codabench entry point: predict(x, meta) and SubmissionModel class.
                       Pressure zeroed in input feedback, p channel output as zeros.

reports/
  experimental_report.tex  52 KB LaTeX
  experimental_report.pdf  144 KB
  figures/                 8 architecture diagrams
checkpoints/               (empty)
data/                      (empty)
requirements.txt           torch, numpy, pyyaml, pyarrow (for fetch_data), requests,
                           matplotlib (for visualize)
```

---

## 2. Five architectures implemented (model.py)

| Arch | Params | Step (B=32) | Notes |
|---|---|---|---|
| `FNO2dTime` | 7.12M (width=48) | 0.16 s | Default. T folded into channels. FiLM per-domain. |
| `FNO3dTime` | 10.63M | 1.59 s | 3D spectral — 10× slower at training and inference. Single-seed, deprioritised. |
| `UNetTime` | 7.40M (base_width=64) | 0.18 s | **Best accuracy** (Round 12). Plain conv, FiLM, optional attn bottleneck. |
| `UFNOTime` (bottleneck) | 1.87–7.92M | 0.18 s | U-Net encoder/decoder + FNO at bottleneck. |
| `UFNOTime` (every level) | ~7M | 0.18 s | FNO at every level — see `foil_ufno_everylevel.yaml`. |

All share: `(B, T_in, H, W, C_in) → (B, T_out, H, W, C_out)` plus optional log-variance
head for SPS bounds. Domain conditioning via per-block `nn.Embedding(2, 2*width)` FiLM
(zero-init → identity at start, with residual around the block).

Time-axis convention: every architecture (except FNO3dTime) folds T into channels via
`fc0 = Conv2d(in_step*in_channels + 2, width, 1)` then runs 2D spectral/conv blocks. +
normalised x/y grid (2 ch) concatenated. + optional 3 input-stats channels (mean_u, mean_v, tke).

---

## 3. Training pipeline (train.py)

3 stages, matching RealPDEBench paradigms:
- `sim` — pretrain on numerical/CFD only
- `real` — train on real/PIV from scratch (baseline to beat)
- `finetune` — initialise from sim checkpoint, train on real (with LR split)

Loss: `rel_l2 + mse + nll_weight*gaussian_nll + tke_weight*tke_rel_l2 + mvpe_weight*mvpe`
(all in normalised space). The `tke_weight` term (default 0.1) was a key win in Round 8 —
directly optimises TKE since pointwise losses don't push the model to match fluctuation
statistics. `mvpe_weight` is exposed but defaulted to 0.

LR split for finetune: backbone @ `lr * 0.3`, FiLM + heads @ `lr * 10`. Without the split
the simulation prior was being overwritten.

---

## 4. Submission packaging (make_submission.py + submission.py)

Submission zip layout (must follow exactly):
```
submission.zip
  submission.py      (entry: predict() and SubmissionModel)
  _model.py          (vendored realpde/model.py)
  model.pth          (state_dict + cfg + norm stats; ~54MB for FNO2d+residual)
```

`submission.py` does:
- Zeros the `p` channel in the *feedback* window (PIV never sees p; can't condition on model p)
- Autoregressive rollout in normalised space, decoding back to physical for output
- Emits `{prediction, lower, upper}` from the model's variance head
- Z-multiplier env var `REALPDE_Z` (default 1.96)
- `REALPDE_BATCH` controls chunk size (default 8)
- Defensive: NaN→0, lower>upper swap, time-axis padding if model produced fewer frames

`make_submission.py` enforces the 256MB extracted limit and *fails fast* with a formula
(`bytes ≈ 2*n_layers*width²*modes1*modes2*8`) for how to shrink it.

`validate_submission.py` is a pre-flight format checker that runs the submission in a
subprocess with the eval image's *blocked* libraries (scipy, pandas, matplotlib, h5py,
einops, sklearn, cv2) and times inference. Catches format issues before spending a
daily submission slot.

---

## 5. The 14-round experimental study (experimental_report.{tex,pdf})

This is the critical part. **Someone has already done most of the work** for the project.
The big numbers (final_score = 100/(1+0.5*err) proxy; not the same as Codabench's):

| Round | Finding | Result |
|---|---|---|
| 1-2 | infra + warmup | GPU utilisation fixed (cache_size tuning) |
| 3-4 | data-scarcity regime (5 traj) | finetune 8.4% rel_l2 better, but TKE worse, 0.1pt gap (later partially retracted) |
| 5-7 | full data, 3-seed validation | finetune wins rel_l2/MVPE, baseline wins TKE → tie on final_score |
| 8 | `tke_weight=0.1` loss term | closes TKE gap, new default |
| 9 | residual connections on FNO blocks | biggest single FNO win |
| 10 | `nll_weight` resweep (0.02-0.15) | flat on sps, negative result |
| 11 | FNO3dTime | underperforms + 10× slower at inference; deprioritised |
| 12 | **U-Net** | **beats FNO2d+residual on every metric, 3-seed validated** |
| 13 | U-Net/FNO hybrid | fastest inference (0.0012s), but no accuracy win |
| 14 | **byte/units bug fix** | final_score +1.5–1.8 across the board |

### Headline results (Round 14, post-bug-fix, 3-seed mean):

| Config | rel_l2 | TKE | **final_score** |
|---|---|---|---|
| FNO2d+residual baseline | 0.0181 | 0.436 | 91.62 |
| FNO2d+residual finetune | 0.0182 | 0.419 | **91.77** |
| U-Net baseline | 0.0175 | 0.395 | **91.95** |
| U-Net finetune | 0.0171 | 0.400 | 91.91 |

Conclusion from the report:
- **U-Net > every FNO variant on every accuracy metric, 3-seed confirmed.** TKE improvement
  is the biggest win — consistent with the hypothesis that FNO's mode-truncation
  discards the fine-scale fluctuations TKE measures.
- **FNO2d+residual + finetune** is the *confirmedly* sim-to-real-beats-baseline result
  (91.77 vs 91.62, non-overlapping seed ranges).
- **U-Net** is essentially a tie between baseline and finetune (91.95 vs 91.91, overlapping
  ranges) — but the *baseline* is already at 91.95, so probably ship U-Net baseline as
  the simpler, more reproducible choice.

### The bug fixed in Round 14

`Normalizer.decode_logvar()` was missing: `evaluate.py` and `train.py` decoded the
prediction to physical units but passed `logvar` in *normalised* units. The
`coverage_and_width` function then compared a physical residual against a normalised sigma,
producing intervals up to 16× too wide for the `v` channel (std ≈ 0.061 on real data).
Coverage was ~99.9% (severely overcovered); after the fix it dropped to ~96–97% (close
to the 0.95 nominal). This single fix raised every final_score in the report by 1.5–1.8
points and reversed the FNO2d "finetune loses" conclusion.

---

## 6. What's *new* in this bundle vs. our existing track1/ skeleton

Our existing code at `code/track1/` (in `project_memory.md` §11) is at the
skeleton level:
- `models/fno.py` — FNO3d, 50M params, MPS smoke test
- `data/dataset.py` — HF streaming
- `evaluation/scorer.py` — 5-metric scorer
- `end_to_end.py` — random-weights demo, final_score 61.6

`code/track1_realpde/` is the *complete project*, with:
- 5 architectures (FNO2d, FNO3d, U-Net, U-Net+attn, UFNO) vs. our 1
- Full 3-stage training loop with FiLM LR split
- Per-domain normalisation + the logvar decode fix
- Heteroscedastic head with NLL training for SPS bounds
- Submission packaging + format validation pipeline
- 14 rounds of experimental results
- The bug fix that the report correctly attributes as responsible for ~1.5–1.8 pt gain
- Per-config `use_input_stats` ablation (input mean_u, mean_v, tke as 3 extra fc0 channels)

**The right move is probably to adopt this bundle as our starting point**, not rebuild
from our skeleton. The report even includes a 91.95-score configuration we can ship as-is
(modulo final fine-tuning on cluster GPUs).

---

## 7. Things to verify / next steps

1. **Run `make_synthetic.py`** to get a working end-to-end without downloading data
2. **Run `validate_submission.py` on a dummy submission** to verify the format pipeline
3. **The current best** (U-Net, final_score 91.95 baseline, 91.91 finetune, 3-seed) needs
   cluster GPU retraining to verify on our A100/H200 hardware
4. **The unet_stats ablation** (`use_input_stats: true`) is *untested* in the report
   (comment in `foil_unet_stats.yaml`: "Untested at time of writing; default false to leave
   prior rounds' results unaffected") — could be a quick +0.X point win
5. **The nll_weight=0.02 finding** from Round 10 was within single-seed noise and not
   adopted, but is a 0.03-point candidate worth re-sweeping under 3 seeds
6. **Cluster paths** in `foil_ufno_everylevel.yaml` and `foil_unet_stats.yaml`:
   `/hs/work0/home/users/u0001982/realpde_t1/{checkpoints,outputs}` — this is on someone
   else's HPC. Update before running on Prof. Ranjan's cluster.

---

## 8. Important findings to add to project memory

- A complete pipeline exists in `code/track1_realpde/`
- 5 architectures, all with FiLM domain conditioning for sim→real
- U-Net is the best (final_score 91.95) at 7.4M params, 0.18s/step
- The normaliser units-bug is the single biggest "fix our scoring" insight
- `submission.py` is the actual Codabench entry point — copy this verbatim
- `make_submission.py` enforces 256MB; cannot be deferred to post-compression
- `validate_submission.py` is a pre-flight for daily-submission slot economy
- Heteroscedastic head + NLL training → calibrated SPS bounds without an ensemble
