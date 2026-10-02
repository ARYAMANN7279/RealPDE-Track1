# RealPDE Track 1 — Sim2Real Operator Learning (UGP-I)

Codabench competition #17363 · NeurIPS 2026 RealPDE challenge, Track 1 (Sim2Real)

Rajesh Ranjan · Aryamann Srivastava · Prashant Kumar
Department of Aerospace Engineering, IIT Kanpur

This is the live working directory for the project, tracked directly. Bulk artefacts are
excluded by `.gitignore` and listed at the bottom — everything else that matters is here.

## Task

Given 20 consecutive frames of a 2-D velocity field `(u,v)` on a 32x64 PIV grid behind a
NACA 4418 airfoil, predict the next 20 frames **and** return a lower and upper bound for
every predicted scalar. Training data is 81 experimental trajectories (69,085 frames),
Re 3,750-26,700, AoA 0-20 degrees. Test conditions are unseen and extend past the training
Reynolds range; neither Re nor geometry is given to the model.

Hard limits: **256 MB** total package (the supplied FNO backbone alone is 201.4 MB),
**300 s** inference, PyTorch and NumPy only.

## Final result

| | `rel_l2` | `tke` | `mvpe` | `time` | `sps` | **combined** |
|---|---:|---:|---:|---:|---:|---:|
| kit FNO baseline | 94.17 | 74.03 | 92.84 | 92.66 | 14.08 | 72.65 |
| **development set** (`STACK`, banked) | 93.977 | 78.623 | 93.244 | 90.771 | 39.558 | **80.081266** |
| **final decision phase** (same archive) | 91.158 | 59.882 | 91.598 | 78.952 | 53.847 | **75.087416** |

Development board: roughly 61st of 182. Leader 82.218, top-50 cutoff 80.435, top-10 cutoff 81.901.

## Method

1. **Point forecast.** The official FNO3d backbone, fine-tuned on `train_real`, with
   independently fine-tuned checkpoints averaged in weight space (model soup). The memory
   budget admits exactly one backbone at inference, so averaging buys ensemble-like gains
   at zero inference cost.
2. **Fluctuation correction.** Squared-error training leaves the forecast over-smoothed,
   retaining about 74% of the true fluctuation energy. A small auxiliary head rescales the
   predicted fluctuations. Worth +2.21 on `tke` in isolation.
3. **Prediction intervals.** A U-Net consumes the 20-frame input concatenated with the
   forecast (80 channels) and emits a centre correction `c` and a predicted residual
   magnitude. That magnitude goes through a 24-bin lookup table to a spatially varying
   half-width `h`, giving `[y + a*c - h, y + a*c + h]`.

Because the half-width enters only the bounds, interval work leaves `prediction`
bit-identical, so the three accuracy channels provably cannot move. Verified live to
<= 3e-4. This is what made the calibration work cheap and safe.

## Component ablation (development set, live scores)

| configuration | `tke` | `sps` | combined | isolated delta |
|---|---:|---:|---:|---:|
| fine-tuned FNO, weight-averaged (`SOUP_v1`) | 76.00 | 34.37 | 78.45 | — |
| + U-Net interval head (`SHIFT_v2`) | 76.00 | 37.48 | 79.25 | +0.80 |
| + fluctuation correction (`FA1BMT`) | 78.62 | 38.29 | 79.59 | +0.10 |
| + interval width calibration (`NARROW080`) | 78.62 | 39.56 | 80.06 | +0.20 |

Backbone refinements lie between rows, so the combined column is not the running sum of
the deltas. A final inference-path optimisation (device-side checkpoint load, uncompressed
asset archive) reached the banked 80.081266 without touching any model.

## Scoring, recovered

The development-phase weights are undisclosed. Least squares over 40 public leaderboard
rows recovers them, reproducing every published total to within 0.005:

```
final = 0.46743*rel_l2 + 0.10027*tke + 0.09420*mvpe + 0.09689*time + 0.24737*sps + 0.9119
score_error(e) = 100 / (1 + 0.5*e)
time           = 100 / (1 + sqrt(t / 728.96))
sps           ~= E[ exp(-w/sigma) * 1(covered) ],  sigma = 0.0563870259
```

**The final decision phase does not use these weights.** It is a plain unweighted mean:
`(91.157704 + 59.881707 + 91.598165 + 78.952367 + 53.847134)/5 = 75.0874154`, against a
reported 75.087416. Exact to six decimals. `sps` was the only channel that improved under
the distribution shift (39.6 -> 53.8); `tke` collapsed (78.6 -> 59.9).

## Negative results

These cost real submission slots and are worth more than the positive ones.

- **Divergence-free projection: -9 points.** 45% of measured fluctuation energy is
  non-solenoidal. Ground-truth `|div|` is 0.00677 against our forecast's 0.00599 — the
  measurements are *less* divergence-free than the prediction already was.
- **Momentum / vorticity-transport loss: ineffective.** 2-D vorticity transport explains
  only R^2 = 0.044-0.115 of the observed temporal tendency.
- **POD truncation removes turbulence, not noise.** The discarded tail has lag-1 spatial
  correlation 0.75 and temporal 0.79, and there is no spectral gap (225-364 modes for 99%
  energy). Spectral, Wiener and temporal filtering all failed for the same reason.
- **The interval head is feature-limited, not resolution-limited.** Going from 24 to 4096
  LUT bins buys +0.008. Position-stratified tables gained +0.041 in-sample and lost 0.059
  out-of-sample.
- **CUDA graphs: no effect** (3.403 s vs 3.412 s). Batch 48 is optimal; 16 and 32 are
  slower by 4% and 1%. Interleaved repetitions were required to see this — a naive sweep
  reports batch 16 fastest because of thermal drift.

The common explanation for the first three is in the paper: PIV measures a plane of a
3-D flow, `y = M(q) + eps`, and a constraint valid for the latent `q` need not hold for
the measured `y`.

## Layout

```
project_memory.md   the primary record: every experiment, every live score, and the
                    reasoning behind each submission slot. Read this before proposing
                    anything — most ideas are already measured and closed.
shipped/            the code that actually scored: submission.py and load_baseline.py,
                    extracted from submission_STACK.zip
hipc_srs/           HiPC 2026 Student Research Symposium paper (LaTeX, PDF, figure source)
analysis/           bounds, calibration, validation and training analysis
harness/            local scoring harness and benchmarks
local_harness/      earlier harness generation, kept for the runs that cite it
pipeline/           fine-tuning and model-soup training
train/ train_mvpe/ _tke/   training entry points per objective
artifacts/          shipped LUT, timing curve, per-run result JSON
docs/               competition spec, validation gates, research notes
docs/task_briefs/   the 25 agent task briefs and handoffs from the run, kept for provenance
attic/              one-off artefacts: applied patches, b64 payloads, scraped forum pages
figs/ results/      figures and result dumps
agents/ mcp/ scripts/      tooling used during the run
```

The ~105 loose `.py` files at the root are the competition scripts. They stay at the root
deliberately: 28 of them import each other (mostly `submission` and `u_fno`) and they read
data by paths relative to this directory, so filing them into subfolders would break both.

## Not tracked

Excluded by `.gitignore`, all reconstructible or externally sourced:

- `submissions/` (1.3 GB) — every scoring archive, each bundling the 201.4 MB backbone
- `underPINN/` (690 MB) — a separate upstream repository cloned here for reference
- `data/`, `starting_kit*/`, `realpdebench/` — the official competition release
- `venv/`, logs, caches, `*.pth`, `*.zip`, and the interval/fluctuation head assets
- `hipc_srs/Aryamann.pdf` — the signed advisor letter

## Paper

`hipc_srs/` holds the two-page IEEE paper submitted to the HiPC 2026 Student Research
Symposium on 1 October 2026 (decisions 1 November), covering the resource-constrained
forecasting framework, the decoupled interval calibration, and the measurement-operator
analysis of why the physics constraints failed.
