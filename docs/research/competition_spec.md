# NeurIPS 2026 RealPDE Competition — Track 1: Sim2Real

**Codabench ID:** 17363
**Source:** https://www.codabench.org/competitions/17363/
**Compiled:** 2026-08-01 from the Codabench public API (no auth required)
**Cross-reference:** RealPDEBench (ICLR 2026 Oral) → https://github.com/AI4Science-WestlakeU/RealPDEBench
**Reproduction:** `curl -sL "https://www.codabench.org/api/competitions/17363/" -A "Mozilla/5.0" | python3 -m json.tool`

---

## 1. Identity

- **Title:** NeurIPS 2026 RealPDE Competition — Track 1: Sim2Real
- **Description:** "Scientific ML for real-world physical systems: Simulation-to-Real transfer on paired NACA4418 airfoil flow data."
- **Created by:** Tailin Wu (Westlake University), 2026-07-03
- **Collaborators:** peiyanhu, wenhao2026, haodong_feng
- **Active participants:** 201
- **Submissions so far:** 484
- **Prize pool:** $21,000 across both tracks → $6k/$3k/$1.5k per track (funded by Uniforce AI Ltd.)
- **Contact:** realpde-competition@googlegroups.com
- **Track 2 (sister):** Codabench #17385 (LTTTA — different topic)

---

## 2. The 6 Pages (verbatim from Codabench API)

### 📄 Page 1 — Overview

```
# NeurIPS 2026 RealPDE Competition - Track 1: Sim2Real

Part of the [NeurIPS 2026 RealPDE Competition](https://realpdecompetition.github.io/).
Track 2 (LTTTA) runs on Codabench: [codabench.org/competitions/17385].

Registration required. Registration via the team form is required for both tracks
(deadline: August 20, 2026):
- [Track 1 registration form](https://forms.gle/LYeTgUTfr4ygntNu6)
- [Track 2 registration form](https://forms.gle/qmMYaK5u9r86rPGKA)

Joining on Codabench alone is not registration.

## Announcements
- 2026-07-22: Starting kit v6. Adds a worked example of returning optional SPS
  interval bounds (lower / upper) from predict(). Interface and scoring unchanged.
- 2026-07-18: Warm-up submissions cleared. If you had submissions in Warm-up,
  please re-submit at any time.
- 2026-07-05: Competition launched.

## At A Glance
- Dataset: paired real-world and simulated trajectories of 3D NACA4418 airfoil
  cross-sectional flow.
- Task: predict real-world future flow fields from input flow windows.
- Modalities: simulation provides u, v, and p; real measurements provide u and v,
  with p treated as unmeasured.
- Evaluation resolution: approximately 32 x 64 after spatial subsampling.
- Ranking: final score is the mean of five 0-100 subscores: Rel-L2, TKE, MVPE,
  Time, and SPS.

## Schedule
- Warm-up Phase: July 5 - July 19, 2026 (UTC).
- Main Development Phase: July 20 - September 27, 2026 (UTC). Hidden validation
  leaderboard.
- Final Decision Phase: September 28 - October 25, 2026 (UTC). Top 10 teams
  from dev leaderboard are shortlisted; organizers re-train and evaluate on
  private test set.
- Final results announced: November 10, 2026.
- Code submission deadline: November 25, 2026.
- NeurIPS presentation: December 6, 2026.

## Prizes
6,000 / 3,000 / 1,500 USD to top 3 of each track (funded by Uniforce AI Ltd).
Top 3 invited to oral presentation; top 5 invited to co-author joint results
paper.

## Resources
Starting kit on the Files tab (visible after sign-in + approval). It vendors
baseline models and the official scoring program:

starting_kit/
├── README.md               data format, geometry, usage
├── submission_template.py  Track 1 predict() API; rename to submission.py
├── load_baseline.py        load CNO / FNO / Transolver checkpoints
├── pack_ckpt_fp16.py       complex-safe fp16 packing (256 MB cap)
├── scoring.py              exact official scoring program
├── smoke_test_kit.py       self-check
├── _vendor/einops/         vendored
└── rpde_baselines/         vendored CNO / FNO / Transolver model code

Pretrained baseline checkpoints on Google Drive (folder 1Cg23DoTuSvWXR3Mm1uRfmMNAbkyaIhrQ).
```

### 📄 Page 2 — Data

```
# Data

## Physical Setting
- Geometry: 3D NACA4418 airfoil cross-sectional flow.
- Real-world measurement: time-resolved Particle Image Velocimetry (PIV) in a
  circulating tunnel.
- Simulation: matched 3D CFD under same geometry and operating conditions.
- Angles of attack: 0, 5, 10, 15, and 20 degrees.
- Reynolds numbers: 2,968 to 27,975.
- Channels: simulation has [u, v, p]; real data is scored on measured [u, v],
  while p is set to zero.

## Resolution
Raw 64 x 128 → evaluation at 32 x 64 (2x spatial downsample).
Tensors: (N, T_in=20, H=32, W=64, C=3) → (N, T_out=20, 32, 64, 3).

## Public Data
[Google Drive folder 1Cg23DoTuSvWXR3Mm1uRfmMNAbkyaIhrQ]:
- train_sim: simulated training trajectories for pretraining
- train_real: real-world training trajectories for finetuning
- example_data: small samples for tensor shape checking

## Codabench Data Layout
data/<phase>/
  input_data/samples.npz
  reference_data/targets.npz

- samples.npz contains input (N, T_in, H, W, C) and optional sample_id.
- targets.npz contains target (N, T_out, H, W, C).
- T_in = T_out = 20, H = 32, W = 64, C = 3.

Only input windows exposed to submissions; targets held out by scorer.
```

### 📄 Page 3 — Evaluation

```
# Evaluation

final_score = mean(rel_l2_score, tke_score, mvpe_score, time_score, sps_score)
All 0-100, equal-weight. Higher is better.

## Scores You See
Public leaderboard: single column final_score.
Your own submission's detailed results show the 5 subscores.

## Metrics

### Relative L2
||pred - target||_2 / ||target||_2
Real airfoil: only u, v scored.

### TKE
TKE = 0.5 * (mean_t((u - mean_t(u))^2) + mean_t((v - mean_t(v))^2))
Reported as relative L2.

### MVPE
Mean velocity profile error at probe locations behind airfoil.
Lower MVPE = better long-horizon physical consistency.

### Time
r = t_neural / t_numerical, with t_numerical = 0.72896 seconds.
score = 100 * 1 / (1 + sqrt(r))

### SPS (Safe Prediction Score)
sigma_global = 0.0563870 (frozen — mean of u,v std on train_real)
Per element: inside = (lower <= target <= upper), nil = (upper - lower) / sigma_global
SPS sums 3 error branches (DM = Rel-L2, TKE, MVPE):
  pm = e / (0.5 + e)
  branch = mean over elements of (1 - pm) * exp(-nil), gated by inside
  weighted = 0.5*branch_dm + 0.3*branch_tke + 0.2*branch_mvpe
  sps_score = 100 / (1 + exp(-weighted))

If you don't return bounds, scorer uses default lower = pred - 0.05*|pred|,
upper = pred + 0.05*|pred| (width 0.1*|pred|).

SPS bounded in [50, ~73].

## Execution Time Limit
3 minutes container execution in Warm-up + Development phases.
Over → Failed, no score.

## Score Mapping
For Rel-L2, TKE, MVPE: score = 100 / (1 + 0.5 * error)

## Zero-Score Conditions
Submission scores 0 on every subscore if:
- prediction contains non-finite values (NaN / Inf)
- prediction shape does not match (N, 20, 32, 64, 3)
- lower / upper bounds wrong shape, non-finite, or reversed (lower > upper)
```

### 📄 Page 4 — Submission

```
# Submission

## Expected Archive Layout
submission.zip
  submission.py
  model.pth              # optional
  any_supporting_files/  # optional, e.g. vendored pure-Python packages

## Submission Size Limit
256 MB after extraction (model checkpoint included).
Oversized → fail immediately without evaluation.

Aim to meet this by design (model width, depth, architecture).
Half-precision (fp16) packing is a lossy workaround — discouraged.
Safetensors format keeps checkpoints compact without loss.

## Evaluation Environment
pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime (Python 3.10, CUDA 12.1)
Nothing installed at evaluation; no network access; requirements.txt ignored.

Available: torch/torchvision/torchaudio 2.2.2, numpy 1.26, pillow, PyYAML,
requests, tqdm, sympy, networkx.
Not available: scipy, pandas, matplotlib, h5py, einops, scikit-learn, opencv.

If you need a pure-Python package, vendor it in your archive.

## Minimal Interface
def predict(input_array, metadata=None):
    """Return predictions with shape (N, T_out, H, W, C)."""

input_array: NumPy array shape (N, T_in, H, W, C).
Return: NumPy array or Torch tensor.

## Class-Based Interface
class SubmissionModel:
    def __init__(self, submission_dir=None, device="cpu"): ...
    def predict(self, input_array, metadata=None): ...
    def load_checkpoint(self, path, device): ...  # optional

If model.pth exists, ingestion calls load_checkpoint.

## Output Shape
(N, T_out=20, 32, 64, 3). Scorer evaluates measured real channels u, v only.
Pressure channel p may be zeros.

## Optional Interval Output
{
    "prediction": pred,
    "lower": lower_bound,
    "upper": upper_bound,
}
All arrays same shape. If bounds omitted, default interval used.

## Practical Notes
- Do not access external network.
- Keep inference deterministic and bounded.
- Use relative paths.
- Heavy training done before submission; Codabench evaluates inference only.
```

### 📄 Page 5 — Rules

```
# Rules

## Eligibility
Individuals and academic/industrial teams worldwide. Max 3 members per team.

## Submission Limits
- Warm-up: 3 per day
- Main Development: 1 per day
- 100 per phase per team
- 256 MB max archive size
- 3 minutes execution time limit
- Multiple accounts for same team prohibited

## External Resources
Public datasets, pretrained models, OSS OK when allowed by their licenses.
At evaluation: no private APIs, no participant-controlled keys, no network.
All inference files in submitted archive or starting kit only.

## Reproducibility
Winning teams must provide complete reproducible code.
Prize eligibility requires open-source before NeurIPS presentation.

## Final Verification
Top 10 on dev leaderboard → organizers re-train from scratch on their cluster
→ evaluate on private test set with unseen AoA/Re → that decides top 3.
Organizers verify code integrity, check rule compliance, run robustness tests.
Non-reproducible = disqualified.

Final submission package format announced ~5 days before end of Development.

## Ranking
Track 1 ranking by final_score on official Sim2Real leaderboard.
Organizers reserve right to audit top submissions.
```

### 📄 Page 6 — FAQ

```
# FAQ

Q: Submission size limit?
A: 256 MB extracted. Oversized fails immediately.

Q: Execution time limit?
A: 3 minutes container execution per submission (data download excluded).
   Official Transolver baseline runs ~1 minute.

Q: "Transient platform storage error while downloading the bundle"?
A: Platform-side Codabench issue. Failed = no quota penalty. Resubmit ~1 hour later.

Q: Install extra packages via requirements.txt?
A: No. Vendor pure-Python packages in your archive.

Q: Model/checkpoint loading counted in Time score?
A: No. Model construction + loading runs outside timed region. Only inference timed.
   Platform-level execution time limit (3 min) does include loading.

Q: How are final rankings determined?
A: Top 10 dev → organizers re-train on their cluster → eval on private test
   with unseen AoA/Re → those scores decide top 3.

Q: How and when do finalists submit training code?
A: Format announced ~5 days before end of Development Phase.

Q: What timezone are deadlines?
A: All UTC. NOT Anywhere-on-Earth.

Q: Questions?
A: Competition forum, or realpde-competition@googlegroups.com.
```

---

## 3. Phases

| Phase | Window | Sub/day | Sub total | Eval data | Status |
|---|---|---|---|---|---|
| Warm-up | Jul 5 – Jul 19, 2026 | 3 | 100 | Format validation, public RealPDEBench | Previous |
| **Main Development** | **Jul 20 – Sep 27, 2026** | **1** | **100** | **Hidden validation leaderboard** | **Current (day 12 of 70)** |
| Final Decision | Sep 28 – Oct 25, 2026 | 1 | 1 | Private test, unseen AoA/Re | Next |

---

## 4. ⚠️ CRITICAL DISCREPANCY: NACA4418 vs NACA0025

The Codabench competition spec **explicitly says NACA4418** (5 occurrences across pages 1, 2, etc.).

The RealPDEBench repo (the underlying benchmark) uses **NACA0025** (`tapered_naca0025` in `ThreeD_NACA.jl`).

**What this means:**
- Either the competition uses a different airfoil than the benchmark (i.e., the warm-up data and HF data are NACA0025 but the dev/test sets use NACA4418), OR
- There's a typo in the competition spec/benchmark, OR
- The competition has modified the benchmark data to use NACA4418

**Action item:** ask the organizers on the competition forum. Work as if both are possible — implement the loader to be flexible.

Other possibilities:
- The "Foil" dataset in RealPDEBench is one variant; the competition may have added a NACA4418 variant
- The competition's "train_sim" / "train_real" on Google Drive may differ from the HF dataset

---

## 5. The 5 Evaluation Metrics — Detailed

### 5.1 Rel-L2 (Data fidelity)
```
score = 100 / (1 + 0.5 * ||pred - target||_2 / ||target||_2)
```
For real airfoil: only `u, v` scored. Example: error=0.1 → score≈95.2; error=1.0 → score≈66.7.

### 5.2 TKE (Turbulent Kinetic Energy)
```
TKE = 0.5 * mean_t((u - mean_t(u))^2 + (v - mean_t(v))^2)
```
Evaluated as relative L2 of this scalar field. Captures velocity fluctuations.

### 5.3 MVPE (Mean Velocity Profile Error)
Time-averaged velocity profiles at probe locations downstream. Lower = better long-horizon physical consistency.

### 5.4 Time (Efficiency)
```
r = t_neural / t_numerical    # t_numerical = 0.72896 s
score = 100 / (1 + sqrt(r))
```
- If model is 10× faster than numerical solver (r=0.1): score ≈ 76
- If model matches numerical speed (r=1): score ≈ 50
- If model is 10× slower (r=10): score ≈ 24

### 5.5 SPS (Safe Prediction)
```
sigma_global = 0.0563870
For each element:
  inside = (lower <= target <= upper)
  nil = (upper - lower) / sigma_global
For each branch (DM=Rel-L2, TKE, MVPE):
  pm = e / (0.5 + e)
  branch = mean((1 - pm) * exp(-nil))  # gated by inside
weighted = 0.5*branch_dm + 0.3*branch_tke + 0.2*branch_mvpe
sps_score = 100 / (1 + exp(-weighted))     # bounded in [50, ~73]
```

**Critical insight:** returning `lower` / `upper` intervals gives significant SPS gains. Default band is 0.1·|pred| wide; returning tighter, well-calibrated intervals (e.g., from MC dropout or ensemble) can boost SPS by ~5-10 points.

---

## 6. Submission Spec (recap)

| Item | Constraint |
|---|---|
| Archive | `submission.zip` with `submission.py` at root |
| Size | ≤ 256 MB extracted |
| Time | ≤ 3 min container execution |
| Network | None at eval |
| Image | `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime` |
| Available libs | torch 2.2.2, numpy 1.26, pillow, PyYAML, requests, tqdm, sympy, networkx |
| NOT available | scipy, pandas, matplotlib, h5py, einops, sklearn, opencv |
| Output shape | (N, T_out=20, 32, 64, 3) |
| Channels | u, v, p (p can be zeros) |
| Optional | lower / upper bounds for SPS credit |

---

## 7. Leaderboard Schema

| Column | Key | Sort | Hidden? |
|---|---|---|---|
| Final | `final_score` | desc | shown |
| Rel-L2 | `rel_l2_score` | desc | yes |
| TKE | `tke_score` | desc | yes |
| MVPE | `mvpe_score` | desc | yes |
| Time | `time_score` | desc | yes |
| SPS | `sps_score` | desc | yes |
| Raw Rel-L2 | `agg_rel_l2` | asc | yes |
| Raw TKE | `agg_tke` | asc | yes |
| Raw MVPE | `agg_mvpe` | asc | yes |
| Mean Time (s) | `mean_t_neural_s` | asc | yes |
| SPS Coverage | `sps_coverage` | desc | yes |

---

## 8. Key Dates

- **Aug 20, 2026** — Team registration deadline (hard)
- **Sep 27** — Main development ends
- **Oct 25** — Decision phase ends
- **Nov 10** — Final results
- **Nov 25** — Code submission deadline
- **Dec 6** — NeurIPS presentation

---

## 9. Important Links

- Competition site: https://realpdecompetition.github.io/
- Codabench #17363: https://www.codabench.org/competitions/17363/
- Codabench #17385 (Track 2): https://www.codabench.org/competitions/17385/
- Track 1 registration form: https://forms.gle/LYeTgUTfr4ygntNu6
- Track 2 registration form: https://forms.gle/qmMYaK5u9r86rPGKA
- RealPDEBench data: https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDEBench
- RealPDEBench code: https://github.com/AI4Science-WestlakeU/RealPDEBench
- RealPDEBench paper: https://arxiv.org/abs/2601.01829
- Public training data: Google Drive folder `1Cg23DoTuSvWXR3Mm1uRfmMNAbkyaIhrQ`
- Models (HF): https://huggingface.co/AI4Science-WestlakeU/RealPDEBench-models
- Codabench API: `https://www.codabench.org/api/competitions/17363/`
