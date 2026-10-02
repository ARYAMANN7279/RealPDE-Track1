# Execution Brief — Round 6: The Time-Mean Bias Correction

Runs **alongside** Round 5 (timing). Independent — different GPUs, different artifact.
Use **GPUs 2 and 3**; leave 0/1 for the Round 5 timing work.

**Do not submit. Do not mark anything CLOSED.** Full background: `project_memory.md` §35.

---

## 0. The idea, and why it is new

`tke` is computed from `u − mean_t(u)`. **Adding a correction that is CONSTANT in time
leaves that expression algebraically unchanged, so tke cannot move.** Verified empirically:
`|Δtke| = 1.8e-06` (float32 noise) on 664 honest windows.

§30.11 previously applied the **full** (time-varying) centre correction to `prediction` and
abandoned it because tke fell 2.52. **The time-constant part carries the gains without the
penalty.**

Oracle bound, measured on honest `re_lohi`:

| α | rel_l2 | tke | mvpe | Δfinal (local) |
|---:|---:|---:|---:|---:|
| 0 | 95.3069 | 78.7002 | 96.0540 | — |
| 0.50 | +0.552 | +0.000 | +1.933 | |
| **1.00** | **+0.757** | **−0.000** | **+3.946** | **+0.7257** |

**mvpe hits exactly 100** at full correction — mvpe *is* the time-averaged profile at the
probes, so it is entirely recoverable by this class of correction.

### ⛔ Why the nets we already have do NOT work here
Taking `mean_t` of existing centre-net output captures only **2–7%** of the oracle, versus
12–17% those same nets achieve on the SPS objective:

| net | Δfinal (local) | % of oracle |
|---|---:|---:|
| C4 dropout 0.2 | +0.0538 | 7% |
| shipped joint W96 | +0.0463 | 6% |
| C3 stride3 | +0.0166 | 2% |

They are trained to predict the **instantaneous** residual under L1, and the time-mean of an
L1-optimal instantaneous predictor is **not** the optimal time-mean predictor. Every one of
them also prefers α = 0.50, which is the optimiser shrinking a mis-specified signal.

**⇒ Do not reuse these nets. Train a predictor whose target IS the time-mean.**

---

## TASK A — Train a dedicated time-mean residual predictor

### A1. The task
* **Input:** the same 80 channels the centre nets use — the 20-frame input window (40ch)
  concatenated with the model's prediction (40ch).
* **Target:** `mean_t(residual)` — shape `(N, 32, 64, 2)`. **Two output channels, not 40.**
* **Loss:** start with L1. Also try L2 — the target is a conditional *mean*, so squared error
  is arguably the better-matched loss here, unlike the instantaneous case.
* **Data:** `cache_lohihonest.npz` has `XI`, `PR`, `RS`, `SC`, `wt`. Target is
  `RS.mean(axis=1)`. Use the stride-3 cache if RAM allows (`mmap_mode='r'` on extracted
  `.npy`, as you did in Round 4).
* **Split:** honest `re_lohi` (hold out Re 3750 / 5025 / 25425 / 26700). Confirm on `aoa15`
  as the second split before believing any result.

This target is genuinely easier than the instantaneous one: 20× fewer outputs, and the
chaotic component is averaged away leaving systematic bias.

### A2. Architecture
Start from the same U-Net (`train_es/centre.py`, `UNet(80, 2, w)`) — only `co` changes from
40 to 2. Sweep width {64, 96, 128} and dropout {0, 0.1, 0.2}. Round 2 showed dropout and
data volume both helped the instantaneous predictor; expect the same here.

### A3. ★ Report against the oracle, in these exact terms
For every variant, sweep α ∈ {0.25, 0.5, 0.75, 1.0, 1.25} and report the best:

| variant | α | rel_l2 | tke | mvpe | Δfinal (local) | **% of oracle (+0.7257)** |
|---|---|---|---|---|---|---|

**⚠️ REVISED THRESHOLDS — the payoff is larger than first stated.** `mvpe` is *exactly*
the time-averaged profile at the probes, so a perfect correction drives it to 100 **by
construction, with no local→real transfer discount**. And the REAL headroom is bigger than
the local one: real mvpe 92.874 → 100 is **7.13 points**, versus only 3.95 locally. **This
lever is worth MORE in reality than in our measurements** — the opposite of everything else
in this project. Recomputed against the live board (top-50 cutoff has moved to **80.401**):

| f (fraction of true time-mean captured) | final | + Round-5 timing (5.0 ms) |
|---:|---:|---:|
| 0.2 | 79.68 | 79.84 |
| 0.3 | 79.85 | 80.01 |
| 0.4 | 80.02 | 80.18 |
| 0.5 | 80.20 | **80.36** |
| **0.6** | **~80.38** | **~80.54 → TOP 50** |

**Acceptance / stopping rule:**
* **≥ 0.50** → top-50 territory once combined with timing. Build the artifact (Task B).
* **0.30 – 0.50** → a solid, real gain (79.85–80.20). Build it.
* **0.15 – 0.30** → worth having (+0.3–0.5). Report before building.
* **< 0.15** → record the ceiling and stop.

Report `f` as the fraction by which the correction reduces the **mvpe error**
(`f = 1 − err_mvpe_corrected / err_mvpe_baseline`), not as a ratio of dfinal — mvpe error
reduction is the quantity that maps directly to the table above.

**Also report `|Δtke|` for every variant.** It must be ≤ 1e-5. Anything larger means the
correction is not actually time-constant and there is a bug — stop and report.

---

## TASK B — Build the artifact (if Task A reaches f ≥ 0.15)

`prediction_new = prediction + α · mean_t(ĉ)`, broadcast over the 20 output frames.

### B1. Interaction with the existing bounds — get this right
The shipped bounds are `lower = pred + α_b·c − h`, `upper = pred + α_b·c + h`. If
`prediction` changes, the bounds must be re-derived **relative to the new prediction**, or
the two corrections will double-count.

Report both variants and their measured E:
1. **prediction corrected, bounds unchanged in absolute terms** (same `lower`/`upper` values
   as v3 — only `prediction` moves)
2. **prediction corrected, bounds re-centred on the new prediction**

Variant 1 is the safer default: the bounds keep exactly the geometry that scored 79.34, and
only the accuracy subscores move. **Prefer it unless variant 2 measurably wins.**

### B2. ⚠️ This breaks bit-identity — say so explicitly
Every artifact since the banked one has kept `prediction` bit-identical, which is why their
accuracy subscores were guaranteed. **This one does not.** Report:
* measured `rel_l2`, `tke`, `mvpe` on the honest split **and** on `aoa15`
* `|Δtke|` (must be ≤ 1e-5 — that is the whole safety argument)
* `max|Δprediction|` versus v3, and the median and 99th percentile of the correction

### B3. Gates
Everything in `TASKS_ROUND5.md` §A5 applies: validator 13/13, extracted < 268,435,456 B,
zero `.pyc`, md5 on VM and Mac, GATE 2, GATE 3 timing ratio, per-file byte diff vs source.
The extra 2-channel net is small (a W96 2-output U-Net is ≈ the same encoder cost), but
**state its MB cost** — v3 has only **5.5 MB** of headroom, so it may need to replace an
ensemble member.

---

## Context: where this leaves us, honestly

Realistic stack from the current 79.342:

| lever | expected |
|---|---:|
| Round 5 timing → 5.0 ms | +0.16 |
| this time-mean correction (at f = 0.3–0.5) | +0.51 … +0.86 |
| improved centre ensemble (Round 4 Task B) | +0.06 |
| **projected total** | **≈ 80.0 – 80.4** |

Top-50 has moved to **80.401** (rank 50, `phgelado`). At f ≥ 0.5 this route DOES
reach it once combined with the timing work — see the revised table in Task A3.

If `f` lands below 0.5, the shortfall would have to come from elsewhere: E 0.551 → 0.613
(a centre predictor ~4× better than the measured class ceiling) or a materially better base
model (rel_l2 +0.65 and tke +2.0 against a 16-config plateau). Neither has a demonstrated
route today, so **`f` is the single number that decides whether top 50 is live.**

**Measure `f` honestly and report it, whatever it is.** A well-measured ceiling is the
deliverable, not a number that flatters the method — and `f` is now the most consequential
quantity in the whole project.

---

## Rules

* Do not submit. Do not mark anything CLOSED.
* Verify an artifact **exists** and its per-file sizes match the source before reporting done.
* Never write scratch to `/tmp` or the root partition — use `/SML_DISK_24TB/.../UGP/`.
* Never put a password in a shell command; `ssh vm` uses key auth.
* Confirm same script / same metric / same baseline before comparing two numbers.
* Halve every local timing projection before quoting a live estimate.
* Label every number **measured** or **projected**.
