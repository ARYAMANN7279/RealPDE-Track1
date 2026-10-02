# Execution Brief — Round 5: Cross 79.5

Continuation of `HANDOFF_GEMINI.md` + Rounds 2–4. **`submission_ENSEMBLE_FAST_v3.zip`
scored `79.341788` — a new personal best.** Banked is now 79.3418.

**Target: ≥ 79.5.** Needed: **+0.158**.

**Your role is execution. Do not submit. Do not mark anything CLOSED.** Where a judgement
call is needed, STOP and report.

---

## 0. What the live result taught us — read this first

| subscore | banked | v3 | delta |
|---|---:|---:|---:|
| rel_l2 | 94.045224 | 94.045224 | 0.000000 |
| tke | 75.998641 | 75.998641 | 0.000000 |
| mvpe | 92.873514 | 92.873514 | 0.000000 |
| **time** | 90.304009 | 90.662719 | **+0.358710** |
| **sps** | 37.480269 | 37.721250 | **+0.240981** |
| **final** | 79.246428 | **79.341788** | **+0.095360** |

Attribution using the solved weights (`0.09689·time + 0.24737·sps`):

| component | predicted | **actual** | verdict |
|---|---:|---:|---|
| centre ensemble | +0.056 | **+0.0596** | **106% — the minimax instrument is TRUSTWORTHY at fixed widths** |
| one-pass speedup | +0.075 | **+0.0348** | **46% — halve all future local timing projections** |

**Two calibration facts to carry forward:**
1. The minimax instrument called the sps gain to within 7%. **Trust it** for centre changes
   at fixed widths. (It remains untrusted for *width* changes — §Round 4 showed it
   over-predicts the tight arcsinh anchor by +0.80 sps.)
2. **A local speedup ratio transfers at roughly half strength.** Local A/B measured 0.89;
   live was 8.404 → 7.732 ms = ratio 0.920. Assume ~50% of any local timing gain.

### ★ SIXTH REAL ANCHOR — add to the calibration set in `HANDOFF_GEMINI.md` §2
```
submission_ENSEMBLE_FAST_v3.zip | centre ensemble (3xW96, a=0.95) + banked widths
                                | real sps 37.721250 | real E 0.55105
```
This is the first anchor with an *ensembled* centre at *unchanged* widths, and it is the
regime we now ship in. Use it in every future minimax fit.

---

## ★ TASK A — DRIVE INFERENCE TIME FROM 7.73 ms TO ≤ 5.0 ms

**This is the whole brief. Time alone crosses 79.5.**

| ms/sample | time score | final | |
|---:|---:|---:|---|
| 7.732 | 90.663 | 79.342 | where we are |
| 6.000 | 91.682 | 79.441 | |
| **5.000** | **92.351** | **79.505** | **← target** |
| 4.500 | 92.715 | 79.541 | |
| 4.079 | 93.040 | 79.573 | `simon-zhou` (#10) actually achieves this |

**Why this lever and not the others:**
* `time` is computed from a stopwatch, not a model. **No calibration, no distribution shift,
  no minimax uncertainty.** It is the only subscore we can predict from first principles.
* Predictions stay bit-identical ⇒ **rel_l2/tke/mvpe/sps cannot move.** Zero downside risk.
* A competitor demonstrably runs at 4.079 ms, so ≤5.0 ms is not speculative.

### A1. Profile the shipped artifact honestly — find where 7.73 ms actually goes
Live is ~2.9× our VM measurement (7.73 live vs ~2.65 local), so **the evaluation host
penalises CPU-side and host↔device work far more than our A800 box does.** Profile
`submission_ENSEMBLE_FAST_v3.zip`'s own `predict()` and report a breakdown in ms/sample:

| stage | ms/sample | % |
|---|---|---|
| FNO forward | | |
| **U-Net forward ×3 (the ensemble)** | | |
| feature construction | | |
| LUT lookup (`torch.bucketize`) | | |
| host↔device transfers | | |
| output array allocation / assembly | | |
| everything else | | |

Profile **on the CPU path too** (CUDA hidden, in a subprocess). The eval host is the slow
one, and CPU-side cost is what matters there.

### A2. ★ The obvious target: we now run THREE U-Nets sequentially
`v3` runs the ensemble as three separate forward passes. Two ways to cut that, in order of
preference:

**A2a — Batch them into ONE forward pass (outputs bit-identical, zero risk).**
The three nets share an architecture and differ only in weights. Stack them along the batch
dimension, or use grouped convolutions, so the whole ensemble is a single pass.
**Acceptance: `max|Δlower| = max|Δupper| = 0` versus `v3`.** This is a pure speed change and
must not alter a single output bit.

**A2b — Distil the 3-net ensemble into ONE net (changes outputs; measure the cost).**
Train a single W96 to match the ensemble's averaged centre field. Cuts U-Net cost ~3× and
frees ~30 MB of archive budget. **But it will give back some of the +0.0596 sps.** Report
both: the ms saved AND the minimax sps delta versus the 3-net ensemble, then compute the net
final-score change. Only worth it if `Δtime_gain > Δsps_loss`.

Do **A2a first** — it is free. Attempt A2b only if A2a alone does not reach 5.0 ms.

### A3. Eliminate remaining CPU-side work
Earlier profiling found `np.digitize` at 22% of runtime and numpy output allocation at 12%;
both were addressed. Look for what is left:
* any remaining `.cpu()` / `.numpy()` round trip inside the per-batch loop
* output arrays built by strided or per-channel writes rather than whole-slice copies
* feature construction done in numpy that could run on-device
* dtype conversions and redundant `astype` calls
* `predict()` being called more than once — model load is paid **per call** (§17.4), so
  anything cacheable at module scope should be

### A4. Batch size
Earlier measurement: bs 64 → 2.216 ms, bs 128 → 2.177 ms (saturated on our GPU). **Re-check
on the CPU path**, where the optimum is usually very different and where the eval host lives.

### A5. Deliverable
One zip, fully gated, with:
* **`max|Δprediction|` vs banked `= 0.000e+00`** (the accuracy lock)
* if A2a only: **`max|Δlower| = max|Δupper| = 0` vs v3** (pure speed change)
* measured local A/B ratio vs `v3`, ≥5 interleaved reps, GPU **and** CPU paths
* **projected live ms/sample = 7.732 × ratio, then apply the 50% transfer haircut from §0**,
  i.e. report both the optimistic and the halved estimate
* validator 13/13, extracted < 268,435,456 B, zero `.pyc`, md5 on VM and Mac

---

## TASK B — Finish the improved-ensemble training (already running)

The dropout-0.2 + 4×-data W96 nets from Round 4 are training. Finish and evaluate them.

* Report the **minimax worst/mean/best** versus the **new banked v3 baseline** (not the old
  one) — the instrument is trustworthy here, per §0.
* Report **pairwise correlation between members' `c` fields**. The current three are highly
  correlated (median |c| agreeing to 3 decimals); decorrelation is what makes an ensemble
  gain, so this number predicts whether a swap will help.
* **Size envelope is fixed at what v3 occupies**: 262,971,337 B extracted, 5.5 MB free.
  Improved nets must **REPLACE** current members. Report extracted bytes for every
  configuration.
* **Widths stay at the banked values.** v3's width ratio of exactly 1.0000 is why it is safe.

Expected value ≈ +0.06 final if the improved members deliver another +0.24 sps. **Combined
with Task A at 5.0 ms this lands ≈ 79.57.**

---

## Priority and time budget

Next slot is 00:00 UTC / 05:30 IST. **Task A is the priority** — it is worth +0.16 on its
own, carries no risk, and needs no calibration. Task B is worth ~+0.06 and is already
running, so let it finish in the background.

If Task A reaches ≤5.0 ms, ship Task A alone. If both land, ship both. If Task A stalls above
6.5 ms, report the profile and we will decide.

---

## Rules

* **Do not submit.** A human spends the daily slot.
* **Do not mark anything CLOSED.**
* Verify an artifact **exists** and its **per-file sizes match the source** before reporting
  it done. Two of the last three rounds had artifact-level errors (a truncated checkpoint,
  and a file reported built that did not exist).
* Never write scratch to `/tmp` or the root partition — use
  `/SML_DISK_24TB/rajeshr/Aryamann/UGP/`. Check `df -h /` around any extract.
* Never put a password in a shell command; `ssh vm` uses key auth.
* Before comparing two numbers, confirm same script, same metric, same baseline.
* A tighter `h_u` is **not** evidence of quality. Only minimax worst-case is — and for
  *widths* the instrument is known optimistic (+0.80 sps on the arcsinh anchor).
* **Halve every local timing projection** before quoting a live estimate (§0).
