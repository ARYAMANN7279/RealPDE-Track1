# Round 5 — Answer to the W64/W48 question: NO

## The decision

**Do not distil, and do not train a W64 or W48 replacement.** Your own measurement makes
the case against it.

The 3-net centre ensemble is worth **+0.0596 final, MEASURED LIVE** — it is the entire sps
gain of the submission that just scored 79.341788 (sps 37.480269 → 37.721250). That is not
a projection; it is a leaderboard fact.

| scenario | ms | Δtime | **net vs v3** |
|---|---:|---:|---:|
| distilled W96 (your 6.49 ms projection) | 6.49 | +0.069 | **+0.010** |
| distilled W96, if it keeps half the ensemble | 6.49 | +0.069 | +0.040 |
| distilled W64 (faster, worse centre) | 5.50 | +0.130 | +0.071 |
| **A2a batching (outputs bit-identical)** | 7.00 | +0.040 | **+0.040** |
| **A2a + CPU-side cleanup** | 5.00 | +0.164 | **+0.164** |

Distillation buys +0.069 of time and throws away −0.0596 of measured sps. **Net +0.010.**
A narrower net buys more time but loses more centre quality, so it moves along a trade curve
we do not want to be on at all.

---

## What to do instead — in this order

### 1. ★ A1: the profile, which has not been reported
Round 5 §A1 asked for a stage-by-stage breakdown of where the 7.73 ms goes. Please produce
it before optimising anything further:

| stage | ms/sample | % |
|---|---|---|
| FNO forward | | |
| U-Net forward ×3 (the ensemble) | | |
| feature construction | | |
| LUT lookup (`torch.bucketize`) | | |
| host↔device transfers | | |
| output array allocation / assembly | | |
| everything else | | |

**Profile the CPU path specifically** (CUDA hidden, in a subprocess). The evaluation host is
CPU-bound relative to our A800 box — live is ~2.9× our local measurement — so CPU-side cost
is what actually matters. FNO alone was ~2.2 ms and the board floor is ~4.1 ms, so roughly
**3.6 ms is overhead** and we do not currently know how it splits.

### 2. ★ A2a: batch the three nets into ONE forward pass — this was skipped
Round 5 §A2 said: *"Do A2a first — it is free. Attempt A2b only if A2a alone does not reach
5.0 ms."* A2a keeps the ensemble **and** saves time; A2b (distillation) is the lossy
fallback. Please do A2a.

The three nets share an architecture and differ only in weights. Stack them along the batch
dimension, or use grouped convolutions, so the whole ensemble runs as a single pass.

**Acceptance: `max|Δlower| = max|Δupper| = 0` versus `submission_ENSEMBLE_FAST_v3.zip`.**
This is a pure implementation change and must not alter a single output bit. If it does,
something is wrong — stop and report.

### 3. A3: eliminate remaining CPU-side work
Guided by the profile from step 1. Candidates from earlier rounds:
* any `.cpu()` / `.numpy()` round trip inside the per-batch loop
* output arrays built by strided or per-channel writes rather than whole-slice copies
* feature construction in numpy that could run on-device
* redundant `astype` / dtype conversions
* anything cacheable at module scope — model load is paid **per `predict()` call**

### 4. A4: re-check batch size on the CPU path
bs 64 → 2.216 ms and bs 128 → 2.177 ms were measured **on the GPU**, where it is saturated.
The CPU optimum is usually very different, and the CPU path is where the eval host lives.

---

## Only if steps 1–3 stall above ~6.5 ms

Then distillation becomes worth reconsidering — but distil to **W96, not W64/W48**, and
report the **minimax sps loss** alongside the time gain so the net can be computed. The rule
is `Δtime_gain > Δsps_loss`, and at W64 that inequality gets harder, not easier.

---

## Note on process

Two things to keep in mind, neither serious:

* You used `pkill` to clear stalled evaluations. That works, but the standing warning is
  never to `pkill -f <pattern>` where the pattern also matches your own ssh command string —
  it kills the session. Prefer killing by explicit PID.
* Duplicate evaluation processes piling up and thrashing memory is worth guarding against:
  check `ps -eo pid,etime,args | grep [p]ython3` before launching, and prefer one queued
  runner over parallel launches of the same script.

## Unchanged rules

* Do not submit. Do not mark anything CLOSED.
* Verify an artifact exists and its per-file sizes match the source before reporting it done.
* Never write scratch to `/tmp` or the root partition.
* Halve every local timing ratio before quoting a live estimate — and apply the halving to
  the **ratio**, not to absolute milliseconds (you did this correctly).
* Label every number **measured** or **projected**.
