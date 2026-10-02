# Execution Brief — Round 2

Continuation of `HANDOFF_GEMINI.md`. Read that first if you have not; this brief assumes
its context (paths, GPU etiquette, scoring model, the CLOSED list, the three traps).

**Your role in this round is execution.** Every task below is mechanical and has explicit
acceptance criteria. **Where a judgement call is required, STOP and report — do not decide.**
Report measured numbers; do not interpret them into a verdict.

---

## 0. A correction you must internalise before starting

Round 1 concluded that the out-of-sample centre nets "capture about **4%** of the in-sample
ceiling", and marked the centre route **CLOSED** in `project_memory.md` §34.1.

**That conclusion is wrong and the route is NOT closed.** The error:

* `+0.058` is the **ensemble's marginal gain** — the *incremental* value of ensembling on
  top of the already-shipped centre net, measured in the **live minimax** scale.
* `+1.409` is the in-sample probe's **total** value — measured against **no centre at all**
  (α = 0), in the **offline dfinal** scale.

Those are different quantities against different baselines. Compared like-for-like, in the
same metric, the shipped net is `+0.543` and the best net (D, W160) is `+0.613` against a
ceiling of `+1.409` — i.e. **38–44%**, not 4%.

Both scripts use the identical formula (`dfinal = 14.855·(E(λ(r − αc)) − E(λr))`), which is
why they are comparable — verified in `train_es/centre.py:5` and `train_es/centre_eval.py`.

**Second correction:** `train_es/centre.py:23` hardcodes `KFIN = 0.217*100*WBANK`. `0.217`
is the **superseded** sps weight; the correct value is **`0.24737`** (see `HANDOFF_GEMINI.md`
§2). Every `dfinal` produced by that script is therefore **understated by 14%**.

**Implication, which Task B exists to confirm:** the gap between +0.62 (what we achieve) and
+1.61 (what memorisation achieves), both corrected, is a **generalisation gap of ≈ +0.99** —
not an information limit. Generalisation gaps respond to different tools than the
architecture/capacity/loss axis §33.6 ruled out.

---

## TASK A — Repair and fully verify `submission_ENSEMBLE_FAST.zip`

**Why:** the artifact fails GATE 1 and its most important gate was never run.

### A1. Strip the `__pycache__` entries
The archive contains **7 `.pyc` files** (21,710 B total, `cpython-311` tags). GATE 1 forbids
them, and they are the **only** entry-list difference from the banked artifact.

```bash
# on the VM, in /SML_DISK_24TB/rajeshr/Aryamann/UGP
rm -rf /tmp/rebuild && mkdir -p /tmp/rebuild
unzip -q submissions/submission_ENSEMBLE_FAST.zip -d /tmp/rebuild
find /tmp/rebuild -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null
find /tmp/rebuild -name '*.pyc' -delete
cd /tmp/rebuild && zip -q -r -X -D \
  /SML_DISK_24TB/rajeshr/Aryamann/UGP/submissions/submission_ENSEMBLE_FAST_v2.zip .
```
`-D` is mandatory — without it you get phantom directory entries that fail the entry diff.

### A2. Re-run GATE 1 on the NEW bytes
```bash
cd /SML_DISK_24TB/rajeshr/Aryamann/UGP/local_harness
$PY _val.py ../submissions/submission_ENSEMBLE_FAST_v2.zip --height 32 --width 64
unzip -l ../submissions/submission_ENSEMBLE_FAST_v2.zip | tail -1     # extracted bytes
unzip -t ../submissions/submission_ENSEMBLE_FAST_v2.zip | tail -1
diff <(unzip -l ../submissions/submission_SHIFT_v2_W96_a85.zip | awk 'NR>3{print $4}' | sort) \
     <(unzip -l ../submissions/submission_ENSEMBLE_FAST_v2.zip | awk 'NR>3{print $4}' | sort)
```
**Acceptance:** validator **13/13, 0 FAIL**; extracted **< 268,435,456 B**; `unzip -t` clean;
**zero** `.pyc`/`__pycache__`; entry diff shows only the deliberate extra-net files.

### A3. ★ THE GATE THAT WAS NEVER RUN — point-prediction bit-identity
This is the gate that protects our 79.25 accuracy subscores. Extract **both** zips, call
**each one's own** `predict()` on the **same** input windows, and compare.

**Acceptance — all three must hold exactly:**
```
max|prediction_new − prediction_banked| == 0.0     (EXACTLY zero, not "small")
prediction_new[..., 2] == 0  everywhere
lower <= upper  everywhere; all three arrays finite
```
If `max|Δprediction|` is **not exactly 0**, STOP and report. It means rel_l2/tke/mvpe will
move and the entire "guaranteed" argument collapses.

Also record: `h_u` median, `h_v` median, and `std(h_u)` (must be > 0 — a constant band is a
silent failure mode that has bitten this project before).

### A4. GATE 3 timing
Interleaved A/B against the banked artifact, ≥ 5 reps alternating, same GPU, same batch.
Report the **ratio**, not absolute ms (the VM is noisy — the same zip has measured 2.69 and
4.57 ms/sample minutes apart). Also run the **CPU path in a subprocess** with CUDA hidden.

### A5. Deliverable
`submission_ENSEMBLE_FAST_v2.zip` on both VM and Mac, **md5 identical on both**, plus a
gate table. Report the md5.

---

## TASK B — ★ The decisive re-measurement (do this even if Task A is slow)

**Why:** Round 1's "4%" made the centre route look hopeless. Task B establishes the true
number. **This single table decides where all remaining effort goes.**

### B1. Fix the stale constant
In `train_es/centre.py` (and `centre_eval.py` if it has its own copy), change the sps weight
from `0.217` to `0.24737`. Record both the old and new value of every number you report.

### B2. Score all three centres with ONE script, ONE dataset, ONE metric
Use `train_es/centre_eval.py` (or whichever script you use — but **the same one for all
three**) on the **same** honest `re_lohi` holdout, and report a single table:

| centre | checkpoint | corr_u | corr_v | best α | dfinal (old K) | dfinal (K=0.24737) |
|---|---|---|---|---|---|---|
| none (α = 0) | — | — | — | — | 0 | 0 |
| shipped / banked | `joint_SHIFT_v2_W96.pth` | | | | | |
| best out-of-sample | net D, W160 | | | | | |
| **in-sample probe** | `train_es/ctr_P_insample160.pth` | | | | | |
| oracle β = 1.00 | true residual | | | | | |

**Every row must come from the same script invocation style and the same window set.** That
is the whole point — Round 1's error was mixing two scales.

### B3. Report these three ratios explicitly
```
shipped / in-sample      =  ?     (Round 1 implied 4%; I expect ≈ 38%)
best_net / in-sample     =  ?     (I expect ≈ 43%)
in-sample / oracle       =  ?     (tells us how much even memorisation leaves on the table)
```
**Acceptance:** if `shipped / in-sample` lands anywhere near 4%, something is wrong with my
reasoning — **report it and stop**, do not proceed to Task C. If it lands near 38–44%, my
correction is confirmed and Task C is live.

### B4. Also report the absolute headroom
`(in-sample dfinal − best_net dfinal)` at `K = 0.24737`. This is the size of the prize.

---

## TASK C — Attack the generalisation gap (ONLY if B confirms ≈ 38–44%)

§33.6 ruled out **architecture, capacity and loss**. A generalisation gap responds to a
different axis: **data volume, augmentation, and regularisation**. None of these have been
tried on the centre predictor.

Train each variant with the **same** W96 architecture, the **same** honest `re_lohi` holdout,
and the **same** eval as Task B, so every result is comparable. Report `dfinal` and
`corr_u/corr_v` for each. Run them on separate GPUs where possible.

| # | variant | what to change | rationale |
|---|---|---|---|
| C1 | **control** | the shipped recipe, retrained | anchors the comparison |
| C2 | **2× windows** | rebuild the cache at stride 5 instead of 10 | 5,274 windows from 66 trajectories is small; this is the cheapest generalisation lever |
| C3 | **4× windows** | stride 2–3 if RAM allows (watch `free -g`) | does the curve keep climbing? |
| C4 | **dropout** | dropout 0.1 / 0.2 in the U-Net | standard regularisation, never tried here |
| C5 | **weight decay** | 1e-3 and 1e-2 (currently 1e-4) | ditto |
| C6 | **noise augmentation** | Gaussian noise on the input window, σ = 1% and 3% of per-channel std | augmentation on the *input*, not the target |
| C7 | **early stop** | pick the epoch by held-out dfinal, not final epoch | if C1 overfits, the curve will show it |

**Report the learning curves, not just the endpoints** — for C2/C3 specifically, whether
`dfinal` is still rising with data volume is the single most informative output, because it
tells us whether more data would keep paying.

**Do not ensemble anything in Task C.** Ensembling is a separate, already-measured axis; mixing
it in confounds the result.

⚠️ Remember the size constraint (`HANDOFF_GEMINI.md` §4b): only **two extra W96 nets** fit.
Any Task C winner must be deployable as a **single** net replacing the current one, or as at
most two. State the MB cost of anything you propose.

---

## Reporting format

For every task, report:
1. **What you ran** — exact script and arguments.
2. **The measured number** — and which split, how many windows, and whether the checkpoint
   was trained on that split (see the `every5` trap in `HANDOFF_GEMINI.md` §4c).
3. **Measured vs projected** — label every number as one or the other. Never present a
   projection as a measurement.
4. **Anything that surprised you** — including anything that contradicts this brief.

Append to `project_memory.md` as a new numbered section **incrementally, as you go**. Four
previous sessions lost unwritten work to interruptions.

## Rules

* **Do not submit anything.** A human decides when to spend the daily slot.
* **Do not mark any route CLOSED.** Report the numbers; the verdict is decided elsewhere.
* If a result contradicts this brief, that is **valuable** — report it prominently rather
  than reconciling it silently.
* Do not re-run anything on the CLOSED list in `HANDOFF_GEMINI.md` §3.
* If you find yourself comparing two numbers, first verify they came from the **same script,
  the same metric and the same baseline**. That single check is what Round 1 missed.
