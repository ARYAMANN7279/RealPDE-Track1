# Execution Brief — Round 4

Continuation of `HANDOFF_GEMINI.md`, `TASKS_ROUND2.md`, `TASKS_ROUND3.md`. Read those first.

**Round 3's Task A was the most valuable work done so far.** The four-row decomposition
cleanly separated a genuine gain from a harmful one, and flagging the hybrid's index-mapping
bug rather than shipping it was exactly right. Two items need correcting, and then there is
one clear direction worth real GPU time.

**Your role is execution.** **Do not submit. Do not mark anything CLOSED.** Where a
judgement call is needed, STOP and report.

---

## 0. Status corrections

### 0.1 Task B was reported complete but the file did not exist
Round 3 reported *"the rebuilt archive was validated against `gate23_es.py` and is fully
sound for submission."* There was no `submission_ENSEMBLE_FAST_v3.zip` on the VM or the Mac —
only the original and the corrupt v2.

**I have since built and fully gated it myself.** It is done; do not rebuild it.
`submissions/submission_ENSEMBLE_FAST_v3.zip`, md5 `11a8e5dd41de33491424f4c425bec623`
(identical VM and Mac): validator 13/13 / 0 FAIL, checkpoint intact at 201,396,349 B, zero
`.pyc`, extracted 262,971,337 B (97.96% of cap), **bit-identity `max|Δpred| = 0.000e+00`**,
GATE 2 all pass, **width ratios exactly 1.0000 vs banked**, 3 centre nets present.

⇒ **Before reporting any task complete, verify the artifact exists and check its file size
against the source.** `ls -l` and a per-file byte diff. This is the second artifact-level
error in two rounds (the first was the truncated checkpoint).

### 0.2 C8 is settled — and the hybrid is now a lower priority than it looked
Round 3's own numbers, minimax worst-case vs banked:

| configuration | worst |
|---|---:|
| C8 centre + banked widths | **+0.0310** |
| banked centre + C8 widths | **−0.0772** |
| C8 centre + C8 widths (`C8_FINAL` as built) | **−0.0380** |

`C8_FINAL` would have lost. Good result, correctly obtained.

**But compare like-for-like against what we already have:**

| centre | minimax worst |
|---|---:|
| **centre ensemble** (banked + s1 + s10) | **+0.056** |
| C8 centre (W160, dropout 0.2, stride-3) | +0.031 |

**The existing ensemble beats C8's centre.** So fixing the hybrid's index bug would produce
an artifact *worse* than the one already built and gated. **Do not spend time on the hybrid
index bug.** The C8 centre is still valuable — but as an *ensemble member*, which is Task B.

---

## TASK A — ★ Validate the minimax instrument against known anchors

**Why this comes first:** every decision now rests on minimax worst-case numbers produced by
your evaluation script, and that script has never been checked against ground truth. I can
verify artifacts (file sizes, bit-identity, bound ratios) but I cannot verify a measurement
script without validating its output against something we know independently.

We have **five real leaderboard scores** on the same checkpoint (`HANDOFF_GEMINI.md` §2).
Those are ground truth. The instrument must reproduce them.

### A1. Reproduce the five anchors
For each of the five artifacts below, feed its **actual** `lower`/`upper` (from running the
zip's own `predict()`, not a reconstruction) through your evaluation pipeline and predict its
real `sps`. Report predicted vs actual:

| artifact | actual sps | your predicted sps | error |
|---|---:|---|---|
| `submission_SHIFT_v2_W96_a85.zip` (banked) | 37.4803 | | |
| `submission_SOUP_v1.zip` | 34.3656 | | |
| `submission_LUTFIX.zip` | 33.9166 | | |
| `submission_WIDE125.zip` | 33.8038 | | |
| `submission_ASYM_W96_safe_arcsinh.zip` | 27.8581 | | |

The arcsinh row matters most — it is the only **tight-bounds** anchor we own, and tight
bounds are exactly where the instrument has to be trusted.

### A2. Report the calibration set's honesty
Under the calibration that best fits all five, and under the **worst** admissible
calibration, report the predicted sps for each anchor. State the max absolute error in sps
and in final-score points.

### A3. Acceptance
* If the instrument reproduces all five anchors to within ~0.5 sps, it is trustworthy and
  every minimax number from it stands.
* If it reproduces the four wide anchors but **misses the arcsinh (tight) anchor badly**,
  say so explicitly — it would mean the instrument is unreliable in precisely the regime
  where we most need it, and every "tightening is safe" conclusion becomes suspect.
* Report the numbers either way. **Do not tune the instrument to fit the anchors** — that
  would destroy its value as an independent check.

---

## TASK B — ★ The real opportunity: ensemble the BETTER-REGULARISED centre nets

Round 2's Task C showed regularisation and data volume both improve a single centre net:

| variant | dfinal (single-calibration) |
|---|---:|
| C1 control (W96, no reg) | +0.6376 |
| C4 dropout 0.2 | +0.6856 |
| C3 stride-3 (4× data) | +0.7462 |

And separately, **ensembling** independent nets is worth +0.056 minimax.

**Nobody has combined the two.** The current shipped ensemble is three *unregularised* W96
nets differing only in data-order seed — which is why they are highly correlated (median
|c| identical to 3 decimals). Nets trained with **dropout + 4× data** should be both
individually stronger **and** more decorrelated, and ensembles gain from decorrelation.

### B1. Train an ensemble of improved members
Train **5–6 W96 centre nets** with the C3+C4 recipe (dropout 0.2, stride-3 cache), varying
only the seed. W96, not W160 — see the size constraint below.

### B2. Evaluate under MINIMAX, not single-calibration
This is the step Round 2 skipped. Report the minimax worst / mean / best over the admissible
calibrations, versus banked, for:

| configuration | worst | mean | best |
|---|---|---|---|
| current shipped ensemble (banked + s1 + s10) | +0.056 | | |
| 2 improved nets | | | |
| 3 improved nets | | | |
| 2 improved nets + C8's W160 centre | | | |
| best mix you can find within the size budget | | | |

Also report the **pairwise correlation** between members' `c` fields — that is the number
that explains whether an ensemble will gain, and it is worth knowing regardless.

### B3. ⛔ Size budget — hard constraint
The banked archive leaves **38.2 MB**. W96 = 15.3 MB fp16, W128 = 27.1 MB fp16, W160 ≈ 42 MB
fp16. The built v3 already uses 2 extra nets and sits at **97.96%** of the cap.
**State the MB cost of every configuration you report.** A configuration that does not fit
is not a result.

### B4. Keep the widths fixed
Use the **banked widths, unchanged**. The v3 artifact has width ratio exactly 1.0000 and
that is why it is safe. **Do not touch widths in this task.** A tighter `h_u` is not
evidence of quality — only minimax worst-case is.

---

## TASK C — GATE 3 timing on the built artifact (quick)

I ran every gate on `submission_ENSEMBLE_FAST_v3.zip` except timing. Run an interleaved A/B
against `submission_SHIFT_v2_W96_a85.zip`, ≥5 alternating reps, same GPU, same batch size,
and report the **ratio** (not absolute ms — the VM is noisy). Also run the CPU path in a
subprocess with CUDA hidden. Expected ratio ≈ 0.83–0.94; anything > 1.0 should be reported
prominently since the artifact is supposed to be faster.

---

## Reporting format

For each task: the exact command, the measured number, which split and how many windows,
whether the checkpoint was trained on that split, and **measured vs projected** labelled
explicitly. Append to `project_memory.md` incrementally.

## Rules

* **Do not submit.** A human spends the daily slot.
* **Do not mark any route CLOSED.**
* Never write scratch to `/tmp` or the root partition — use
  `/SML_DISK_24TB/rajeshr/Aryamann/UGP/`. Check `df -h /` around any extract.
* **Verify an artifact exists and its file sizes match the source before reporting it done.**
* Before comparing two numbers, confirm same script, same metric, same baseline.
  Single-calibration `dfinal` and minimax worst-case are **not** comparable.
* A tighter `h_u` ratio is **not** evidence of quality. Only the minimax worst case is.
* ⚠️ **Never put a password in a shell command.** `ssh vm` uses key authentication and needs
  no password. A plaintext credential appeared in the Round 3 transcript; it is being rotated.
