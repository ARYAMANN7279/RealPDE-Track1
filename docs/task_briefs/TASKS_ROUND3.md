# Execution Brief — Round 3

Continuation of `HANDOFF_GEMINI.md` and `TASKS_ROUND2.md`. Read both first.

**Round 2 produced good work.** The KFIN fix landed correctly, the Task C sweep is sound,
and the generalisation-gap finding (dropout and data volume both beating a much larger
network) is a real and valuable result. Two things now need fixing before anything is
submitted, and one of them is my fault.

**Your role is execution.** Every task has explicit acceptance criteria. **Where a
judgement call is required, STOP and report — do not decide, and do not submit.**

---

## 0. Two corrections

### 0.1 ⛔ `submission_ENSEMBLE_FAST_v2.zip` is CORRUPT — my instruction caused it

```
sim_real_fno_fp16.pth   in v1 : 201,396,349 B
                        in v2 :  41,119,744 B   <-- truncated to 20%
```

My Round-2 command said `unzip -q ... -d /tmp/rebuild`. `/tmp` is on the **47 GB root
partition**, which was already 88% full. The unzip exhausted it mid-write on the largest
file, and `zip` then archived the truncated result. That is also what froze the VM — it was
a disk-space failure, not RAM.

**This failure is nearly invisible:** `unzip -t` reports clean (the *archive* is consistent;
the *content* is truncated), the entry list is a correct 63 entries, and the validator
likely passes because it never loads the full model. **Only a byte-size comparison against
the source catches it.**

⇒ **Never use `/tmp` or any root-partition path on this VM.** All scratch work goes under
`/SML_DISK_24TB/rajeshr/Aryamann/UGP/`. Check `df -h /` before and after any extract.

### 0.2 ⚠️ `submission_C8_FINAL.zip` sits in a regime with a KNOWN live failure

The archive itself is fine — checkpoint intact (201,396,349 B exactly), 240,892,384 B
extracted = 89.74% of the cap, bit-identical point predictions. The concern is the bounds:

| artifact | h_u median | live score |
|---|---:|---:|
| banked `SHIFT_v2_W96_a85` | 0.776 | **79.246** |
| `ASYM_W96_safe_arcsinh` | 0.3876 | **76.705** (sps 37.48 → 27.86) |
| **`C8_FINAL`** | **0.3809** | unknown |

**C8 is tighter than the artifact that collapsed.** `HANDOFF_GEMINI.md` §4a records that the
`h_u` ratio is a *local* metric which **anti-correlates** with live score across every point
we have measured. "Crushing the h_u ratio record" is therefore not evidence of quality.

**However — C8 is not automatically a repeat**, and Task A exists to settle which it is.
Arcsinh squeezed widths around *the same* centre. C8 changed **both**: a better-regularised
centre (dropout + 4× data) *and* tighter widths. If the centre genuinely improved
out-of-sample, the post-centring residual really is smaller and the tighter widths are
**earned**. That is plausible. It is also unmeasured — C8's only justification is
`centre.py`'s **single-calibration** `dfinal` at `lam = 2.225`, and single-calibration
numbers are precisely what has failed to transfer three times.

---

## TASK A — ★ THE DECISIVE MEASUREMENT: minimax-score C8

**Do this before anything else. It decides tomorrow's submission.**

Every live-safe number we own came from the **minimax over the 362 admissible
calibrations** (the instrument behind §32.4 and §32.10). C8 has never been through it.

### A1. Score C8's bounds under the full calibration set
Use the same minimax instrument that produced §32.10's table. For **each** of the 362
admissible calibrations, compute the final-score delta of C8's `(centre, width)` policy
relative to the **banked artifact**, and report:

| quantity | value |
|---|---|
| worst-case Δfinal over 362 calibrations | |
| mean Δfinal | |
| best-case Δfinal | |
| number of calibrations where Δfinal < 0 | |

Use `K = 0.24737 · 100 · 0.684538` (the corrected sps weight).

### A2. Decompose it — centre vs width
This is the question that matters, so isolate the two changes:

| # | configuration | worst | mean | best |
|---|---|---|---|---|
| 1 | banked centre + banked widths | 0 (reference) | 0 | 0 |
| 2 | **C8 centre + banked widths** | | | |
| 3 | banked centre + **C8 widths** | | | |
| 4 | C8 centre + C8 widths (= the built zip) | | | |

Row 2 is the key row: **it measures whether the new centre is genuinely better, with the
dangerous width change held out.** Row 3 measures the width change alone.

### A3. Acceptance / decision rule — report, do not act
* If **row 4 worst-case > 0** → C8 is live-safe by our own standard. Report it.
* If **row 2 worst-case > 0 but row 4 worst-case < 0** → the centre is a genuine gain and
  the width squeeze is what breaks it. Report both; a centre-only variant becomes the
  candidate.
* If **row 2 worst-case ≤ 0** → the improvement was single-calibration overfitting. Report
  plainly.

**Do not submit under any outcome. Do not mark anything CLOSED.**

---

## TASK B — Rebuild `ENSEMBLE_FAST` correctly (the sound guaranteed play)

Independent of Task A. This is the artifact whose components are already minimax-verified:
speedup **+0.075** (byte-identical outputs) plus centre ensemble **+0.052 worst case** ≈
**79.37**.

### B1. Rebuild on the big disk
```bash
cd /SML_DISK_24TB/rajeshr/Aryamann/UGP
df -h /                                     # record BEFORE
rm -rf work_rebuild && mkdir -p work_rebuild
unzip -q submissions/submission_ENSEMBLE_FAST.zip -d work_rebuild
echo "exit=$?"                              # MUST be 0
find work_rebuild -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null
find work_rebuild -name '*.pyc' -delete
cd work_rebuild && zip -q -r -X -D ../submissions/submission_ENSEMBLE_FAST_v3.zip .
cd .. && df -h /                            # record AFTER
```

### B2. ★ MANDATORY: per-file byte comparison against the source
The check that would have caught the corruption. For **every** entry in the new zip,
compare its uncompressed byte size to the same entry in `submission_ENSEMBLE_FAST.zip`:
```bash
diff <(unzip -l submissions/submission_ENSEMBLE_FAST.zip    | awk 'NR>3{print $1, $4}' | sort -k2) \
     <(unzip -l submissions/submission_ENSEMBLE_FAST_v3.zip | awk 'NR>3{print $1, $4}' | sort -k2)
```
**Acceptance:** the ONLY differences are the 7 removed `.pyc` entries. **Any size change on
any other file means the rebuild is corrupt — stop and report.**
Explicitly confirm: `sim_real_fno_fp16.pth == 201,396,349 B`.

### B3. Full gates on the new bytes
* validator `_val.py <zip> --height 32 --width 64` → **13/13, 0 FAIL**
* extracted **< 268,435,456 B**; `unzip -t` clean; **zero** `.pyc`
* **GATE 2 / bit-identity — still never actually run.** Extract both this zip and
  `submission_SHIFT_v2_W96_a85.zip`, call **each one's own** `predict()` on the same
  windows, and require:
  ```
  max|prediction_new − prediction_banked| == 0.0      (EXACTLY zero)
  prediction[..., 2] == 0 everywhere
  lower <= upper everywhere; all arrays finite
  std(h_u) > 0                                        (a constant band is a silent failure)
  ```
* GATE 3: interleaved A/B vs banked, ≥5 alternating reps, report the **ratio**; plus the
  CPU path in a subprocess.
* md5 identical on VM and Mac — report it.

---

## TASK C — Only if time remains, and only after A and B

If Task A's row 2 shows the C8 centre is genuinely better under minimax, build **one**
variant: **C8's centre + the banked widths**, at α re-swept over {0.85, 0.95, 1.00}. That
isolates the safe half of C8. Gate it exactly as in B3. Report; do not submit.

---

## Reporting format

For each task: what you ran (exact command), the measured number, which split and how many
windows, whether the checkpoint was trained on that split, and **measured vs projected**
labelled explicitly.

Append to `project_memory.md` incrementally as a new numbered section. Include the disk-space
post-mortem from §0.1 — the "archive tests clean but content is truncated" failure mode is
worth recording permanently.

## Rules

* **Do not submit.** A human spends the daily slot.
* **Do not mark any route CLOSED.** Report numbers; verdicts are decided elsewhere.
* Never write scratch data to `/tmp` or the root partition. Check `df -h /` around any extract.
* Before comparing two numbers, confirm they came from the **same script, the same metric and
  the same baseline**. Single-calibration `dfinal` and minimax worst-case are **not**
  comparable, and conflating them is what produced both of the last two errors.
* A tighter `h_u` ratio is **not** evidence of quality. Only the minimax worst case is.
