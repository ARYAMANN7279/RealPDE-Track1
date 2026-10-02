# TASK — build the SUBMISSION LEDGER + audit the new zip (CPU only, pure extraction)

**Author: Claude. Executor: Gemini 3.1 Pro.**
**Date: 10 Sep 2026. Banked: 79.484440. `$B` = `/SML_DISK_24TB/rajeshr/Aryamann/UGP`.**

## 0. Rules

1. ⛔ **DO NOT USE ANY GPU.** Three of Claude's `r46_*` jobs are training. This is file reading,
   `grep`, and `md5sum` only.
2. ⛔ Never kill a process that is not yours. Never reboot.
3. **Extract, do not interpret.** Write `UNKNOWN` rather than guess. Do not compute new predictions,
   do not recommend anything, do not draw conclusions. **If a number is not written down somewhere,
   it is UNKNOWN — do not reconstruct it.**
4. Scratch only under `$B/scratch/`. ⛔ Never `/tmp` (root partition ~90% full).

---

# PART A — THE SUBMISSION LEDGER (the main task)

## A.1 Why

Every submission we have ever made was preceded by a **predicted** score and followed by an
**actual** one. Those pairs are scattered across 9,000 lines of `project_memory.md` and have never
been collected. On 10 Sep `W73` missed its prediction by **−0.258** while `SCREEN` had missed by only
**−0.005**, and we suspect prediction error is systematic. **We cannot test that on two points.**

## A.2 What to produce

**`$B/SUBMISSION_LEDGER.md`** — one row per LIVE submission ever made, oldest first:

| # | date | zip filename | predicted final | actual final | error (actual − predicted) | rel_l2 | tke | mvpe | time | sps | § where recorded |

* **Source: `$B/project_memory.md` only.** Search for live results — try
  `grep -nE "79\.[0-9]{6}|78\.[0-9]{6}|77\.[0-9]{6}|76\.[0-9]{6}" project_memory.md`, and also
  `grep -niE "predicted|scored|final_score|actual"`.
* A row belongs in the ledger **only if the score actually appears as a live/Codabench result** in
  the record. ⛔ **Do not invent rows and do not include predictions that were never submitted**
  (e.g. `LONG80`, `CORNER3` were built but NEVER submitted — exclude them).
* Where a submission has full per-subscore numbers recorded, fill those columns; else `UNKNOWN`.
* If a submission's **predicted** value was never written down, put `UNKNOWN` — **do not derive it.**
* Known anchors to make sure you find (there are more — find them all):
  `SV2` 79.462591 · `FP16` 79.378881 · `SCREEN` 79.484440 (predicted 79.4895) ·
  `BLIND` 79.466919 · `W73` 79.441217 (predicted 79.6995) · `TMEAN` · `LUTCAL` · `SOUP_v1` ·
  and three early losses recorded around 78.34 / 78.30 / 76.70.

## A.3 Second table in the same file — PREDICTION ACCURACY

| zip | predicted | actual | error | was the candidate a BLEND/SOUP with a leaky partner? (YES/NO/UNKNOWN) | § |

For the last column, use ONLY what the record explicitly says about that artifact's ingredients.
`grep -n "r19_screen_best.json\|ingredients\|soup\|blend" project_memory.md` and the sections named
in the ledger. **If the record does not state the ingredients, write UNKNOWN.**

## A.4 Report back

Paste both tables. Then state: **how many submissions have a recorded predicted value**, and
**the largest and smallest prediction error you found**. Nothing else.

---

# PART B — PRE-SUBMISSION AUDIT of `submission_RECIPE.zip`

⏳ **This zip does not exist yet** (Claude is building it, ~25 min). **Poll for it**:
`while [ ! -f $B/submissions/submission_RECIPE.zip ]; do sleep 60; done`
Then run **exactly the procedure in `$B/TASKS_PRESUBMIT_AUDIT.md`**, substituting
`submission_RECIPE.zip` for `submission_W73.zip` everywhere. In particular:

1. `md5sum` of the zip and of `sim_real_fno_fp16.pth` inside it.
2. ★ **THE DUPLICATE CHECK** — md5 its three payload files against **every** `.zip` in
   `$B/submissions/`. **Does any other zip share its backbone md5?** This is the check that has
   twice caught a wasted slot. Name any match.
3. Structural: `unzip -t` clean · entry list identical to `submission_SV2.zip` · **0** `.pyc` ·
   extracted size < 268,435,456 B (expect ≈244.36 MB = 91.03%) · **186** bounds keys ·
   every non-backbone entry byte-identical to SV2.
4. A single **GO / NO-GO** line, `GO` only if all of the above pass.

⛔ Do not submit anything. ⛔ Do not rebuild or modify any zip. ⛔ Do not recommend whether to ship it.
