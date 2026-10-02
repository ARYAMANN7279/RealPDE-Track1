# TASK — finish the registry: resolve the remaining UNKNOWNs by EVAL-SET FINGERPRINT

**Author: Claude. Executor: Gemini 3.1 Pro. CPU only.**
**Date: 9 Sep 2026. Banked: 79.484440.**

## 0. Rules

1. ⛔ **DO NOT USE ANY GPU.** All four are busy — a Claude subagent is on GPUs 0–1, Claude's AMSE
   arms are on 2–3. This task is file reading only.
2. ⛔ Never kill a process that is not yours. Never reboot.
3. **Write `UNKNOWN` rather than guess.** You did this correctly last time — keep doing it.

## 1. Why this is blocking

Your registry left **45 UNKNOWN**. I closed 23 of them (the `local_harness/ft_*` family — see the
ADDENDUM I appended to `CHECKPOINT_REGISTRY.md`), but one gap is actively costing us:

**`ft_md_*` scored in the TOP FIVE of a blend screen and I had to exclude it**, because its
`ft_md_w15lr3_result.json` reports `base = [95.781, 77.026, 96.14]` — which matches *neither* known
evaluation set. Until its split is established it cannot be used as a blend partner, and it may be
a genuinely good one.

## 2. ★ The method: fingerprint the eval set from the `base` triple

Every `*_result.json` / `ftaug_*.json` records the **unfine-tuned model's score on whatever eval set
that run used**. That triple is a fingerprint. Known signatures:

| base triple (rel_l2, tke, mvpe) | eval set | leakage on OUR ruler |
|---|---|---|
| **[95.4738, 75.8957, 96.0945]** | `re_lohi` (our standard 900-window ruler) | disjoint ⇒ `HONEST` if init is also KIT |
| **[96.325, 75.264, 96.861]** | `every5` (`vidx = range(0, ntraj, 5)`) | trained on `re_lohi` ⇒ `EVERY5` |
| **[95.781, 77.026, 96.14]** | ⚠️ **UNIDENTIFIED — this is your job** | unknown |

## 3. What to do

**3.1 Identify the third eval set.**
Grep the repo for the scripts that produced `ft_md_*` (try
`grep -rln "ft_md" $B/*.py $B/local_harness/*.py`, and check any `.sh` or log files). Read how that
script builds its validation index. Report the literal line, e.g. `vidx = ...`. Candidate splits
used in this project: `re_lohi` (Re in 3750/5025/25425/26700), `every5` (`range(0,ntraj,5)`),
`aoa15`, `aoa0`, `none`. If it is a fourth thing, describe it exactly.
**Then state plainly whether `ft_md_*` trained on the `re_lohi` trajectories or not.** That single
yes/no is what I need.

**3.2 Fingerprint every remaining UNKNOWN.**
For each still-UNKNOWN checkpoint with a sidecar, read its `base` triple and match it against the
table above (tolerance 0.01). Update its LEAKAGE column accordingly. Checkpoints with no sidecar at
all stay `UNKNOWN` — say how many.

**3.3 Fix the "in a scored artifact?" column.**
It currently only detects whole-checkpoint md5 matches, so it misses **soup membership** and is
wrong in at least two places. Add these known memberships explicitly:
* `ftaug_sv3_3e5_10.pth` → **member of `submission_SCREEN.zip`** (the BANKED artifact, live 79.484440)
* `local_harness/soup_v2_fp16.pth` → member of `submission_SV2.zip` (live 79.462591) **and** of `submission_SCREEN.zip`
* `local_harness/ft_long_w15lr3_best.pth` → member of `submission_LONG80.zip` (built, unsubmitted)
* `ftaug_r29_ema100_d999.pth`, `ftaug_r29_ema100_d9999.pth` → members of `submission_BLIND.zip` (live 79.466919)
Then check the build scripts (`$B/r19_screen.py`, `r30_build.py`, `r35_build.py`, `r21_build.py`)
for any other soup memberships you can establish.

**3.4 Add one column: `USABLE AS BLEND PARTNER?`**
`YES` if LEAKAGE is `EVERY5` or `ALL-DATA` (these match the regime our live-calibrated screen is
fitted on). `NO` if `INIT-LEAK` or if the checkpoint is one of `ftaug_r24_a100_*` /
`ftaug_r29_ema100_*` (known memorisers). `HONEST-ONLY` if `HONEST` (usable for honest evaluation,
but mixing it into a leaky blend distorts the screen). `UNKNOWN` otherwise.

## 4. Report back

* The literal split line for `ft_md_*` and the yes/no on whether it saw `re_lohi`.
* Updated counts per LEAKAGE class.
* The list of checkpoints newly moved out of UNKNOWN, with what moved them.
* How many remain UNKNOWN and why (no sidecar / no producing script found).

Do not draw conclusions about which checkpoint is better — that is not this task.
