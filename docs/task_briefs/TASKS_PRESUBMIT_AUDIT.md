# TASK — PRE-SUBMISSION AUDIT of the three candidate zips (CPU only, pure bookkeeping)

**Author: Claude. Executor: Gemini 3.1 Pro.**
**Date: 9 Sep 2026. Banked: 79.484440. `$B` = `/SML_DISK_24TB/rajeshr/Aryamann/UGP`.**

## 0. Rules

1. ⛔ **DO NOT USE ANY GPU.** GPUs 1 and 2 are running Claude's `r38_*` jobs; GPUs 0 and 3 belong to
   ANOTHER USER. This task is `md5sum`, `unzip -l`, `unzip -t` and file reading only.
2. ⛔ Never kill a process that is not yours. Never reboot.
3. ⛔ **Do not build, rebuild, modify or delete any zip.** Read-only. If something looks wrong,
   **report it — do not fix it.**
4. Write `UNKNOWN` rather than guess.

## 1. Why this matters

We spend **one submission slot per UTC day and a FAILED submission still burns it.** Twice we have
nearly wasted a slot on an artifact that was not what its filename claimed:
* `submission_SV2_V4_GOLD.zip` turned out **byte-identical to `submission_FP16.zip`**, which had
  already been scored 79.378881 — resubmitting it would have bought nothing.
* `submission_SV3_MAX.zip` turned out to be the Aug-22 SOUP_v1 stack (66 bounds keys instead of 186
  — no centre ensemble, no TIER2D meta-head).

**The next slot is going to `submission_W73.zip`.** Before it is spent, prove it is a genuinely new
artifact and that it is structurally sound.

## 2. The three candidates

| zip | expected zip md5 | expected backbone md5 |
|---|---|---|
| `submission_W73.zip` | `3bdf77b97c3ef5b5c5c0e79865982d98` | `8d112b78550e87fa91241c15af8eae84` |
| `submission_CORNER3.zip` | `7b6b125dfe27d7453ca82e4753530ea8` | `fd4a4231a7402982d06c3917b96e05c8` |
| `submission_LONG80.zip` | `6bae48b8b5ba1478614baf95fa10c6fc` | `43c7823d7004c9a4eb4f895e82d5be6a` |

## 3. What to do

**3.1 Confirm the fingerprints above.** For each of the three: `md5sum $B/submissions/<zip>` and
`unzip -p $B/submissions/<zip> sim_real_fno_fp16.pth | md5sum`. **Report any mismatch loudly** —
a mismatch means the file on disk is not the one that was gated.

**3.2 ★ THE DUPLICATE CHECK — the one that has actually saved us.**
For **every** `.zip` in `$B/submissions/`, compute the md5 of its three payload files:
`sim_real_fno_fp16.pth`, `bounds_assets.npz`, `submission.py`.
Build a table `zip | backbone md5 | bounds md5 | submission.py md5`.
Then answer explicitly: **does any OTHER zip share W73's backbone md5, or all three md5s?**
If yes, name it. (These have been scored live and must never be resubmitted: `submission_SV2.zip`,
`submission_FP16.zip`, `submission_TMEAN.zip`, `submission_LUTCAL.zip`, `submission_SOUP_v1.zip`,
`submission_SCREEN.zip`, `submission_BLIND.zip`.)

**3.3 Structural verification of W73, CORNER3, LONG80.** For each report:
* `unzip -t` result (clean / errors)
* the **entry list**, and whether it is **identical to `submission_SV2.zip`'s** (`unzip -l`, compare
  names only, ignore sizes/dates)
* count of `.pyc` entries — **must be 0**
* the **total extracted size** (sum of uncompressed sizes) in bytes, and that as a % of the
  **268,435,456 B** cap. Expected ≈ 244.36 MB = 91.03%.
* the number of keys in `bounds_assets.npz`
  (`unzip -p <zip> bounds_assets.npz > /tmp/x.npz` ⛔ NO — write to `$B/scratch/x.npz`, never
  `/tmp`, the root partition is ~90% full and once truncated a file silently. Then
  `python -c "import numpy;print(len(numpy.load('$B/scratch/x.npz').files))"` using
  `/SML_DISK_24TB/rajeshr/Aryamann/env/bin/python`.) **Expected 186.** Flag anything ≠ 186.
* whether every entry OTHER than `sim_real_fno_fp16.pth` is byte-identical to the corresponding
  entry in `submission_SV2.zip` (md5 each entry and compare).

**3.4 Registry cross-check.** In `$B/CHECKPOINT_REGISTRY.md`, confirm the blend partner
`local_harness/ft_long_w15lr3_best.pth` is classified **`EVERY5`** and
`local_harness/ft_md_w10lr3_best.pth` is **`EVERY9`**. Report both rows verbatim.

## 4. Report back

A go/no-go line for each of the three zips — `GO` only if: md5s match §2, `unzip -t` clean, entry
list identical to SV2, 0 `.pyc`, 186 bounds keys, extracted size < 268,435,456 B, and **no other zip
shares its backbone md5**. Otherwise `NO-GO` with the exact failing check.

⛔ **Do not recommend which one to submit — that decision is mine and is already made (W73).**
Your job is only to prove the artifact is what we think it is.
