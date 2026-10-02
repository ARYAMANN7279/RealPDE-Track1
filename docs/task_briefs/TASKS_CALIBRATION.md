# TASK — refit the local→live calibration across ALL cached anchors (CPU only, pure computation)

**Author: Claude. Executor: Gemini 3.1 Pro. Date: 14 Sep 2026. Banked: 79.487049 (SPEED_SAFE).**
`$B` = `/SML_DISK_24TB/rajeshr/Aryamann/UGP`. Python: `/SML_DISK_24TB/rajeshr/Aryamann/env/bin/python`.

## 0. Rules
1. ⛔ **NO GPU.** GPUs 0–1 are training, GPU 2 is reserved. This is numpy on cached arrays.
2. ⛔ Never kill a process that is not yours. Never reboot. Scratch only under `$B/scratch/` (never `/tmp`).
3. ⛔ **Use the kit's own `scoring.py` functions. Do NOT reimplement any metric.**
4. **Report numbers only. No forecasts of live scores, no recommendations.** Write `UNKNOWN` rather than guess.

## 1. Why
To predict what a new submission will score on every subscore, we calibrate local scores against live
scores of artifacts we already submitted. The current calibration (§100) uses only **5** anchors that mostly
share one backbone. The bounds agent cached predictions and bounds for **15** submitted artifacts; refitting on
all of them tells us how reliable each channel's local→live transfer really is.

## 2. Inputs (already on disk)
`$B/agents/bounds/cache734/`: for each TAG in
`ARCSINH BLIND ENSFASTv3 FP16 LUTCAL LUTFIX RECIPE2 SCREEN SHIFT85 SOUPv1 SV2 TIER2D TMEAN W73 WIDE125`
the float32 arrays `(N,20,32,64,2)` (channels u,v): `P_<TAG>.npy` prediction · `C_<TAG>.npy` ·
`HD_<TAG>.npy` · `HU_<TAG>.npy`, plus `TGT.npy` (the targets). Window starts: `$B/_agent2/winset.npz`.
Reconstruct exactly (from `$B/agents/bounds/mk_anchor_cache.py`'s docstring):
```
centre = P + C      lower = centre - HD      upper = centre + HU
```
Load with `mmap_mode="r"` and process in chunks — each file is 240 MB.

## 3. Steps
**3.1 Find the kit's metric functions.** In `$B/starting_kit_v9/realpde_t1_starting_kit_v9/scoring.py`, locate the
functions the kit uses for rel_l2, TKE, MVPE and **SPS** (grep `def `), and how its top-level scoring applies the
SPS mask (airfoil body / outside field of view). Use them exactly as the kit's own scoring entry point does.
Report the function names you used. Note that channel 2 (`p`) is not in the cache — append a zero channel if a
function requires `C=3`.

**3.2 ★ GATE — must pass before anything else.** For `TAG=SCREEN` compute local `sps`. It must reproduce
`sps_local = 52.9198` and overall coverage `0.9561` from `$B/agents/bounds/calib_SCREEN.json` (tolerance 0.01 /
0.001). **If it does not match, STOP and report both numbers and your function calls.** A failed gate IS the finding.

**3.3** For every TAG compute local `rel_l2`, `tke`, `mvpe`, `sps` scores and local coverage.

**3.4** Join each TAG with its **live** subscores. Sources: `$B/SUBMISSION_LEDGER.md`, and the table in
`$B/agents/bounds/live_E_table.out` (sps rows). If a live subscore is not written down, `UNKNOWN`. `TIER2D` has no
live submission — keep it in the local table only.

**3.5** For each channel (rel_l2, tke, mvpe, sps): fit `live = a*local + b` by least squares over the anchors that
have both, report `a`, `b`, max in-sample residual, and **leave-one-out** residual for every anchor. Then refit
using ONLY the anchors whose backbone is fine-tuned (exclude SOUPv1-era if their backbone differs — use the
`backbone` column of `live_E_table.out`) and report that fit too.

## 4. Output
`$B/CALIBRATION_TABLE.md` with: the gate result; the per-TAG local table; the joined local/live table; both fits per
channel with residuals and LOO residuals. Paste the gate result and the fit summaries in your reply.
⛔ Do not interpret, forecast, or recommend.
