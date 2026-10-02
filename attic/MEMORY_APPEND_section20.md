# APPEND TO project_memory.md (could not write directly — file returned EPERM)

---

## 20. ⛔ 79 IS NOT REACHABLE WITH THIS MODEL — arithmetic + refutation (Aug 26)

Target: **+0.5434** over banked 78.4566.

### Step 1: no MEASURED signal reaches it, even at an impossible 100% transfer
| signal | ships? | best-case final |
|---|---|---:|
| ensemble K=3 (corr .558, sd 1.277 — best signal found) | ✗ 3×201MB ≫ 256MB cap | 78.91 |
| ensemble K=9 | ✗ 1.8 GB | 78.97 |
| 2-model \|soup−member\| | ~251MB, marginal | 78.79 |
| input-perturbation sensitivity | ✓ 1 ckpt | corr 0.420 — below the 0.48 that already failed (§7) |

**Every shippable measured signal caps below 79 before any transfer loss.**

### Step 2: the only route over 79 is the head we ALREADY ship, if it transferred
Local edge over constants **+0.0540**; realised **+0.0054 (~10%)**. At 100% transfer
that is real E 0.5506 → **79.18**. **The entire gap to 79 IS the head's transfer loss.**

### Step 3: Reynolds extrapolation is NOT the cause — HYPOTHESIS REFUTED
Built the split the rules imply ("unseen parameter regimes"): train on 12 interior Re,
test on the 3 lowest + 3 highest (1021 windows). Four head capacities:

| head | params | random edge | Re-extrap edge | kept |
|---|---:|---:|---:|---:|
| linear | 28 | +0.0436 | +0.0447 | 102.6% |
| tiny | 1,922 | +0.0504 | +0.0517 | 102.6% |
| small | 13,090 | +0.0545 | **+0.0552** | 101.3% |
| full (shipped) | 605,698 | +0.0550 | +0.0487 | **88.5%** |

**The full head keeps 88.5% under Re-extrapolation but only 10% in reality.**
Re-shift explains ~11 points of a ~90-point loss. The constants baseline barely moves
(0.5318 → 0.5285), so the extrapolation split is **not appreciably harder** than random.
**Our local split being "too easy" was NOT the problem — do not rebuild splits to fix this.**

⚠️ The script's projected finals (79.04–79.20) are **NOT credible**: they assume the
extrapolation edge transfers 1:1 — the exact assumption reality has refuted four times
(M55 +0.0537→+0.0056; head +0.0540→+0.0054). At the measured 10% rate the best head
beats the shipped one by ~**+0.01 final**.

### Real, reusable finding: head capacity is ~46× too large
`small` (13k params) beats `full` (606k) on BOTH splits (+0.0552 vs +0.0487 extrap).
Worth adopting on principle; worth ~+0.01 at real transfer rates, so **not** slot-worthy.

### The transfer loss is still UNEXPLAINED (this is the open problem)
Not Re-shift (refuted). Untested candidates:
- angle-of-attack shift (18 Re × ~4-5 AoA; only Re was tested)
- a different window-cutting procedure in the private eval
- a **level** mismatch in `global_scale`: local E 0.5857 vs real 0.5020 is a gap of
  **0.084 in level**, not just in edge — the LUT half-widths may be systematically
  mis-scaled live, which no amount of head redesign fixes

### DECISION: do not submit. 78.4566 stands.
Per the standing rule (submit only if >79 is demonstrable), nothing qualifies.

### Added to §12 discipline
- **Check the deployment budget (size/time) BEFORE running the experiment.** The
  disagreement study was completed in full before noticing 3 checkpoints cannot fit.
- **A projection resting on an assumption reality already refuted is not a projection.**
  The 79.04–79.20 numbers looked like success and were worthless.
- **When a hypothesis predicts a 90% effect and measures an 11% one, it is wrong** —
  do not ship on it anyway because the projected number is attractive.
