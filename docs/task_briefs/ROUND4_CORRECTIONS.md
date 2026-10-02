# Round 4 — Two Mid-Flight Corrections

Good progress. Task C is complete and clean. Two things need correcting before Task B
produces an artifact, and one is a hard blocker.

---

## ⛔ CORRECTION 1 — The 38.2 MB headroom is measured against the WRONG artifact

Your note says:

> "The `submission_ENSEMBLE_FAST_v3.zip` artifact has exactly 38.2 MB of headroom. A single
> FP16 W96 net weighs 15.3 MB. Therefore the absolute mathematical limit for the ensemble is
> adding 2 extra nets (totaling 30.6 MB), leaving 7.6 MB of safe headroom."

**38.2 MB is the headroom of the BANKED artifact, which contains ONE centre net. `v3` already
contains THREE.** Measured:

| artifact | centre nets | extracted | free vs 256 MiB cap |
|---|---:|---:|---:|
| `submission_SHIFT_v2_W96_a85.zip` (banked) | 1 | 230,262,703 B | **38.2 MB** |
| `submission_ENSEMBLE_FAST_v3.zip` | **3** | 262,971,337 B | **5.5 MB** |

Confirmed from the file itself: `bounds_assets.npz` grows 28,409,725 B → 61,115,258 B, and
its keys are `w_*`, `w_e1_*`, `w_e2_*` — a base net plus two extras. **The two extra nets you
are budgeting for are already spent.**

**A W96 fp16 net is 15.3 MB and only 5.5 MB is free ⇒ you cannot ADD any net to `v3`.**

### What this means for Task B
The improved (dropout + 4× data) nets must **REPLACE** the current ensemble members, not
extend them. The envelope for the whole bounds stack is fixed at what `v3` already occupies:

* **Maximum total = 3 × W96**, or an equivalent mix within ~32.7 MB of net weights
  (W96 15.3 MB, W128 27.1 MB, W160 ≈ 42 MB — so W160 cannot appear alongside two W96s).
* Every configuration you report must state **extracted bytes**, not just net count, and must
  come in under **268,435,456 B**.

Report the table as: *3 improved nets* vs *2 improved + 1 current* vs *the current 3* — all
at equal size. That is a like-for-like comparison and it is the one that matters.

---

## ⚠️ CORRECTION 2 — The anchor check as described is circular

You wrote:

> "the script's `E_SHIP` constant (0.50199) translates cleanly to `34.3656` SPS (which
> perfectly matches the `SOUP_v1` anchor), and `E_FIX` translates cleanly to `33.9166`"

If `E_SHIP` was **defined** as `34.3656 / (100 · W)`, then recovering 34.3656 from it is
arithmetic, not validation — it confirms only that you divided and multiplied by the same
constant. It cannot detect an error in the instrument.

**The real test is the one you have running:** take each zip's **actual `predict()` output**
`lower`/`upper`, push it through the full evaluation pipeline, and compare the predicted sps
to the real leaderboard value. Please report that table, and label the two clearly:

| artifact | actual sps | predicted **from the zip's own predict() output** | error |
|---|---:|---|---|
| `SHIFT_v2_W96_a85` (banked) | 37.4803 | | |
| `SOUP_v1` | 34.3656 | | |
| `LUTFIX` | 33.9166 | | |
| `WIDE125` | 33.8038 | | |
| **`ASYM_W96_safe_arcsinh`** | **27.8581** | | |

**The arcsinh row is the whole point.** It is our only tight-bounds anchor. If the instrument
reproduces the four wide anchors but misses arcsinh, then it is unreliable exactly where we
most need it — and every "this tightening is safe" conclusion becomes suspect. Report that
outcome prominently if it happens; do not tune the instrument to fit.

### Minor
You used `W_OURS = 0.684593`. From the full-precision subscores (94.045224 / 75.998641 /
92.873514) it is **0.684538**. The difference is 0.008% ≈ 0.004 sps — negligible, so do not
re-run anything, but use 0.684538 going forward.

---

## ✓ Task C — accepted, and it completes the gate table

GPU ratio 0.8898, CPU ratio 0.9156, both interleaved. Combined with the gates I ran
(validator 13/13, bit-identity `max|Δpred| = 0.000e+00`, GATE 2 clean, checkpoint intact,
zero `.pyc`, width ratios exactly 1.0000), **`submission_ENSEMBLE_FAST_v3.zip` is fully
gated and is the submission candidate.** Nothing further is needed on it.

---

## Note on the OOM

Sequential queuing was the right call. For context: the stride-3 cache is 23.4 GB and the box
has 62 GB shared with other users. If you need more parallelism, consider `mmap_mode='r'` on
the cache rather than loading it fully — several scripts in this codebase already do that
(`np.load(..., mmap_mode="r")` on `tr_frames.npy`), and it would let 2–3 trainers coexist.
Optional; sequential is perfectly fine.

---

## Unchanged rules

* Do not submit. Do not mark anything CLOSED.
* Widths stay at the banked values — `v3`'s width ratio of exactly 1.0000 is *why* it is safe.
  A tighter `h_u` is not evidence of quality; only minimax worst-case is.
* State extracted bytes for every configuration.
* Verify an artifact exists and its per-file sizes match the source before reporting it done.
* Never write scratch to `/tmp` or the root partition.
* Never put a password in a shell command — `ssh vm` uses key auth.
