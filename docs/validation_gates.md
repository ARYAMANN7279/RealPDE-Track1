# TEST.md — the gate every submission must pass before I say "this is good to go"

Every gate below exists because something went wrong. The date in brackets is when
it bit us. **A zip that fails any BLOCKER gate is not submitted, no matter how good
the score estimate looks.**

Run order matters: cheap correctness gates first, expensive score gates last.

---

## GATE 0 — Rules compliance (BLOCKER, check FIRST)

Failing this is disqualification, not a low score. It cannot be detected later.

- [ ] **Every weight traces to the official release.** Allowed: the competition Google
      Drive `1Cg23DoTuSvWXR3Mm1uRfmMNAbkyaIhrQ` (`train_real`, `baseline_checkpoints`).
      **NOT allowed:** HuggingFace `AI4Science-WestlakeU/RealPDEBench-models`, or any
      model pretrained elsewhere. *[Aug 18: `submission_hf_fno.zip` used an HF
      checkpoint. It scored 64.76 so we dropped it — we escaped by luck, not care.]*
- [ ] **Every training sample traces to a released one** through a transform we can
      point at. Generating new trajectories is out, even if the generator was trained
      only on the release.
- [ ] **No cross-window information.** Each prediction depends only on its own input
      window. `metadata` is `{}` on scored calls. Using other evaluation windows is a
      **disqualifying condition**, not a penalty.
- [ ] **The method is reproducible from scratch** — see GATE 6.

## GATE 1 — Format and structural validity (BLOCKER)

- [ ] Official validator passes **13/13**, run as
      `python _val.py <zip> --height 32 --width 64`.
      *(Its default is 64x128, the raw PIV resolution, which no fixed-shape FNO
      accepts. Overriding is correct, not a workaround.)*
- [ ] Extracted size **< 256 MB** (`unzip -l` total, not the zipped size).
- [ ] Entry list diffed against the last archive that actually scored. Anything added
      or removed is deliberate and justified.
- [ ] No `__pycache__` / `*.pyc` in the zip.
- [ ] `unzip -t` reports no errors; **md5 identical on Mac and VM**.
- [ ] If the checkpoint was repacked to fp16: it is **byte-identical** to a known-good
      one, or the repack was verified. *[The FNO has 16 COMPLEX spectral tensors:
      `.float()` silently drops the imaginary part (a soup scored -7.34 this way) and
      `v.half() if v.is_floating_point()` skips them entirely, leaving 403 MB. Always
      branch on `v.is_complex() or torch.is_floating_point(v)`; prefer the kit's
      `pack_ckpt_fp16.py`.]*
- [ ] Checkpoint filename inside the zip is `sim_real_fno_fp16.pth` —
      `load_baseline` infers the architecture from the FILENAME.

## GATE 2 — End-to-end behaviour of the ACTUAL zip (BLOCKER)

Not a simulation of the method. Extract the zip and call its own `predict()`.

- [ ] Output shapes `(N, 20, 32, 64, 3)` for prediction, lower and upper.
- [ ] All three arrays **finite**; `lower <= upper` everywhere.
- [ ] `prediction[..., 2] == 0` (p is unmeasured in real data).
- [ ] Bounds are all-or-nothing: **every** call returns bounds, or none does. Mixing
      makes the scorer discard all of them and fall back to the default band.
- [ ] If bounds are meant to vary per element, confirm they actually do
      (`std > 0`) — a silent fallback to a constant band is easy to miss.

## GATE 3 — Time budget (BLOCKER)

*[Aug 20: three consecutive submissions Failed with 0-byte results. Cause was a
timeout, not the archive. A two-number diff from a working zip failed, which should
have told us content was irrelevant much sooner.]*

- [ ] **Know the limit: 300 s for the PREDICTION stage** ("5 minutes in the Warm-up
      and Development phases, container execution only; data download time is not
      counted"). *[CORRECTED Aug 25, see memory 17.2 — this gate said 180 s, and the
      whole Aug-25 timeout investigation was sized against a limit 2x tighter than
      reality. `docs/competition_spec.md` is a stale pre-restart copy.]* A warm-up call
      absorbs one-time costs (imports, CUDA context) and is **not timed**.
- [ ] **Budget is measured per CALL**, not from module import — otherwise the untimed
      warm-up call eats the scored call's budget.
- [ ] Measure the **ratio to the bare FNO**, not absolute ms. The shared VM is noisy;
      the same zip measured 2.69 and 4.57 ms/sample minutes apart. The ratio is stable.
- [ ] Convert with the validated anchor: **our FNO = 6.59 ms/sample on the A800**
      (predicted 6.57 — the factor is trustworthy). `est = 6.59 * ratio`.
- [ ] **Account for the `time_score` penalty**, which no accuracy harness reports:
      `score = 100/(1+sqrt(t/0.72896))`. Doubling inference cost ~0.29 final.
- [ ] Keep heavy work **on GPU**. CPU is shared by up to 8 concurrent evaluations on
      the host. *[A `np.fft` over the whole array was costing 0.6x the FNO's runtime;
      `torch.fft` cut the overhead 1.94x -> 1.32x.]*
- [ ] Time-budget fallback verified at **full trip** (constant bounds everywhere) and
      **partial trip** (mixed), both still finite and correctly ordered.

## GATE 4 — Score assurance (BLOCKER)

- [ ] **Self-check passes AND USES A BOUNDS-MATCHED ANCHOR.** The harness must
      reproduce a submission whose real score we know **in the same bounds regime as
      the candidate**. The bias is NOT constant across regimes:
      | anchor | bounds | predicted | actual | bias |
      |---|---|---:|---:|---:|
      | `submission_fno_plain_sps.zip` | wide [0.030, 0.010] | 77.35 | 77.20 | **+0.15** |
      | `submission_ROBUST.zip` | tight [0.0129, 0.0098] | 78.64 | 78.07 | **+0.57** |
      *[Aug 20: quoted 79.14 for a tight-bounds candidate using the WIDE anchor's
      +0.15 bias. The correct figure was ~78.7. Same class of error as projecting
      44.72 sps when the all-time max was 42.84 — a correction applied in the wrong
      regime. Use the tight anchor for tight bounds; if the candidate is tighter than
      any anchor we have, say so and widen the stated range.]*
      If the matched self-check is off by more than 0.5, the estimate is void.
      *(This assertion has caught three real bugs: an inverted bisection, an inverted
      error-scale, and a split-inconsistent reference that shifted a whole table by -0.5.)*
- [ ] Split is **disjoint BY TRAJECTORY**, never a random window split — neighbouring
      windows share frames and would leak.
- [ ] Estimate computed on the **actual zip's own outputs**, predictions and bounds.
- [ ] Apply BOTH corrections to the raw estimate, using the **bounds-matched** bias:
      `final = raw - selfcheck_bias(matched regime) - time_penalty`
- [ ] **Predicted final > current banked best.** Currently **78.07**.
- [ ] Sanity vs reality: does the estimate exceed something a real competitor has
      actually achieved on this model? If yes, explain why or distrust it.
      *[A projection of 44.72 sps was once produced when the all-time max was 42.84.]*
- [ ] State a **range**, not a point. Run both split directions if time allows.

## GATE 4B — Error-ranking integrity (BLOCKER)

*[Aug 20-21: `submission_FULLSTACK_v6` scored **76.61** vs an estimate of 78.72 —
below the banked 78.07. Cause: the bound lookup table was fitted on errors that had
been multiplied per-element by `(alpha + beta*|pred|)`.]*

A per-element uncertainty policy is worth exactly as much as its **ranking** of which
elements are hard. Any transform applied to errors before that ranking is derived can
destroy it.

- [ ] **Never apply a per-element transform to errors before fitting a bound map.**
      `(alpha + beta*|pred|)` was fitted to match AGGREGATE E at two bound settings,
      then applied per-element; its multiplier varies ~1.6x to ~2.75x across the field,
      so it REORDERS elements rather than rescaling them.
      *Measured on held-out data: clean LUT E 0.6198, transformed LUT E 0.5569 — the
      transform costs 0.063 and drops the head BELOW plain constants (0.5659).*
- [ ] Calibrate a **global scale**, or calibrate the resulting **half-widths** — never
      the per-element errors the ranking comes from.
- [ ] Sanity-check any per-element policy against the best constant **on the same
      held-out data**. If it does not beat constants there, it will not beat them live.
- [ ] ⚠️ **`fs_assure.py` CANNOT score a per-element policy.** It scores bounds against
      `|P-Y|*(alpha+beta*|pred|)`, and that transform is only valid in aggregate — so
      scoring a per-element policy against per-element-distorted errors is as invalid
      as fitting on them. It stays valid for CONSTANT bounds (a constant h is immune to
      reordering), which is why the 77.20 and 78.07 self-checks both passed.
      For a per-element policy, estimate instead from a **measured out-of-sample
      increment over constants** (the M55 protocol) applied to our known real constant
      performance.
- [ ] In-sample fitting is NOT by itself disqualifying: measured on M55, the in-sample
      over-estimate is +0.029 for BOTH constants and the head, so it does not favour
      one policy over the other. Correct for it globally; do not use it as a reason to
      avoid learned policies.

## GATE 5 — Calibration hygiene

- [ ] Error model anchored to REAL leaderboard anchors, not local scores. Known
      anchors for the original checkpoint (W = 0.6784):
      | bounds | real sps | real E |
      |---|---:|---:|
      | `0.05*abs(pred)` | 14.08 | 0.2076 |
      | `[0.030, 0.010]` | 29.84 | 0.4399 |
      | `[0.0129, 0.0098]` | 33.08 | 0.4876 |
- [ ] **Do not extrapolate the error model to tighter bounds.** It fits both wide
      anchors and over-predicts the tight one by +0.029. Constant bounds are exhausted.
- [ ] A single error-scale factor is **not** sufficient — use `(alpha + beta*|pred|)`,
      fitted to at least two anchors. One factor leaves ~15x the residual.

## GATE 6 — Reproducibility (BLOCKER before the Decision Phase)

*"Organizers re-train each shortlisted method from scratch... a method that cannot be
reproduced is disqualified and replaced by the next-ranked team."*

- [ ] `train/01..06` runs end-to-end on the released data alone.
- [ ] The rebuilt artifact scores **within noise** of the hand-built one.
      *[Verified Aug 20: E 0.5559 vs 0.5551.]*
- [ ] Seeds fixed and printed; the excluded duplicate `7575_0.h5` is handled.
- [ ] Any augmentation/auxiliary model ships as CODE, not just weights.

## GATE 7 — After submitting (do not skip)

- [ ] Record the real subscores in `project_memory.md`.
- [ ] Compute **predicted minus actual** and update the harness bias. This is the
      single most valuable number we get per day.
- [ ] If it **Failed**: check whether the result zip is **0 bytes**. That means the
      container was killed (timeout/OOM) and the archive contents are almost certainly
      innocent — go to GATE 3, do not start editing the zip.
- [ ] ⛔ **A failed submission DOES consume the daily slot.** *[CORRECTED Aug 25, see
      memory 17.1/18 — this gate said the opposite, and that belief produced four
      submissions in one UTC day, of which only the first ran. Three rebuilds and a full
      forensic pass were spent root-causing two "failures" that never executed.]*
      **One submission per UTC day, reset 00:00 UTC = 05:30 IST.** Before concluding
      anything from a "Failed", check the submission count for that day FIRST; if it is
      not the first, the result carries no information.
      `Force_Best` means the banked score can never be lost.

---

## Quick command reference

```bash
# GATE 1
python _val.py <zip> --height 32 --width 64      # expect 13/13
unzip -l <zip> | tail -1                          # extracted bytes vs 256 MB
diff <(unzip -l proven.zip | awk 'NR>3{print $4}') <(unzip -l new.zip | awk 'NR>3{print $4}')

# GATE 2 + 3
python local_harness/fstest.py                    # shapes, finiteness, timing, fallback

# GATE 4
python local_harness/fs_assure.py                 # self-check + estimate on the real zip

# GATE 6
cd train && ./run_all.sh
```
