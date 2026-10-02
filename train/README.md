# Reproducible training pipeline — RealPDE Track 1 (team: Rajesh Ranjan / Aryamann Srivastava / Prashant Kumar)

Required by the Decision Phase rule: *"organizers re-train each shortlisted method
from scratch on the competition data before the ranking is decided... a method that
cannot be reproduced is disqualified"* (forum, wenhao2026, Aug 5 2026).

## Provenance — every input traces to the official release

| input | source | allowed? |
|---|---|---|
| `train_real/*.h5` (82 trajectories) | competition Google Drive `1Cg23DoTuSvWXR3Mm1uRfmMNAbkyaIhrQ` | yes — released training data |
| `sim_real_fno.pth` / `sim_real_fno_fp16.pth` | same Drive, `baseline_checkpoints/sim_real_ft/` | yes — official baseline checkpoint |

**No outside data and no outside weights are used anywhere in the chain.**
In particular we do NOT use the HuggingFace `AI4Science-WestlakeU/RealPDEBench-models`
checkpoints; an early experiment with those was abandoned and is not part of this
solution.

## What the method actually is

The submitted model is the **unmodified official FNO baseline**. On top of it:

1. **Temporal spectral correction.** The FNO under-penalises high-wavenumber error
   under an Lp loss and loses energy (measured: predicted/target temporal KE = 0.637).
   We fit a per-frequency gain `g(f)` on the released training data and apply it to
   the prediction along the time axis. Amplitude only; phase untouched.
2. **Learned per-element SPS bounds.** A small CNN predicts `log|error|` per element
   from inference-available features. A non-parametric lookup maps predicted error
   to the half-width that maximises `exp(-2h/sigma)*P(|err|<=h)`.

Neither step alters the model weights, so `rel_l2` behaviour is the baseline's plus
the spectral correction.

## Run order

```bash
python 01_prep_data.py      # train_real/*.h5 -> windows at 32x64 (subsample by 2)
python 02_cache_preds.py    # official FNO -> cached predictions on those windows
python 03_fit_spectral.py   # per-frequency temporal gain
python 04_train_head.py     # uncertainty CNN (seeded)
python 05_build_lut.py      # predicted-error -> optimal half-width table
python 06_build_submission.py
```

Deterministic: every script seeds numpy and torch and prints the seed it used.
Total runtime on one A800: roughly 40 minutes, dominated by 04.

## Validation protocol used during development

All reported numbers use a **trajectory-disjoint** split of the 82 released
trajectories — never a random window split, which would leak neighbouring frames.
The harness carries a hard assertion that the unmodified baseline plus its known
bounds reproduces its real leaderboard score (77.20); if that fails the run aborts,
because the calibration is then untrustworthy.

## Known data issue
`train_real/7575_0.h5` duplicates `6300_0.h5` (reported on the forum by benslash2,
Aug 6). `01_prep_data.py` excludes `7575_0.h5`.
