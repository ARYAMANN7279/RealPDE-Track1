# What actually shipped

Extracted from `submissions/submission_STACK.zip`, the archive that scored **80.081266** on the
development set and **75.087416** in the final decision phase.

- `submission.py` — scoring entry point. Differs from the working copy at the repository root;
  this one is the graded artifact, that one is the scratch version. Trust this one.
- `load_baseline.py` — backbone loader.

Not here, by design: `sim_real_fno_fp16.pth` (201.4 MB), `bounds_assets.npz` (42.5 MB),
`tkehead_assets.npz` (6.5 MB), the organisers' `rpde_baselines/` package, and vendored `einops`.
All of it lives in `submissions/`. Rebuild an archive by zipping with `-0` (stored, not deflated —
worth 6.2x on asset load).
