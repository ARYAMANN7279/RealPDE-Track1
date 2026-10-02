# SUBMISSION LEDGER

## PART A.1: The Submission Ledger

| # | date | zip filename | predicted final | actual final | error (actual − predicted) | rel_l2 | tke | mvpe | time | sps | § where recorded |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Aug 23 | submission_SOUP_v1.zip | UNKNOWN | 78.456600 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | 34.365600 | §11A, §34.4 |
| 2 | Aug | submission_LUTFIX.zip | 79.24–79.38 | 78.338653 | −0.90 | UNKNOWN | UNKNOWN | UNKNOWN | 90.373861 | 33.916631 | §27, §28 |
| 3 | Aug | submission_WIDE125.zip | UNKNOWN | 78.300000 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | 35.019500 | §31 |
| 4 | Aug 31 | submission_ASYM_W96_safe_arcsinh.zip | UNKNOWN | 76.704691 | UNKNOWN | 94.045224 | 75.998641 | 92.873514 | UNKNOWN | 27.858126 | §38 |
| 5 | Sep 1 | submission_SHIFT_v2_W96_a85.zip | UNKNOWN | 79.246428 | UNKNOWN | 94.045224 | 75.998641 | 92.873514 | UNKNOWN | 37.480269 | §40 |
| 6 | Sep 1 | submission_ENSEMBLE_FAST_v3.zip | UNKNOWN | 79.341788 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | §43 |
| 7 | Sep 2 | submission_TMEAN.zip | ~79.41 | 79.369466 | −0.0405 | 94.072631 | 75.998641 | 93.141584 | 90.334470 | 37.803117 | §45 |
| 8 | Sep 3 | submission_LUTCAL.zip | 79.679466 | 79.279237 | −0.400229 | 94.074298 | 75.998641 | 93.124054 | 90.402099 | 37.424168 | §53 |
| 9 | Sep 4 | submission_FP16.zip | 79.395–79.403 | 79.378881 | −0.016…−0.024 | 94.074271 | 75.998930 | 93.124176 | 90.408319 | 37.815686 | §59, §60 |
| 10 | Sep 5 | submission_SV2.zip | UNKNOWN | 79.462591 | UNKNOWN | 94.050465 | 76.900655 | 93.130900 | 90.251551 | 37.887566 | §65 |
| 11 | Sep 9 | submission_SCREEN.zip | 79.489500 | 79.484440 | −0.005060 | 95.637000 | 76.425823 | 96.503700 | 90.569428 | 50.863200 | §102, §116.1 |
| 12 | Sep 9 | submission_BLIND.zip | UNKNOWN | 79.466919 | UNKNOWN | UNKNOWN | 76.501839 | UNKNOWN | UNKNOWN | UNKNOWN | §112 |
| 13 | Sep 10 | submission_W73.zip | 79.699500 | 79.441217 | −0.258283 | UNKNOWN | 76.437468 | UNKNOWN | UNKNOWN | UNKNOWN | §127 |

## PART A.2: Prediction Accuracy & Leakage

| zip | predicted | actual | error | was the candidate a BLEND/SOUP with a leaky partner? (YES/NO/UNKNOWN) | § |
|---|---|---|---|---|---|
| submission_SOUP_v1.zip | UNKNOWN | 78.4566 | UNKNOWN | YES (trained on 65 of 81 trajectories, every5 holdout) | §65.1 |
| submission_LUTFIX.zip | 79.24–79.38 | 78.338653 | −0.90 | UNKNOWN | §27 |
| submission_WIDE125.zip | UNKNOWN | 78.300 | UNKNOWN | UNKNOWN | §31 |
| submission_ASYM_W96_safe_arcsinh.zip | UNKNOWN | 76.704691 | UNKNOWN | UNKNOWN | §38 |
| submission_SHIFT_v2_W96_a85.zip | UNKNOWN | 79.246428 | UNKNOWN | UNKNOWN | §40 |
| submission_ENSEMBLE_FAST_v3.zip | UNKNOWN | 79.341788 | UNKNOWN | UNKNOWN | §43 |
| submission_TMEAN.zip | ~79.41 | 79.369466 | −0.0405 | UNKNOWN | §45 |
| submission_LUTCAL.zip | 79.679466 | 79.279237 | −0.400229 | UNKNOWN | §53 |
| submission_FP16.zip | 79.395–79.403 | 79.378881 | −0.016…−0.024 | YES (byte-identical to SV2_V4_GOLD which is soup_v1 80%) | §66.1 |
| submission_SV2.zip | UNKNOWN | 79.462591 | UNKNOWN | YES (soup_v2 100%) | §66 |
| submission_SCREEN.zip | 79.4895 | 79.484440 | −0.005060 | YES (0.5*soup_v2 + 0.5*sv3_3e5_10) | §102 |
| submission_BLIND.zip | UNKNOWN | 79.466919 | UNKNOWN | YES (100%-data EMA soup blended with soup_v2) | §112 |
| submission_W73.zip | 79.6995 | 79.441217 | −0.258283 | YES (both sides are leaky) | §127 |

### Summary
Number of submissions with a recorded predicted value: **6** (LUTFIX, TMEAN, LUTCAL, FP16, SCREEN, W73)
Largest prediction error found: **−0.90** (LUTFIX)
Smallest prediction error found: **−0.0051** (SCREEN)
