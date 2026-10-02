# Fitbase / Soup Analysis Report

## 1. Objective
To improve the base model accuracy by creating "model soups" (weighted averages of checkpoints) and fine-tuning them with various learning rates and durations.

## 2. Methodology
- **Members**: Generated a library of checkpoints with varying weight decays ($w=15, 33$) and learning rates ($lr=1, 3, 6, 10 \times 10^{-5}$).
- **Souping**: Evaluated various combinations of these members.
- **Gating**: Used a `cos0` gate (cosine similarity to the base model) to prevent "divergent" members from poisoning the soup, and a "magnitude law" to filter out local gains that are likely noise/leakage.

## 3. Key Findings
- **Recipe Failure**: The `RECIPE2` candidate, which showed promise on a small honest-holdout split, failed to transfer its gains when promoted to the full dataset.
- **Leakage**: Confirmed that "leakage" in small-sample measurements can inflate perceived gains. The "honest-holdout" gain was $\approx 0.14$ locally but negative live.
- **SPS Ceiling**: Most soup variations tie or marginally underperform the banked 78.45 score.

## 4. Conclusion
Simple weighting and souping of current checkpoints have hit a plateau. Further accuracy gains require more diverse initializations or a change in the training objective rather than hyperparameter tuning of the existing family.
