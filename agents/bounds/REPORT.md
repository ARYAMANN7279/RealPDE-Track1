# Bounds Analysis Report

## 1. Objective
To find a bound-width policy that maximizes the SPS score by optimizing the trade-off between coverage and the penalty for wide intervals.

## 2. Key Findings
- **Optimal Width**: Through analytic and numeric sweeps, we identified that the optimal half-width $h^*$ is approximately $0.1 \times \sigma$ to $0.2 \times \sigma$ for standard error distributions (Gauss/Laplace), resulting in coverage around 70-90%.
- **Current Best**: `ENSEMBLE_FAST_v3` is the most effective current policy, achieving an $E_{live} \approx 0.551$.
- **Gap to Leader**: The top team on the leaderboard has $E \approx 0.616$, indicating a significant remaining head-room ($\approx 0.065$ in $E$ units).
- **Local vs Live**: The local harness over-states the sensitivity to bound width by $\approx 1.7\times$. This explains why local optimums sometimes diverge from live results.

## 3. Verified Constraints
- **Symmetry**: Asymmetric intervals (shifting the centre) provided negligible gains over symmetric intervals centered on the prediction.
- **Scaling**: Mis-scaling the bounds (e.g., by $0.5\times$ or $1.5\times$) results in a relatively graceful degradation of the reward $R$, but the peak is sharp enough to justify precise calibration.

## 4. Conclusion
The current learned per-element head is near its limit. Further gains likely require a fundamentally different uncertainty signal or joint training of prediction and uncertainty (heteroscedastic NLL) to avoid transfer loss between training and live distributions.
