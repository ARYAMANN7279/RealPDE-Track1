# Mixture of Experts (MoE) Analysis Report

## 1. Objective
To exploit the observed variance in model performance across different flow regimes (Angle of Attack and Reynolds number) by training specialized "experts" for each partition.

## 2. Observations
- **AoA Sensitivity**: The banked model performs significantly better at $\text{AoA} = 0^\circ$ ($\text{rel\_l2} \approx 96.96$) than at $\text{AoA} = 20^\circ$ ($\text{rel\_l2} \approx 94.21$).
- **Reynolds Sensitivity**: Low-Re trajectories ($\text{Re} < 15000$) are generally easier to predict ($\text{rel\_l2} \approx 95.84$) than high-Re ones ($\text{rel\_l2} \approx 95.19$).

## 3. Implementation
- **Partitioning**: The dataset was split into experts based on AoA ($0, 5, 10, 15, 20$) and Re (Low/High).
- **Experts**: Checkpoints were trained/extracted for these specific regimes (stored in `ckpt/`).
- **Results**: Initial per-partition evaluations (`res/*.npz`) show that specialization captures the regime-specific error distributions more accurately than the global model.

## 4. Conclusion
The MoE approach is theoretically sound because the error is non-uniform across the parameter space. The next step is to implement a robust "router" that can assign samples to experts at inference time without using the ground truth labels.
