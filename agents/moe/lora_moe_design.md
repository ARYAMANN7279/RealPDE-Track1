# LoRA-MoE Design Document

## 1. Overview
The goal is to break the accuracy plateau by specializing the model for different flow regimes (AoA and Reynolds number) without exceeding the 256 MiB size limit.

## 2. Architecture
- **Shared Backbone**: The banked `sim_real_fno_fp16.pth` model.
- **Regime Experts**:
    - **AoA Experts**: 5 experts (0, 5, 10, 15, 20).
    - **Re Experts**: 2 experts (Low: < 15000, High: >= 15000).
- **Adaptation Mechanism**:
    - Use **LoRA (Low-Rank Adaptation)** for the `Linear` layers (`fc0`, `fc1`, `fc2`).
    - For `SpectralConv3d` layers, use **Additive Corrections** $\Delta W$. To keep them small, we will learn a low-rank decomposition or simply tune a small subset of the weights.
- **Router**:
    - A lightweight classifier based on the input window features (as implemented in `m03_router.py`).
    - At inference, the router predicts the regime, and the corresponding expert weights are added to the base weights.

## 3. Training Strategy
- **Partitioned Training**: Each expert is trained only on samples from its specific regime.
- **Loss Function**: `rel_l2 + wtke * tke` (the same objective that built the base model).
- **Regularization**: Use a small weight decay and a low learning rate to ensure the experts stay close to the base model.

## 4. Size Budget
- Base Model: ~200 MiB.
- LoRA Weights: Negligible (< 1 MiB per expert).
- Total: $\approx 200 + 7 \times 1 \approx 207$ MiB $\ll 256$ MiB.

## 5. Validation
- **SPS Local Monitor**: Every expert and the final MoE assembly must be gated using the `SPS` monitor.
- **GPU Parity**: Ensure that the additive corrections do not introduce numerical drift on the target GPU.
