# RealPDEBench — Codebase Analysis

**Repo:** https://github.com/AI4Science-WestlakeU/RealPDEBench
**Paper:** RealPDEBench (ICLR 2026 Oral) — arXiv:2601.01829
**Authors:** Peiyan Hu, Haodong Feng, Hongyuan Liu, Tongtong Yan, Wenhao Deng, Tianrun Gao, Rong Zheng, Haoren Zheng, Chenglei Yu, Chuanrui Wang, Kaiwen Li, Zhi-Ming Ma, Dezhi Zhou, Xingcai Lu, Dixia Fan, Tailin Wu
**Local clone:** `/Users/aryamannsrivastava/Desktop/IMPORTANT/UGP/code/RealPDEBench/`
**License:** CC BY-NC 4.0 (non-commercial)
**Python:** ≥3.10

---

## 1. High-Level Architecture

```
realpdebench/
├── __init__.py
├── __main__.py
├── cli.py                 # CLI entry: realpdebench <command>
├── train.py               # main training script
├── train_surrogate.py     # surrogate model training
├── eval.py                # evaluation
├── hf_download.py         # HF dataset downloader
├── data/
│   ├── dataset.py         # abstract RealDataset base class
│   ├── fluid_dataset.py   # FSI, Cylinder, ControlledCylinder, Foil
│   ├── fluid_hf_dataset.py  # HF variants
│   ├── combustion_dataset.py
│   ├── combustion_hf_dataset.py
│   ├── combustion_surrogate_dataset.py
│   ├── combustion_surrogate_hf_dataset.py
│   ├── data_normalizer.py
│   ├── generate_surrogate_data.py
│   ├── numerical_real_compare.py
│   └── convert_hdf5_to_hf.py
├── model/
│   ├── model.py           # base model class
│   ├── load_model.py      # dispatch
│   ├── unet.py
│   ├── fno.py             # Fourier Neural Operator
│   ├── cno.py             # Convolutional Neural Operator
│   ├── wdno.py            # Wavelet-Diffusion Neural Operator
│   ├── deeponet.py
│   ├── galerkin_transformer.py
│   ├── dpot.py            # DPOT small + large
│   ├── mwt.py             # Multi-Wavelet Transformer (may be external)
│   ├── dmd.py             # Dynamic Mode Decomposition
│   └── __init__.py
├── configs/
│   ├── cylinder/<model>.yaml
│   ├── controlled_cylinder/<model>.yaml
│   ├── fsi/<model>.yaml
│   ├── foil/<model>.yaml
│   ├── combustion/<model>.yaml
│   └── combustion_surrogate/<model>.yaml
├── utils/
│   ├── utils.py
│   ├── metrics.py          # eval_metrics, mse_loss
│   ├── convert_hdf5_to_hf.py
│   └── dpot_ckpts_dl.py
└── data/sim_generation/
    ├── foil/
    │   ├── README.md
    │   └── ThreeD_NACA.jl  # NACA0025 tapered hydrofoil
    ├── cylinder/fixed_cylinder/
    ├── controlled_cylinder/
    ├── fsi/fluid_structure_interaction/
    └── combustion/
```

**Datasets:** 5 — `cylinder`, `controlled_cylinder`, `fsi`, `foil`, `combustion`
**Trajectories:** 700+
**Baseline models:** 10 — U-Net, FNO, CNO, WDNO, DeepONet, MWT, GK-Transformer, Transolver, DPOT, DMD
**Evaluation metrics:** 9 — RMSE, MAE, Rel L₂, R², Update Ratio, fRMSE, FE, KE, MVPE

---

## 2. The Foil Dataset (the relevant one)

| Property | Value |
|---|---|
| Airfoil | **NACA0025 tapered hydrofoil** (per `ThreeD_NACA.jl`: `tapered_naca0025`) |
| Trajectories | 99 numerical + 98 real |
| Frames per trajectory | 3,990 |
| Resolution (real) | 128×256 |
| Resolution (sim) | 128×256 |
| Modalities (real) | u, v |
| Modalities (sim) | u, v, p |
| Angles of attack | 0°, 5°, 10°, 15°, 20° |
| Reynolds numbers | 2,968–17,031 (19 discrete values) |
| Total memory | 335.64 GB |

**Splits (per real dataset):** remain = 78, in_dist_test = 10, out_dist_test = 10
**Splits (numerical):** remain = 99, in_dist_test = 0, out_dist_test = 0

The `out_dist_test` matches the competition's "novel AoA/Re" claim — those are the out-of-distribution samples.

### ⚠️ NACA0025 vs NACA4418

The actual water-tunnel data and simulation code use **NACA0025** (symmetric 25%-thickness hydrofoil, **tapered**).
The competition spec says **NACA4418** (cambered 4%-camber at 40%-chord, 18%-thickness).

**These are fundamentally different airfoils** — 0025 is symmetric, 4418 is cambered. The aerodynamic behavior differs significantly. This needs resolution from the organizers.

---

## 3. Data Loading (`realpdebench/data/`)

### 3.1 Base `RealDataset` class (`dataset.py`)

Constructor signature (extracted):
```python
RealDataset(
    dataset_name, dataset_root, dataset_type, mode, test_mode,
    mask_prob, in_step, out_step, N_autoregressive, interval,
    train_ratio, split_numerical, trunk_length, noise_scale,
    n_sim_in_distribution, n_sim_out_distribution, n_sim_frame,
    sub_s_real=1, sub_s_numerical=1,
    noise_type='gaussian', optical_kernel_size=4, optical_sigma=1.0
)
```

Key parameters:
- `in_step`: input time steps (e.g., 20)
- `out_step`: output time steps (e.g., 20) — multiplied by `N_autoregressive`
- `N_autoregressive`: number of autoregressive rollout steps
- `interval`: sliding window stride
- `mask_prob`: probability of masking unmeasured modalities (numerical only)
- `noise_scale`, `noise_type`: noise to add to numerical data (gaussian | poisson | optical)
- `sub_s_real`, `sub_s_numerical`: spatial sub-sampling factors
- `mode`: train | val | test
- `test_mode`: all | in_dist | out_dist | seen | unseen

**Test modes:** `in_dist` = same AoA/Re as training, `out_dist` = unseen AoA/Re (the generalisation test).

### 3.2 Fluid datasets (`fluid_dataset.py`)

Classes: `FSI`, `Cylinder`, `ControlledCylinder`, `Foil`. Each inherits from `RealDataset` and overrides `__getitem__` and metadata.

### 3.3 HF variant (`fluid_hf_dataset.py`)

Classes: `CylinderHFDataset`, `FSIHFDataset`, `ControlledCylinderHFDataset`, `FoilHFDataset`. Loads from `hf_dataset/` via `load_from_disk` (HuggingFace Hub).

### 3.4 Normalizer (`data_normalizer.py`)

Three options: `IdentityNormalizer`, `GaussianNormalizer`, `RangeNormalizer`.

Config sets `normalizer: "gaussian"` by default (`configs/foil/fno.yaml`).

### 3.5 Auto-download

`hf_download.py` — pattern-based downloader:
```bash
realpdebench download --dataset-root /path/to/data --scenario cylinder --what metadata
realpdebench download --dataset-root /path/to/data --scenario cylinder --what hf_dataset --dataset-type real
```

Set `--endpoint https://hf-mirror.com` or `HF_ENDPOINT` env to use HF mirror.
`HF_HUB_DISABLE_XET=1` recommended.

---

## 4. Models

### 4.1 Base + dispatch (`model.py`, `load_model.py`)

`load_model(model_name, ...)` dispatches to the right model class. Configs specify `model_name: fno`, etc.

### 4.2 The 10 baselines

| Model | File | Type | Notes |
|---|---|---|---|
| U-Net | `unet.py` | Convolutional encoder-decoder | Spatial-only |
| FNO | `fno.py` | Fourier Neural Operator | Spectral filtering |
| CNO | `cno.py` | Convolutional Neural Operator | CNO paper |
| WDNO | `wdno.py` | Wavelet-Diffusion Neural Operator | Wavelet-based |
| DeepONet | `deeponet.py` | Operator learning | Branch + trunk |
| MWT | `mwt.py` | Multi-Wavelet Transformer | External? |
| GK-Transformer | `galerkin_transformer.py` | Galerkin attention | Transformer-based |
| Transolver | (external) | Physics-Attention | Vendor-provided in Track 1 starting kit |
| DPOT | `dpot.py` | DPOT small + large | Foundation model |
| DMD | `dmd.py` | Dynamic Mode Decomposition | Classical baseline |

### 4.3 Default FNO config (`configs/foil/fno.yaml`)

```yaml
exp_name: "fno_foil"
gpu: 0
seed: 0
results_path: "./results/"

# data
dataset_name: "foil"
dataset_root: "/wutailin/real_benchmark/"
num_workers: 12
normalizer: "gaussian"

# data parameters
mask_prob: 0.1
noise_scale: 0.1

# model
model_name: "fno"
checkpoint_path: ./results/fno/fno_foil_numerical_False/2025-09-17_14-33-39/model_3760.pth
modes1: 4
modes2: 12
modes3: 16
n_layers: 4
width: 64

# training
is_use_tb: True
scheduler: cosine
step_size: 1000
num_update: 4000
train_batch_size: 32
test_batch_size: 64
lr: 0.0001
clip_grad_norm: 0.

# evaluation
N_autoregressive: 1
N_plot: 0
```

**Key params for FNO:**
- `modes1=4, modes2=12, modes3=16` — Fourier modes for time, H, W
- `n_layers=4` — FNO layers
- `width=64` — channel width
- `num_update=4000` — total gradient updates (not epochs)
- `train_batch_size=32, lr=1e-4, cosine schedule`
- `N_autoregressive=1` — direct 20-step prediction, no rollout

### 4.4 Training paradigms

Three training modes via CLI flags:
- `numerical` — train on simulated data only
- `real` — train on real data only
- `is_finetune=True` — pretrain on sim, finetune on real

CLI: `python -m realpdebench.train --config configs/foil/fno.yaml --train_data_type numerical [--is_finetune]`

---

## 5. Training (`train.py`)

Key flow:
1. Parse args (config + train_data_type + flags)
2. Load dataset (HF or local)
3. Setup model + optimizer
4. Loop:
   - Sample batch
   - Compute loss (MSE on (u, v, p))
   - Backprop + step
   - Validate periodically
5. Save checkpoint to `results/{model_name}/{exp_name}_{train_data_type}_{is_finetune}/{timestamp}/model_{step}.pth`

**Loss:** MSE loss from `utils/metrics.py:` `mse_loss`.

**Schedule:** `cosine` (default) or `step`. `step_size=1000` for step.

**Logging:** TensorBoard (`is_use_tb: True`).

**Total budget:** `num_update=4000` typical. With `train_batch_size=32`, that's ~128k samples seen.

---

## 6. Evaluation (`eval.py`, `utils/metrics.py`)

`utils/metrics.py` provides `eval_metrics(pred, target)` and `mse_loss(pred, target)`.

**9 metrics:**
- RMSE — root mean squared error
- MAE — mean absolute error
- Rel L₂ — relative L2 norm
- R² — coefficient of determination
- Update Ratio — likely a temporal stability metric
- fRMSE — filtered RMSE (high-frequency content)
- FE — flow error (L2 of velocity gradient)
- KE — kinetic energy error
- MVPE — mean velocity profile error

The local eval (`eval.py`) computes these. The competition scoring program (`scoring.py` in starting kit) computes a different set containing Rel-L2, TKE, MVPE, Time, SPS (only 5 of the 9).

---

## 7. Data Pipeline Deep-Dive

### 7.1 How training pairs are formed

For each simulation/real trajectory:
- `n_sim_frame` total frames (e.g., 3990)
- `horizon = in_step + out_step * N_autoregressive` (e.g., 20 + 20*1 = 40)
- `n_data_per_sim = (n_sim_frame - horizon + 1) // interval`

Each item is a sliding window: in_step input frames → out_step target frames.

### 7.2 Spatial subsampling

`sub_s_real` and `sub_s_numerical` — sub-sample spatial resolution. **The competition uses 32×64 from raw 128×256**, so `sub_s=4` from the raw data.

### 7.3 Noise injection (numerical only)

To simulate sim-to-real gap, numerical data can be perturbed:
- `mask_prob=0.1` — randomly mask 10% of (u,v,p) channels to simulate unmeasured modalities
- `noise_scale=0.1` — add Gaussian noise at 10% scale
- `noise_type=gaussian | poisson | optical` — different noise distributions

This is the **bridge** between pure sim training and real evaluation — the codebase already supports the noise injection trick.

### 7.4 Output shape

Per the README and config:
- Input: `(B, T_in, C, H, W)` or `(B, T_in, H, W, C)` — depends on the model
- Output: same shape as input, T_in → T_out

The competition expects `(N, T_out, H, W, C)` (channels last). Internal models may use channels-first; need to transpose in `predict()`.

---

## 8. Reproduction Recipe

**Minimal Track 1 submission using RealPDEBench:**

```bash
# 1. Clone + install
git clone https://github.com/AI4Science-WestlakeU/RealPDEBench.git
cd RealPDEBench
pip install -e .

# 2. Download dataset (foil only)
realpdebench download --dataset-root ./data --scenario foil --what hf_dataset --dataset-type real
realpdebench download --dataset-root ./data --scenario foil --what hf_dataset --dataset-type numerical

# 3. Train FNO on numerical data (pretrain)
python -m realpdebench.train --config configs/foil/fno.yaml --train_data_type numerical

# 4. Finetune on real data
python -m realpdebench.train --config configs/foil/fno.yaml --train_data_type real --is_finetune

# 5. Eval
python -m realpdebench.eval --config configs/foil/fno.yaml --checkpoint <path>
```

**Expected GPU:** Likely 1× A100 (40GB) for FNO at width=64. Transolver/CNO may need more.

**Pretrained checkpoints:** Available on HuggingFace at `AI4Science-WestlakeU/RealPDEBench-models` for all 10 models × 5 scenarios × 3 paradigms.

---

## 9. Key Findings from the Paper

1. **Sim-to-real gap exists:** Models trained on simulated data show clear performance degradation when evaluated on real (9.39% to 78.91% improvement in Rel L₂ when training on real data).
2. **Pretrain on sim, finetune on real works best** — outperforms training on real alone with the same budget.
3. **Real data is noisier, simulated data has numerical errors** — they fail in different ways.
4. **DPOT-L** (large pretrained foundation model) achieved best overall performance.

---

## 10. Open Questions for Competition

1. **NACA4418 vs NACA0025** — discrepancy needs clarification.
2. **Same data?** Does the competition use the same Foil dataset as RealPDEBench, or a different one?
3. **Resolution:** Competition uses 32×64; RealPDEBench has 128×256. Likely `sub_s_real=4` from raw.
4. **Hidden validation set:** What's the AoA/Re split? Unknown until you submit.
5. **Starting kit vs RealPDEBench:** Are the starting-kit baselines identical to the repo, or modified?

---

## 11. What RealPDEBench Gives You

- ✅ **Data loaders** for HF + local HDF5
- ✅ **10 baseline models** in PyTorch
- ✅ **3 training paradigms** (numerical / real / finetune)
- ✅ **Checkpoint download** from HF
- ✅ **Noise injection** to bridge sim-to-real
- ✅ **9 evaluation metrics** locally
- ✅ **TensorBoard logging**

## 12. What You'd Need to Build for Track 1

- ❌ **Submission wrapper** (`submission.py` with `predict()`)
- ❌ **Channel handling** — models output (T, C, H, W); competition expects (T, H, W, C)
- ❌ **Optional bounds** (`lower`, `upper`) for SPS credit
- ❌ **Size optimization** if chosen model > 256 MB
- ❌ **Inference speed optimization** for Time score
- ❌ **Possibly: custom architecture** beyond the 10 baselines
- ❌ **Possibly: custom loss** combining MSE + PDE residual + adversarial
