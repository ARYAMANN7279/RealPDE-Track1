# underPINN — Codebase Analysis

**Repo:** https://github.com/Aeroscience-Computations-Analysis-Lab/underPINN
**Version:** v2605 (CalVer YYMM, May 2026)
**Stack:** JAX + Flax + Optax, Python ≥3.9
**License:** GPL-3.0
**Stars:** 29 · Forks: 7 · Last push: Jul 29, 2026
**Authors (per CITATION.cff):** Prashant Kumar, Lohith Senthilkumar, Rajesh Ranjan
**Local clone:** `~/Desktop/sem7/UGP/underPINN/` (⚠️ currently only `.git` + `.venv` — source files missing; run `git pull` inside to restore them)

---

## 1. High-Level Architecture

```
CLI (underPINN/__main__.py)
  → YAML loader (config/loader.py)
  → dispatch table (runner/dispatch.py) → example script
  → construct: model (nn/factory.py) + PDE (pde/*.py) + geometry (geometry/*.py) + loss (losses/*.py)
  → solver (solver/*.py) [SteadySolver / FBPINNSolver / ODESolver / LDCSolver / RANSSolver]
  → callbacks (callbacks/) — checkpoint, log, early stop
  → JAX JIT step → optax update → append loss_history
  → restart/resume via RestartManager (utils/restart.py)
  → checkpoints + plots + npz predictions
```

- ~6,000 LOC in `underPINN/` package
- ~2,000 LOC in `examples/`
- ~2,300 LOC in `tests/`
- No top-level trainer — training loops live in each solver
- Each example is a self-contained runner (script + `./config.yaml`)

---

## 2. Core Abstractions

### 2.1 Base classes (`underPINN/core/base.py`)

- **`BasePDE`** — abstract physics operator. `residual(params, xy)` takes packed `(N, D)` coords, returns `(N,)` or `(N, K)`. Optional `u`, `exact` for eval.
- **`BaseLoss`** — `__call__(params, *args, **kwargs) → (total, components_tuple)`
- **`BaseSolver`** — owns `init()`, `train()`, optional `evaluate()`. Provides `save_checkpoint()`, `restore_checkpoint()`, `_make_opt(lr, lr_schedule)`, callback wiring.

### 2.2 NN module (`underPINN/nn/`)

- `factory.py:22-56` — `build_model(net_cfg)` dispatches by `type`
- Types: `mlp`, `gated_mlp`, `fourier_mlp`, `temporal_fourier`, `siren`, `fbpinn`
- `mlp.py` — `MLP` (line 6), `GatedMLP` (Wang et al. 2022, line 17), `SIREN` (line 80), `TemporalFourierMLP` (line 125), `FourierMLP` (line 192)
- `fbpinn.py:18-44` — FBPINN: overlapping subdomain decomp with sigmoid partition-of-unity windows
- `subdomain.py:5-29` — `SubdomainNetwork` (per-subdomain net with HybridAttention gates)
- `attention.py` — AdditiveAttention, DotProductAttention, SimpleGate, HybridAttention

### 2.3 Loss module (`underPINN/losses/`)

- `loss.py:6-12` — `l2_loss`, `l1_loss` scalar norms
- `loss.py:14-21` — `weight_l2(params)` L2 regularization
- `loss.py:23-101` — `PINNLoss` (pde + ic + bc + reg; RBA weighting at lines 59-68)
- `ode_loss.py:5-58` — `ODELoss` (pde + ic + optional ic-derivative + reg)
- `steady_loss.py:7-72` — `SteadyLoss` (pde + bc + reg, no IC)

### 2.4 Training (`underPINN/training/resample.py`)

- `rar_d_resample` (lines 57-142) — packed RAR-D (Lu et al. 2021, p ∝ |r|^k)
- `rar_d_resample_split` (lines 145-201) — backward-compat shim

### 2.5 Callbacks (`underPINN/callbacks/`)

- `base.py:4-28` — `Callback` ABC
- `logging.py:31-69` — `ConsoleLogger`
- `early_stopping.py:4-41` — `EarlyStopping` (raises `StopIteration`)
- `checkpoint.py:48-126` — `ModelCheckpoint`

### 2.6 Config (`underPINN/config/loader.py`)

- `load_config` (`loader.py:68-87`) — YAML → nested `SimpleNamespace`
- `cfg_get(ns, *attrs, default)` (`loader.py:90-125`) — safe attribute traversal
- `save_config`, `merge_config`, `generate_sweep_configs`

### 2.7 Solvers (`underPINN/solver/`)

- `fbpinn.py:37-473` — `FBPINNSolver` (space-time, `lax.scan` fusion, RAR-D, restart)
- `steady_solver.py:13-219` — `SteadySolver` (2-D steady)
- `ode_solver.py`, `ldc_solver.py`, `rans_solver.py` — domain-specific
- `TrainingConfig` (`core/config.py:8-143`) — central dataclass

### 2.8 Runner dispatch (`underPINN/runner/dispatch.py`)

- 27 registered problem names → (script_path, function_name) triples
- Examples dynamically imported

---

## 3. PDE-Loss System

Every PDE implements `residual(params, xy)` returning the PDE operator applied to network output. Convention: `xy[:, 0:D_space] = spatial`, `xy[:, D_space:] = temporal`.

Examples:
- `NavierStokesPDE.residual` (`pde/navier_stokes.py:14-53`) — `jax.vmap(jax.jacfwd(u_fn))` for 1st derivs, `jax.vmap(jax.hessian(u_fn))` for 2nd. Returns `(N, 3)`.
- `SteadyNS3DPDE.residual` (`pde/navier_stokes_3d.py:31-73`) — full 3-D NS, computes uvwp + 4×3 Jacobian + 4×3×3 Hessian in one fused vmap, returns `(N, 4)`.
- `DiffusionPDE.residual` (`pde/diffusion.py:32-49`) — simple `u_t − α u_xx` via Jacobian+Hessian.

**BCs/ICs are NOT in the PDE object.** They're separate prediction-and-penalize terms in the runner's JIT step. See `examples/airfoil/airfoil_flow.py:246-281` for canonical pattern.

Reusable loss classes:
- `PINNLoss` (`losses/loss.py:51-101`) — packs `(x_r, t_r) → xt_r`, PDE + IC + BC + reg
- `RBA` (`losses/loss.py:59-68`) — element-wise residual-based adaptivity weights `w = |r|/(mean|r|+ε)`, **detached**
- `ODELoss` (`losses/ode_loss.py:36-58`) — supports 2nd-order ODEs

**RBA fix:** `losses/loss.py:64` — bug was weighting the already-reduced scalar; fix weights individual squared residuals.

---

## 4. FBPINN (Finite Basis PINN) — `nn/fbpinn.py:18-44`

```python
class FBPINN(nn.Module):
    layers, shifts, xs_min, xs_max, smins, smaxs, attention_cls
    out_transform: callable = None
    def setup(self):
        self.subnets = [SubdomainNetwork(...) for _ in range(self.shifts.shape[0])]
    def __call__(self, x):
        out = 0.0
        for i, net in enumerate(self.subnets):
            win = window_nd(x, self.xs_min[i], self.xs_max[i], self.smins[i], self.smaxs[i])
            out += net(x - self.shifts[i]) * win
        return out
```

- Overlapping subdomains with smooth sigmoid windows
- Each subdomain gets its own `SubdomainNetwork` with HybridAttention gates
- `out_transform` for positivity/scaling (e.g., k-ε constraint)
- **Status:** only `examples/fbpinn_ode/fbpinn_ode.py` uses it. Major flow examples use plain `mlp`/`gated_mlp`. Treat FBPINN as preliminary.

---

## 5. Configuration System

- YAML → `SimpleNamespace` (no Pydantic/schema validation)
- Silent typos default to safe values (real footgun)

**Top-level keys:** `problem`, `network`, `physics`, `domain`, `data`, `training`, `loss`, `output`

**Sweep files** use `base:` + `sweep:` sections; `generate_sweep_configs` expands the Cartesian product.

---

## 6. Training Loop & Callbacks

**`FBPINNSolver.train()`** (`solver/fbpinn.py:86-409`):
- Python-loop (`n_scan_steps=1`) vs `lax.scan` fusion (`n_scan_steps>1`) — 50–500× less overhead on GPU
- Mini-batch SGD with `safe_choice` (random with-replacement)
- Restart integration, RAR-D adaptive resampling, final `restart.done()` marker

**`SteadySolver.train()`** (`solver/steady_solver.py:47-196`): same architecture but interior + boundary points only.

**Restart/Resume:** `utils/restart.py:RestartManager` saves `params.msgpack`, `opt_state.msgpack`, loss histories, `meta.json`. `done: true` flag is the auto-resume lock. `resume` CLI subcommand MD5-checks config.

For time-marching pulsatile pipe flow, restart happens **per-window** (`utils/pulsatile_time_march.py`).

---

## 7. The Airfoil Example — `examples/airfoil/airfoil_flow.py`

**Closest existing analog to NeurIPS airfoil, but for pure PINN (not data-driven).**

- **PDE:** Steady 2-D incompressible Navier-Stokes `(u·∇)u + ∇p − ν∇²u = 0`, `∇·u = 0`
- **No training data** — PDE residual + BCs only
- **Inputs:** `(x, y)` coords. **Outputs:** `(u, v, p)`
- **Network:** tanh MLP `[2, 128, 128, 128, 128, 128, 128, 3]`
- **AoA:** realized by rotating the airfoil about quarter-chord (free-stream stays horizontal)
- **Training:** Adam + StepLR, 40k epochs, full-batch, resample collocation every 500 epochs
- **Loss:** `L = w_pde·L_pde + w_bc·L_bc`, JIT-compiled step with `jax.value_and_grad(..., has_aux=True)`
- **Outputs:** `airfoil_fields.png`, `airfoil_surface_pressure.png`, `loss_curve.png`, `velocity_profile.png`, `predictions.npz`, checkpoint

Config defaults: Re=100, AoA=0°, NACA 0012, chord=1, U_inf=1, domain (-5, 15, -5, 5), 10k interior + 3k wake collocation, 300 each for inlet/outlet/body/walls, buffer=0.02, epochs=40000, lr=1e-3, lr_step=5000, lr_gamma=0.5, resample_period=500, save_restart_every=1000, w_pde=10, w_bc=1.

**Geometry already supports NACA 4418** (cambered profile) — `NACAAirfoil(naca="4418", chord=1.0, aoa_deg=15.0)` works.

---

## 8. Examples Inventory (which is sim-to-real relevant)

| Folder | Physics | Sim-to-real? |
|---|---|---|
| `airfoil/` | 2-D steady NS, NACA 0012, pure PINN | Closest to NeurIPS but no data |
| `cylinder/` | 2-D steady NS, Re=40 cylinder wake, pure PINN | Benchmark |
| `pipe_flow/` | 3-D steady Hagen-Poiseuille (Re=40) | Forward only |
| `pipe_flow/pipe_flow_unsteady_transfer.py` | 3-D unsteady, Re + temporal transfer | **Yes** — closest existing |
| `pipe_flow/pipe_flow_pulsatile_transfer.py` | 3-D pulsatile via time-marching windows | **Yes** — exact recipe |
| `AAA/` | 3-D steady NS in axisymmetric vessel | Forward |
| `Aneurysm/` | 3-D steady NS in patient-specific STL | Forward |
| `transfer/burgers_transfer.py` | 1-D Burgers: parameter + temporal transfer | **Yes** — blueprint |
| `transfer/heat2d_transfer.py` | 2-D heat: parameter + temporal transfer | **Yes** |
| `heat/forward.py`, `heat/inverse.py` | 1-D heat, inverse (recover α) | **Yes** — pattern for data-informed |
| `burgers/` | 1-D Burgers ν sweep | Forward |
| `K-Epsilon/turbulence.py` | 2-D RANS k-ε, Re=10⁴ | Forward |
| `LDC/` | 2-D lid-driven cavity | Forward |
| `fbpinn_ode/` | 1-D ODE with FBPINN decomp | Forward |
| `ramp/`, `ramp_ns/`, `sod_shock/`, `toro3/` | Compressible Euler + NS w/ shock capturing | Forward |

**Most directly relevant patterns for sim-to-real airfoil:**

1. **`examples/pipe_flow/pipe_flow_pulsatile_transfer.py`** — time-marching transfer scaffold. Horizon split into `[0, dT]` windows; each window warm-starts from previous, uses previous end-state as IC. See `utils/pulsatile_time_march.py:1-595`.

2. **`examples/transfer/burgers_transfer.py`** — explicit `parameter_transfer` (different ν) + `temporal_transfer` (extended horizon) benchmarks. Lines 119-153 (parameter), 230-263 (temporal).

3. **`examples/heat/inverse.py`** — `DATA_W * mean((u_pred - u_obs)²)` is the canonical sim-to-real data term. Log-param `log_alpha` ensures positivity.

---

## 9. CLI / Entry Point

`underPINN/__main__.py:_cmd_run` (lines 22-33):
```python
def _cmd_run(args):
    cfg = load_config(args.config)
    runner = get_runner(cfg.problem)
    runner(cfg)
```

Subcommands: `run`, `sweep`, `list`, `show`, `bench`, `resume`, `status`, `version`
After `pip install -e .`: entry point `underpinn` registered via `pyproject.toml:30`.

CLI sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` before JAX import. Package `__init__.py` forces `jax_default_matmul_precision="highest"` on TPU.

---

## 10. Benchmarks

`benchmarks/run_benchmarks.py` (168 LOC) — thin CLI wrapper around `underPINN.benchmark_utils`.

`underPINN/benchmark_utils/`:
- `benchmark_suite.py` (342 LOC)
- `evaluators.py` (1823 LOC) — per-problem evaluators
- `report.py` (540 LOC) — PNG plots, CSV, Markdown summary

---

## 11. Repo Hygiene

- **CI:** `.github/workflows/ci.yml` — ruff pinned + pytest across Python 3.9–3.12 on CPU
- **Tests:** 2319 LOC, 10 test files. `test_examples.py` smoke-tests every example
- **Docs:** `README.md` (51 KB), `docs/index.html` (placeholder)
- **Recent activity:** last commit `ac2a040` on 2026-07-30. Active.

---

## 12. The Gap — underPINN vs RealPDE Track 1

### 12.1 Architectural mismatch (CRITICAL)

**Track 1 is a video prediction task on a fixed grid**, not a continuous PDE solver.
- Input: 20 frames of `(x, y)` flow at 32×64
- Output: 20 frames of `(u, v, p)` at 32×64
- Standard neural forecasting (FNO/Transformer/U-Net), not pure PINN

**underPINN's airfoil example doesn't fit.** It's a continuous field solver, not a temporal sequence predictor. Transfer learning patterns still apply (pre-train on sim, fine-tune on real) but you'd need a sequence model backbone, not underPINN's MLP.

**Stack mismatch:**
- underPINN = JAX/Flax
- Competition = **PyTorch only** (torch 2.2.2, no JAX)
- You'd port or rewrite, not use directly

### 12.2 What underPINN gives for free

- ✅ Airfoil geometry (`NACAAirfoil` handles NACA 4418)
- ✅ PDE residual machinery for 2-D NS
- ✅ Transfer learning infra (`solver.load_params(src_params)`)
- ✅ Training tricks PINNs need (RAR-D, RBA, JIT)
- ✅ FBPINN for high-Re turbulent flow
- ✅ Restart/resume

### 12.3 What you'd need to build

- ❌ Real-data loader (underPINN is pure PINN)
- ❌ Hybrid loss (PDE + data MSE + uncertainty)
- ❌ 3D unsteady NS extension
- ❌ Sequence model backbone (FNO/Transolver-style)
- ❌ TKE / MVPE / SPS eval
- ❌ Distribution-shift handling

---

## 13. Key Files to Start With

- `~/Desktop/sem7/UGP/underPINN/underPINN/examples/airfoil/airfoil_flow.py` — main runner to copy
- `~/Desktop/sem7/UGP/underPINN/underPINN/core/base.py` — base class contracts
- `~/Desktop/sem7/UGP/underPINN/underPINN/pde/navier_stokes.py` — 2-D NS PDE residual
- `~/Desktop/sem7/UGP/underPINN/underPINN/geometry/airfoil.py` — NACA profile + sampling
- `~/Desktop/sem7/UGP/underPINN/examples/heat/inverse.py` — closest sim-to-real pattern
- `~/Desktop/sem7/UGP/underPINN/examples/pipe_flow/pipe_flow_pulsatile_transfer.py` + `underPINN/utils/pulsatile_time_march.py` — time-marching transfer scaffold
- `~/Desktop/sem7/UGP/underPINN/underPINN/config/loader.py` — config schema/loader
- `~/Desktop/sem7/UGP/underPINN/underPINN/solver/fbpinn.py` — main training loop
- `~/Desktop/sem7/UGP/underPINN/underPINN/utils/checkpoint.py` + `utils/restart.py` — checkpoint/resume

> **Note:** The local `underPINN/` clone currently only contains `.git` + `.venv`. To access these files, run `git pull` inside `~/Desktop/sem7/UGP/underPINN/` to restore the source.
