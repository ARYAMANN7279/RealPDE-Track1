# UGP — Track 1 Sim-to-Real Airfoil Pipeline

**Topic:** NeurIPS 2026 RealPDE Competition — Track 1: Sim2Real on NACA airfoil flow  
**Team:** Prof. Rajesh Ranjan (leader, IIT Kanpur Aerospace), Aryamann Srivastava, Prashant Kumar  
**Date started:** 2026-08-01  
**Last updated:** 2026-08-06  
**Status:** Demo submission packaged (`outputs/submission_demo.zip`) · Best result: **91.95** (U-Net) · PyTorch Blackwell fix pending · Registration by **Aug 20**

> 📖 **Single source of truth:** [`project_memory.md`](project_memory.md) — contains full context, session log, decisions, commands, and GPU server info. Read this first in every new session.

---

## What is this project?

We are **seriously entering** the NeurIPS 2026 RealPDE Track 1 competition as a professor-led team of 3.

- **Competition:** Codabench #17363 — Simulation-to-Real Transfer Learning on paired NACA airfoil flow data
- **Prize pool:** $6k / $3k / $1.5k per track. Top 5 co-author NeurIPS results paper.
- **Organizer:** Tailin Wu (Westlake University). Active participants: 201.
- **Our approach:** Pre-train on CFD (sim) data, fine-tune on PIV (real) data using a U-Net with FiLM domain conditioning and a heteroscedastic head for SPS bounds.

---

## Folder layout

```
UGP/
├── README.md                        ← this file
├── PROGRESS.md                      ← one-page status summary for the professor
├── project_memory.md                ← canonical context for new AI sessions (READ THIS FIRST)
├── REALPDE_T1.zip                   ← source zip of the main codebase (archived)
├── docs/
│   ├── competition_spec.md          ← Codabench #17363 full spec
│   ├── RealPDEBench_overview.md     ← RealPDEBench codebase analysis
│   ├── underPINN_overview.md        ← underPINN codebase analysis
│   └── track1_realpde_overview.md   ← REALPDE_T1 bundle analysis (14 rounds, best=91.95)
├── code/
│   ├── RealPDEBench/                ← clone of RealPDEBench (upstream reference)
│   ├── track1/                      ← early pipeline skeleton (superseded, kept for reference)
│   │   ├── models/fno.py            ← FNO3d (50 M params, MPS smoke-tested)
│   │   ├── evaluation/scorer.py     ← 5-metric scorer (official spec)
│   │   ├── data/dataset.py          ← HF streaming data loader
│   │   └── end_to_end.py            ← full pipeline demo (final_score 61.6 with random weights)
│   └── track1_realpde/              ← ★ PRIMARY CODEBASE — adopt this ★
│       ├── realpde/                 ← 5 architectures + training + metrics + normalizer
│       ├── configs/                 ← YAML configs (FNO2d, U-Net, UFNO, smoke)
│       ├── scripts/                 ← fetch_data, make_synthetic, make_submission, validate
│       ├── submission/              ← submission.py (Codabench entry point)
│       └── reports/                 ← 14-round experimental report (best=91.95 U-Net)
├── underPINN/                       ← git clone of underPINN framework (JAX-based PINN)
├── data/                            ← dataset index files
├── results/                         ← output figures, logs, checkpoints
└── venv/                            ← Python virtualenv (torch 2.13.0 + RealPDEBench)
```

---

## The pipeline

```
HF Dataset (RealPDEBench foil — NACA0025 or NACA4418, TBD)
  → fetch_data.py (streaming, sub_s=4 → 32×64)
  → TrajectoryStore → FoilWindowDataset (sliding windows, T_in=20, T_out=20)
  → Normalize (per-channel mean/std)
  → Stage 1: pretrain on numerical/CFD  (UNetTime, FiLM conditioning domain=0)
  → Stage 2: finetune on real/PIV       (FiLM LR split: backbone×0.3, FiLM+heads×10)
  → Evaluate: Rel-L2, TKE, MVPE, Time, SPS (all 5 Track 1 metrics)
  → make_submission.py → submission.zip → Codabench
```

---

## Stack

- **Data:** HuggingFace `AI4Science-WestlakeU/RealPDEBench` (foil subset)
- **Models:** UNetTime (best), FNO2dTime, UFNOTime — all in `realpde/model.py`
- **Training:** PyTorch 2.13.0 (MPS locally; A100/H200 on cluster)
- **Metrics:** 5 official Track 1 metrics reproduced in `realpde/metrics.py`
- **Submission:** `submission.py` — zeroes p-channel in feedback, autoregressive rollout, emits prediction + bounds

---

## Key results (from 14-round experimental study, `reports/experimental_report.pdf`)

| Config | rel_l2 | TKE | **final_score** |
|---|---|---|---|
| FNO2d+residual baseline | 0.0181 | 0.436 | 91.62 |
| FNO2d+residual finetune | 0.0182 | 0.419 | **91.77** |
| **U-Net baseline** | **0.0175** | **0.395** | **91.95** ← current best |
| U-Net finetune | 0.0171 | 0.400 | 91.91 |

> **U-Net beats every FNO variant on every accuracy metric, 3-seed confirmed.**
> The `tke_weight=0.1` loss term (Round 8) is the key win: it directly optimises TKE since pointwise MSE doesn't push the model to match fine-scale fluctuation statistics.

---

## What's done

- ✅ Full competition spec extracted (`docs/competition_spec.md`)
- ✅ RealPDEBench cloned and analyzed (`code/RealPDEBench/`, `docs/RealPDEBench_overview.md`)
- ✅ underPINN analyzed (`docs/underPINN_overview.md`)
- ✅ Early pipeline skeleton (FNO3d + scorer + data loader + e2e demo, `code/track1/`)
- ✅ Complete REALPDE_T1 bundle adopted (`code/track1_realpde/`, 5 architectures, 14 experimental rounds)
- ✅ Normalizer decode_logvar bug fixed (raises final_score by 1.5–1.8 pts)
- ✅ NACA4418 vs NACA0025 discrepancy identified and flagged

## What remains

- [ ] Run `make_synthetic.py` end-to-end smoke test locally (no download needed)
- [ ] Run `validate_submission.py` on a dummy submission
- [ ] **Fill team registration form by Aug 20, 2026** → https://forms.gle/LYeTgUTfr4ygntNu6
- [ ] Cluster setup: SSH to A100/H200, update hard-coded paths in YAML configs
- [ ] Retrain U-Net baseline on cluster (verify 91.95)
- [ ] Try `use_input_stats: true` ablation (untested, potential +0.X)
- [ ] Submit to Codabench — get account approved, smoke submission

---

## Key dates

| Date | Event |
|---|---|
| **Aug 20, 2026** | ⚠️ Team registration deadline |
| Sep 27, 2026 | Main Development phase ends |
| Oct 25, 2026 | Final Decision phase ends |
| Nov 25, 2026 | Code & fact sheet deadline |
| Dec 6, 2026 | NeurIPS presentation |

---

## For AI assistants

Before doing anything, read `project_memory.md` for canonical context. Update §8 (session log), §9 (not done), and §10 (decisions) at the end of each session. **All files must go in `~/Desktop/sem7/UGP/`.**

**Last updated:** 2026-08-06
