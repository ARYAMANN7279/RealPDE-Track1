# TASKS_ROUND8 — WiSE-FT weight interpolation toward the kit checkpoint

**Author: Claude (design + gates). Executor: Gemini 3.1 Pro (labour only).**
**Date: 7 Sep 2026. Banked: 79.462591 (`submission_SV2.zip`).**

---

## 0. READ THIS FIRST — rules for this round

1. **Report raw numbers. Do not interpret them, do not decide what to submit, do not
   build a submission zip.** This round produces a table and checkpoints, nothing else.
   I price the result and decide.
2. **Never write a live score into `project_memory.md` unless it appears in the Codabench
   feed.** On 7 Sep a live score for `submission_SV2_V4_GOLD.zip` was written into §16 that
   had never been submitted. That zip is byte-identical to `submission_FP16.zip`
   (already scored 79.378881). Do not repeat this.
3. If any step below fails or a file is missing, **stop and report the exact error**.
   Do not substitute a different file, a different split, or a different metric.
4. Scratch only under `/SML_DISK_24TB/rajeshr/Aryamann/UGP/`. Never `/tmp`, never the VM
   root partition.

---

## 1. Why this round exists (context you need, not opinion to act on)

The `ftaug_r2w*` sweep and `H_phase` / `J_noaug` runs on the honest `re_lohi` split show a
consistent pattern I had not seen before:

| run | rel_l2 | tke | mvpe | d_acc |
|---|---:|---:|---:|---:|
| **base (kit checkpoint, no fine-tune)** | **95.4738** | **75.8957** | **96.0945** | 0 |
| `J_noaug` best | 95.1541 (−0.320) | 77.7077 (+1.812) | 96.0153 (−0.079) | +0.2046 |
| `H_phase` best | 95.1023 (−0.372) | 77.6068 (+1.711) | 95.8671 (−0.227) | +0.1165 |
| `I_phasenoise` best | 95.0984 (−0.375) | 77.5972 (+1.702) | 95.8625 (−0.232) | +0.1113 |

★ **Every fine-tune LOSES rel_l2 (~−0.35) and GAINS tke (~+1.8).** rel_l2 is the subscore we
are last-in-the-room on (rank 50 of 50 among the top-50 cohort) and it is worth 4.3× tke per
point. We have only ever sampled the two ENDPOINTS of this trade: α=0 (raw kit) and α=1
(fully fine-tuned soup). **The interior of that line has never been evaluated.**

This is textbook WiSE-FT: fine-tuning degrades out-of-distribution robustness, and blending
the fine-tuned weights back toward the pre-trained anchor recovers it. Our task *is*
distribution shift (sim→real, unseen Re/AoA). The soup members were all fine-tuned **from**
the kit checkpoint, so they sit in the same loss basin and the blend is valid.

**Two properties that make this cheap and safe:**
* **No training.** It is a weight average plus evaluation.
* **`time` is invariant by construction** — identical architecture, identical inference path,
  only the weight values change. The main regression risk on any accuracy work is neutralised.

Also already measured and closed, so do not propose them: input noise (`I_phasenoise` vs
`H_phase` → d_acc +0.1113 vs +0.1165, neutral-to-negative) and lowering `wtke` to buy rel_l2
(sweep peaks at wtke=0.08: 0.03→+0.0972, 0.05→+0.1313, **0.08→+0.1519**, 0.15→+0.1288).

---

## 2. Exact inputs (verified present, md5s confirmed 7 Sep)

| role | path (relative to `/SML_DISK_24TB/rajeshr/Aryamann/UGP`) | md5 / size |
|---|---|---|
| **BASE** kit checkpoint (α=0 anchor) | `data/comp_real/sim_real_fno.pth` | 402,988,026 B, fp32 |
| **HONEST soup** (holds out `re_lohi`) — the ruler | `train_es/soup_v3_fp16.pth` | `b3255e22ab0d39c0b71f5bfa52807c7d` |
| **SHIPPED soup_v2** (100% data, banked backbone) | `local_harness/soup_v2_fp16.pth` | `571062ce3d3ce4313964046ca60feade` |
| shipped soup_v2, fp32 source | `local_harness/soup_v2.pth` | `c9da3ac87bdd8966ba32ea7959351e79` |
| eval harness to copy from | `ftmv.py` (lines 34–56 data/split, 88–99 evaluate/d_acc) | — |

---

## 3. THE TASK

Write `/SML_DISK_24TB/rajeshr/Aryamann/UGP/wiseft.py`. It must:

### 3.1 Build the blend
For a given fine-tuned checkpoint `FT` and coefficient `α`:

```
blended[k] = (1-α) * BASE[k] + α * FT[k]
```

**Rules that matter — get these exactly right:**
* Load **both** state dicts to **fp32 on CPU** before blending. `soup_v3_fp16.pth` and
  `soup_v2_fp16.pth` are fp16; `sim_real_fno.pth` is fp32. Blending in fp16 loses precision.
* If a state dict is wrapped (`{"model_state_dict": ...}`), unwrap it — see `ftmv.py:75`.
* **Interpolate ONLY floating-point tensors.** For any non-float tensor (integer buffers,
  `num_batches_tracked`, etc.) copy the value from `FT` unchanged. Blending an integer
  buffer silently corrupts it.
* Assert the two state dicts have **identical key sets** before blending. If they differ,
  stop and report the differing keys — do not blend a subset.
* α = 1.0 must reproduce `FT` **exactly** (max abs diff 0 after the fp32 round-trip against
  a fp32 cast of FT). Assert this as a self-test before running the sweep.

### 3.2 Evaluate on the honest split
Reuse `ftmv.py`'s evaluation verbatim — do not write a new metric:
* split `re_lohi` (`ftmv.py:38`), the same `va_sub` 900-window sample with `rng =
  np.random.default_rng(0)` (`ftmv.py:51-52`) so results are comparable to the table in §1.
* `evaluate()` from `ftmv.py:88-96`, returning `(rel_l2, tke, mvpe)` via `S.score_error`.
* `d_acc` from `ftmv.py:97-99` with `MV = dict(rel_l2=0.669, tke=0.157, mvpe=0.170)`,
  **measured relative to α=1.0**, not relative to the kit base. I need the delta against
  what we currently ship.

### 3.3 The sweep
Run **both** checkpoints, full grid:

```
FT ∈ { train_es/soup_v3_fp16.pth   (HONEST ruler — this is the one that decides)
       local_harness/soup_v2_fp16.pth (SHIPPED — cross-check only) }
α  ∈ { 0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00 }
```

26 evaluations. Each is a forward pass over 900 windows — no training. Use one idle GPU
(check `nvidia-smi` first, pick via `CUDA_VISIBLE_DEVICES=N`).

### 3.4 Output
Write `/SML_DISK_24TB/rajeshr/Aryamann/UGP/wiseft_results.json`:

```json
{"soup_v3": [{"alpha": 0.0, "rel_l2": ..., "tke": ..., "mvpe": ..., "d_acc_vs_alpha1": ...}, ...],
 "soup_v2": [ ... same ... ]}
```

and print one line per row. Save **only** the two blended checkpoints at the best α per
model, as `train_es/wiseft_v3_a<NNN>.pth` and `train_es/wiseft_v2_a<NNN>.pth` (fp16, same
format as the inputs). Do not save all 26 — that is 10 GB of no value.

---

## 4. Gates — report these, do not act on them

Report the table and these three facts. **I decide what happens next.**

1. **α\* on the HONEST ruler** (`soup_v3`): the α maximising `d_acc_vs_alpha1`.
2. **Is α\* < 1.0 on the honest ruler, and is `d_acc_vs_alpha1(α*) ≥ +0.05`?**
3. **Does `soup_v2` (leaky ruler) put its α\* in the same region (within ±0.15)?**

Note for your own sanity-checking, not for you to act on: the leaky ruler is biased *toward*
α=1, because `soup_v2` trained on all 81 trajectories including `re_lohi`. So if the leaky
ruler ALSO prefers α<1, that is a conservative, strengthened result.

**Expected shapes, so you can spot a bug:**
* At α=0 you must recover approximately `rel_l2 95.4738 / tke 75.8957 / mvpe 96.0945` for the
  `soup_v3` run. **If α=0 does not reproduce the base triple to ~0.01, the harness is wrong —
  stop and report.**
* d_acc at α=1.0 is 0 by definition.

---

## 5. What NOT to do

* ⛔ Do not build any submission zip this round.
* ⛔ Do not retrain anything. No `ftmv.py` runs, no new soup members.
* ⛔ Do not "improve" the metric, the split, or the window sample.
* ⛔ Do not delete any existing checkpoint.
* ⛔ Do not edit `project_memory.md` beyond appending the raw results table under a heading
  `## 67. ROUND 8 RAW RESULTS (Gemini, executed)` — no conclusions, no recommendation,
  no projected live scores.
