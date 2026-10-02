# Handoff Brief — RealPDE Track 1 (Codabench #17363)

You are continuing a live ML-competition effort. This brief is self-contained. Read it
fully before touching anything. **Your work will be reviewed line-by-line afterwards, so
report what you actually measured, not what you hoped for.**

---

## 0. Access

| what | where |
|---|---|
| Compute | `ssh vm` → `/SML_DISK_24TB/rajeshr/Aryamann/UGP` |
| Python | `/SML_DISK_24TB/rajeshr/Aryamann/env/bin/python3` |
| Mac mirror (docs live here) | `~/Desktop/sem7/UGP` |
| Decision record | `~/Desktop/sem7/UGP/project_memory.md` (~3300 lines) |
| Gate checklist | `~/Desktop/sem7/UGP/TEST.md` |

**GPU etiquette.** `CUDA_VISIBLE_DEVICES=0` is pinned in the VM profile, so torch always
reports `device_count 1`. Select a GPU with `CUDA_VISIBLE_DEVICES=N ... --gpu 0`, never
`--gpu N`. Check `nvidia-smi` first — the box is shared with a labmate (`phyguest`,
`rajeshr`) and has 62 GB RAM; several caches are 2–4 GB, so watch `free -g`.

Launch long jobs detached:
```bash
ssh -f -n vm "cd /SML_DISK_24TB/rajeshr/Aryamann/UGP && CUDA_VISIBLE_DEVICES=2 \
  setsid nohup /SML_DISK_24TB/rajeshr/Aryamann/env/bin/python3 -u script.py \
  > logs/x.log 2>&1 < /dev/null"
```
A plain `ssh vm '... &'` holds the channel open and blocks. **Never** `pkill -f <pattern>`
where the pattern also matches your own ssh command string — it kills the session.

**Write findings to `project_memory.md` incrementally, as you get them.** Four agent runs
have been interrupted mid-task and everything unwritten was lost each time.

---

## 1. Current state

**Banked best: `79.246428`** — `submission_SHIFT_v2_W96_a85.zip`.

| subscore | value |
|---|---|
| rel_l2 | 94.045224 |
| tke | 75.998641 |
| mvpe | 92.873514 |
| time | 90.304009 |
| sps | 37.480269 |

Leaderboard rule is **`Force_Best`**: a bad submission costs that day's slot, never the
banked score. **Limit is one submission per UTC day (reset 00:00 UTC = 05:30 IST), and a
FAILED submission still consumes the day.** Development Phase closes **27 Sep 2026**.

**Top-50 cutoff: `80.544`.** Gap from banked: **+1.298**.

### The method, in one paragraph
The point predictions come from a frozen model soup of fine-tuned FNOs (`soup_v1`). They
are held **bit-identical** across submissions, which locks rel_l2/tke/mvpe and makes every
score change attributable to bounds or time alone. On top, a U-Net predicts a per-element
**bound centre correction** `c` and a **half-width** `h`, and we emit
`lower = pred + α·c − h`, `upper = pred + α·c + h` with α = 0.85. The `prediction` array
itself is never modified.

### Is the off-centre bound legal? Yes — settled.
The Codabench Evaluation page enumerates the constraints on participant bounds
exhaustively: shape must match the prediction, values finite, `lower <= upper`. Its
"Zero-Score Conditions" list exactly those. The reward is defined purely on the target:
`inside = (lower <= target <= upper)`, `nil = (upper - lower)/sigma_global`. Nothing
requires the interval to contain or be centred on `prediction`. The organizers explicitly
note "narrow intervals around bad predictions cannot buy score", confirming the intent is
to reward covering the *target*. `sigma_global = 0.0563870`. **Do not re-litigate this.**

---

## 2. The scoring model — use these numbers

Solved by least squares on the top-40 leaderboard rows (all five subscores each),
**max residual 0.0050 over 40 rows**; it predicts our own banked row at 79.2618 vs the
actual 79.246428:

```
final = 0.46743·rel_l2 + 0.10027·tke + 0.09420·mvpe + 0.09689·time + 0.24737·sps + 0.9119
```

⛔ An older fit (0.306/0.163/0.218/0.100/0.217) appears throughout early sections of
`project_memory.md`. **It is superseded — its sps weight is 14% too small.** Any ΔFINAL
quoted before §32.5 must be multiplied by 1.140.

Subscore mechanics (from `scoring.py`, verified):
```
score      = 100 / (1 + 0.5·error)
sps        = 100 · W · E
W          = 0.5(1−n(dm)) + 0.3(1−n(tke)) + 0.2(1−n(mvpe)),   n(x) = x/(0.5+x)
E          = mean over scored elements of exp(−(upper−lower)/sigma) · 1[target inside]
time_score = 100 / (1 + sqrt(t / 0.72896))
```

**Because point predictions are bit-identical across all our artifacts, `W = 0.684538`
exactly, so `sps` is a pure readout of `E`:**

```
E            = sps / 68.4538
Δfinal       = 16.936 · ΔE
+1 sps point = +0.2474 final
+1 time point= +0.0969 final
```

Marginal value per subscore point (direct only): **rel_l2 +0.467, tke +0.100,
mvpe +0.094, time +0.097**. rel_l2 is worth **4.7× tke** — and our rel_l2 (94.05) is the
**lowest among the top 45** (they run 94.31–95.01; the stock kit baseline is 94.17, so our
fine-tuning made rel_l2 *worse*).

### Five real anchors (all on the same checkpoint — this is our calibration set)

| artifact | bounds configuration | real sps | real E |
|---|---|---:|---:|
| `SHIFT_v2_W96_a85` (banked) | centre shift, α=0.85, h_u ratio 0.776 | 37.4803 | 0.54753 |
| `SOUP_v1` | shipped 24-bin LUT | 34.3656 | 0.50203 |
| `LUTFIX` | refit LUT | 33.9166 | 0.49547 |
| `WIDE125` | LUT × 1.25 | 33.8038 | 0.49382 |
| `ASYM_W96_safe_arcsinh` | squeezed, h_u ratio 0.3876 | 27.8581 | 0.40696 |

Top-of-board E for reference: benslash2 0.585, doomduke2 0.607, np-user 0.613.

---

## 3. ⛔ CLOSED — do not re-run any of these

Each cost real GPU time or a real submission slot.

**Bounds / uncertainty**
- Post-hoc LUT recalibration; fitting a local→real scalar error scale. The parametric
  family is **misspecified**: best achievable residual is 0.0128 even with three anchors.
- LUT width scaling in **both** directions — measured live, both lose.
- Asymmetric / offset *global* bounds (+0.0004; signed error is near zero-mean).
- Target quantisation (targets are continuous float32; no atoms).
- FNO input-perturbation channels (corr 0.17–0.27 vs the U-Net's 0.66).
- **Width head**: five architectures, three objectives (log-error regression, quantile
  regression, direct SPS surrogate), 4× capacity range — all cap at **22–23%** of captured
  headroom. An in-sample overfit probe also caps at 21.5% ⇒ **information limit**.
- **Centre predictor**: a 2.8× parameter sweep (7.6M→21.1M), three losses (L1/Huber/L2)
  and a genuinely different function class (3-D spectral operator, same family as the base
  FNO) **all land at β = 0.12–0.17 of oracle**. The spectral operator is *worse* than the
  U-Net. Reaching +1.30 needs β ≈ 0.6–0.73. **Ceiling for the class — 4× short.**
- Centre ensembling: w64+w96+w128 blend buys only +0.05…+0.08 final.
- Per-bin centre offset: nothing.
- α (centre strength) sweep: **α = 0.85 is already the live optimum**, flat top.

**Accuracy**
- Fine-tuning recipe exhausted: 16+ configs across lr, weight decay (no effect across four
  orders of magnitude), steps, tke weight, data amount, init, soup size — all tie, on both
  easy and condition-disjoint splits.
- `soup_v3` (strict-holdout retrain) is +0.07 tke on honest holdout ⇒ an illusion.
- `soup_v2` is **unusable** — trained on 100% of data, so it leaks into any bounds fitted
  on a holdout.
- mvpe in the training loss (its gradient touches 1.2% of elements).
- EMA of weights; prediction ensembling; ensemble distillation (+0.02 over the soup).
- "Better model ⇒ more predictable error": measured — captured rises but E *falls*.
- Test-time augmentation: closed by construction.

**Inference**
- fp16 / bf16: unavailable. cuFFT rejects the 20×32×64 transform in half precision
  (20 is not a power of two). All time gains must come from CPU-side overhead.

---

## 4. ⚠️ The three traps that have cost us submissions

Read these carefully; they are the most valuable content in this brief.

**(a) Local optima do not transfer.** On 2026-08-31 a variant with bounds squeezed to
`h_u` ratio 0.3876 — provably optimal on the local `re_lohi` holdout — scored **76.70**
live, with sps collapsing 37.48 → 27.86. The live error distribution is wider than local;
tight bounds miss coverage and SPS punishes that severely. **The `h_u` ratio is a LOCAL
metric that anti-correlates with live score across everything we have tested (0.776 →
79.25; 0.3876 → 76.70).** Two further slots were lost the same way (LUTFIX 78.34,
WIDE125 78.30). *"Mathematically optimal on the local holdout"* is **not** a justification.

**(b) Prefer moves whose SIGN is calibration-independent.** The centre shift works because
at fixed widths only the coverage indicator moves, so its sign cannot flip with the
unknown calibration. A smaller gain with a guaranteed sign beats a larger one that depends
on the error model being right.

**(c) Check what a checkpoint was TRAINED on before evaluating it.**
`local_harness/finetune.py:32` holds out `every5`, so the `re_lohi` / `aoa15` / `aoa0`
splits are **81–100% training conditions** for every existing fine-tune. A whole line of
work ("best member beats the soup") turned out to be a training-data-fit artifact.

**Validation discipline.** Validate on condition-disjoint splits: `re_lohi` (hold out Re
3750/5025/25425/26700), `aoa15`, `aoa0`. The historical `every5` split over-states gains
~1.9×. A result on one split is not a result — confirm on two independent ones. The max
over N variants on one split is biased upward; re-score on a split you did not select on.

---

## 4b. ⛔ ARCHIVE SIZE IS A BINDING CONSTRAINT — price every idea in MB

The banked archive extracts to **230,262,703 B = 85.8% of the 256 MiB limit**, leaving
only **38.2 MB** of headroom. Net sizes: W64 ≈ 6.8 MB fp16, **W96 15.3 MB fp16** (30.5
fp32), W128 27.1 MB fp16 (54.1 fp32).

⇒ **At most TWO extra W96 nets can ship.** The best measured ensemble (5 seeds, minimax
worst +0.062) is **NOT deployable**. The `w96·.4 + w128·.4 + w64·.2` fp16 blend needs
W64+W128 extra = 33.9 MB and does fit (worst +0.040, mean +0.086).

**Every proposal must state its MB cost alongside its point gain.**

---

## 5. What is OPEN — your actual work

Two workstreams. **Do not submit anything.** Build, measure, gate; a human decides.

### Workstream A — finish the guaranteed artifact (target ≈ 79.37)

Two components are measured and both have guaranteed sign:

1. **One-pass speedup, +0.075.** `_agent2/submission_fast.py`. One pass instead of two:
   the FNO output feeds the U-Net on-device (no host round trip), the input window is
   uploaded once, and `prediction`/`lower`/`upper` are returned as contiguous whole-slice
   copies. Verified `max|dPREDICTION| = max|dLOWER| = max|dUPPER| = 0` — byte-for-byte
   identical, so only `time` can move. Ratio 0.833 over 9 interleaved reps.
2. **Centre ensemble, +0.052 worst case.** `w96·0.4 + w128·0.4 + w64·0.2` with α re-swept
   from 0.85 to **0.95**. Minimax over 364 admissible calibrations: worst **+0.0524**,
   mean +0.0662, best +0.0732, re_lohi +0.0811. Positive under *every* admissible
   calibration. Nets are in `_agent2/`.

**Open sub-question (the actual work):** the 5-net answer is known but undeployable, and
the **deployable frontier (≤ 2 extra nets) has not been mapped**. Find the ≤2-extra-net
mix that maximises the minimax-worst gain, and quantify what fp16 quantisation costs
versus fp32. Then combine with the speedup into one zip and gate it end to end.

⚠️ Two facts that constrain this, both measured at the checkpoint:
* **Weight-souping the centre nets is catastrophic (−3.76 to −3.80 final).** Twelve nets
  with identical init and schedule, differing only in data order, still land in different
  basins. Average the **outputs**, never the weights.
* **α must move 0.85 → 0.95 for any ensemble** — averaging shrinks |c| so the multiplier
  must grow. Running an ensemble at α = 0.85 gives back about a third of the gain.
⚠️ Re-run GATE 2 and the byte-identity check **on the final combined bytes** — the
speedup touches the output path and the ensemble changes the bounds, so neither result is
inheritable from the component tests.

### Workstream B — `train_sim`, the one unused asset

`train_sim` is the official release's **simulated** dataset. It has been downloaded to the
VM and partially characterised (see §33.10). **Every accuracy attempt in this project has
fine-tuned on `train_real` alone**, and that recipe is exhausted — so sim-to-real
pretraining or joint sim+real training is the only structurally new accuracy lever left,
and rel_l2 (where we rank last and which is worth 4.7× tke) is the target.

**Before building anything on it, verify the distribution matches `train_real`:** shapes,
per-channel mean and std, Reynolds coverage, angle-of-attack coverage, and masked-zero
fraction. Report those numbers explicitly.

> Why this matters: §2 of the decision record documents six proxy failures caused by
> training on data whose statistics diverged from the competition set (u std 0.0040 vs
> 0.0217; within-field CV 1.3% vs 47%). It went unnoticed for weeks and invalidated five
> submissions' worth of local analysis.

Also confirm in writing that `train_sim` is inside the rules. It should be — it is part of
the official release — but GATE 0 violations are **disqualification, not a low score**.

**Establish its ceiling and stop.** If sim-pretraining cannot plausibly deliver ≈+1.3,
record the measured ceiling in `project_memory.md` and say so. A well-measured negative
that closes a route is a good outcome.

### Jobs already running on the VM — HARVEST, do not re-run
`joint2.py` seed sweep (tags `s*`/`i*`), `ftaug.py H_phase` and `J_noaug`,
`centre_eval.py --split aoa15`, and a `gdown` of `train_sim`. These are detached and will
finish on their own, writing to `_agent2/` and `train_es/`. Check them before launching
anything; they may already answer your question. GPUs 1–3 may be busy for a while.

⚠️ `ftaug` already has a partial answer: **both arms are net-NEGATIVE past step 2000**
versus the `J_noaug` control. Augmented fine-tuning does not rescue accuracy. Confirm from
the finished logs, record the ceiling, and move on.

### Also finish (cheap, already running or nearly so)
- `centre.py --tag P_insample160` — an **in-sample overfit probe** on the signed residual.
  If it also lands near β 0.17, that upgrades "ceiling for the class" to "information
  limit on the signed residual", which is a stronger and more durable statement.
- `ftaug.py --tag H_phase` vs `--tag J_noaug` — report the delta between both arms.

---

## 6. Deliverables

0. **Read `project_memory.md` §34 first** — it is a checkpoint written at handover and
   is the single most current summary of state, constraints and open questions.
1. **One zip** that is a guaranteed improvement on 79.246428, with a gate table.
2. **One zip** that is the best available shot at 80.544, with its downside stated.
3. A table of every method evaluated, with its **measured ceiling in final-score points**
   and what that ceiling rests on.
4. `project_memory.md` updated incrementally as a new numbered section, negatives included.

### Gates every artifact must pass (from `TEST.md`)
- **GATE 0** — every weight traces to the official release (competition Google Drive:
  `train_real`, `train_sim`, `baseline_checkpoints`). **No HuggingFace weights.**
- **GATE 1** — validator `python local_harness/_val.py <zip> --height 32 --width 64` must
  be **13/13**; extracted size **< 256 MB**; no `__pycache__`/`.pyc`; `unzip -t` clean;
  md5 identical on Mac and VM; entry list diffed against the banked artifact with every
  difference justified.
- **GATE 2** — extract the zip and call **its own** `predict()`: shapes `(N,20,32,64,3)`;
  all finite; `lower <= upper`; `prediction[...,2] == 0`; bounds genuinely vary (std > 0).
- **GATE 3** — execution limit is **300 s** (5 min). Measure the **ratio** to the banked
  artifact, not absolute ms — the VM is noisy (the same zip measured 2.69 and 4.57
  ms/sample minutes apart). Test the CPU path in a subprocess.
- **GATE 4B** — any per-element bounds policy must beat the best **constant** bounds on
  the same held-out data.

### Build mechanics — these have each broken a build before
- `zip -q -r -X -D` — **without `-D`** you get phantom directory entries that fail the
  entry-list diff.
- The checkpoint filename inside the zip **must** be `sim_real_fno_fp16.pth`;
  `load_baseline` infers the architecture from the filename.
- The FNO has **16 COMPLEX spectral tensors**. When repacking to fp16 branch on
  `v.is_complex() or torch.is_floating_point(v)`. Using `.float()` silently drops the
  imaginary parts (a soup scored −7.34 that way); `v.half() if v.is_floating_point()`
  skips complex tensors entirely and leaves a 403 MB file.
- Bounds are **all-or-nothing across the run**: `predict` may be called more than once,
  and if some calls return bounds and others do not, every bound is discarded.

---

## 7. How to report back

For each thing you tried, state:
1. what it was;
2. the **measured** number and how it was measured (which split, how many windows, whether
   the checkpoint was trained on that split);
3. what it rests on and what would falsify it;
4. whether it is open or closed, and its ceiling in final-score points.

Distinguish clearly between **measured** and **projected**. If a projection rests on an
assumption, name the assumption. If something does not hold up, say so plainly — three
submission slots have been lost to confident numbers that were locally true and live false.
