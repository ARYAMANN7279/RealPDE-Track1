# Handoff #2 — Task B is UNBLOCKED. The data lever is real.

**BANKED: 79.462591** (`submission_SV2.zip`, the 100%-data backbone). **+0.0837** over the
previous 79.378881. Do not submit; a human spends the one daily slot (05:30 IST reset).
A FAILED submission still consumes it.

```
final = 0.46743*rel_l2 + 0.10027*tke + 0.09420*mvpe + 0.09689*time + 0.24737*sps + 0.9119
```
New subscores: rel_l2 **94.050465** · tke **76.900655** · mvpe **93.130900** · time **90.251551** · sps **37.887566**
Marginal value/pt (incl. W→sps): rel_l2 0.669 · mvpe 0.170 · tke 0.157 · time 0.097

---

## THE RESULT, AND THE NUMBER YOU NEED

| | in-sample (leaky) | **LIVE** | transfer |
|---|---:|---:|---:|
| soup_v2 tke advantage over soup_v1 | +3.267 | **+0.902** | **27.6%** |

soup_v1 trained on **65 of 81** trajectories (`every5` holdout); soup_v2 on **all 81** —
+25% data ⇒ **+0.90 live tke ⇒ +0.090 final.**

⚠️ **Use 27.6%, not 90%, for anything measured on a split the model trained on.** The ~90%
transfer rate from the tke investigation applies to *honestly measured* gains. These are
different rulers; do not mix them.

⚠️ **My previous handoff said SV2 was a zero-expectation coin flip. That was wrong.** Leakage
inflating a measurement means the measurement cannot SIZE the effect — not that the effect is
absent. I conflated those. Cost: nearly skipped a +0.084 submission.

---

## TASK B (GO) — build SV3 the right way

The naive version ("train 7 more 100%-data members and soup them") is **low value**: souping
is honestly worth only **+0.055 tke** over the mean member (§64.3), and 81 trajectories is
already all of `train_real`, so there is no more data to add. The win came from DATA, not
from souping, and that specific well is now dry.

**Do this instead — validate the recipe honestly, then retrain it on everything:**

1. **Find the best recipe on the HONEST split.** You already have the instrument: your 7
   members trained with a genuine `re_lohi` holdout (NEW_SOUP, honest d_acc +0.7637). Sweep
   on that ruler — `wtke` ∈ {0.05, 0.08, 0.12}, `lr` ∈ {1e-5, 3e-5}, steps ∈ {8k, 16k, 32k}.
   Pick the winner by **CORRECT** MV `(rel_l2 0.669, tke 0.157, mvpe 0.170)`.
2. **Retrain that single winning recipe on all 81 trajectories** (`--split none`), 4–6 seeds.
   ⚠️ The seed is hardcoded 1234 — add a `--seed` argument or every run is bit-identical.
3. **Soup them** (complex-safe: `v.to(torch.complex64) if v.is_complex() else v.float()`;
   see `local_harness/average_soup_v2.py`).
4. Hand back the fp16-packed checkpoint. **Do not build a submission zip.**

This is standard practice — validate to choose hyperparameters, then refit on all data — and
it is the only way left to combine "more data" with "better recipe", since 100%-data models
cannot be checkpoint-selected at all.

**Expected:** the souping increment (+0.055 tke honest) plus whatever the recipe search buys.
Price it modestly. The +0.90 from data will NOT repeat; that lever is spent.

---

## TASK D (NEW, cheap, do first) — explain the time regression

SV2 lost **−0.157 time points (8.205 → 8.505 ms), −0.0152 final** — a third of the tke gain,
for **the same architecture and the same code**. Only the checkpoint weights differ.

Hypotheses worth one hour: weight-dependent cuDNN/cuFFT algorithm selection; denormals in the
fp16-packed weights; or simply eval-host noise (in which case the banked 8.205 was lucky).

Method: benchmark `_bldfp16` vs `_bldsv2` interleaved, ≥9 reps, **on an idle GPU** (check
`nvidia-smi` — another user runs on GPUs 1–2 and a first attempt at this measured a bogus
1.52× before I re-ran it clean). Also try the CPU path, which resembles the eval host far
more than our A800 does.

If it is a real weight-dependent effect, SV3 will inherit it and it is worth fixing. If it is
noise, we stop worrying. **~0.015 final either way — small, but it is a third of what the
data lever bought.**

---

## THE FIVE RULES (unchanged — each from a bug that produced a plausible WRONG table)

1. **Print split sizes and ASSERT your baseline, aborting on mismatch.** Honest `re_lohi`
   baselines: 664-window `rel 95.3069 / tke 78.7002 / mvpe 96.0540`; 900-window
   `95.4680 / 80.8027 / 96.3807`. ⚠️ **These are LEAKY for model-vs-model work** (the shipped
   soup saw ~80% of `re_lohi`) — valid only for correction-vs-correction. For model
   comparisons use a checkpoint that genuinely held out the eval conditions.
2. **NEVER `.float()` an FNO/bounds state dict** — spectral weights are `complex64` and
   `.float()` silently discards the imaginary part (once read `rel 86.78` vs `95.47`).
3. **Index `RE` via `wt`** (per-window), never over `names` (per-trajectory).
4. **`ps -eo args | grep -c "[x]pattern"`, never `pgrep -f x`** (it matches your own command).
5. **Never scratch to `/tmp`/root** (47 GB, ~90% full — it once truncated a 201 MB checkpoint
   and the archive still passed `unzip -t`). Never patch through an ssh heredoc; `rsync` it.

Also: `CUDA_VISIBLE_DEVICES=0` is set in the shell profile ⇒ launch as
`CUDA_VISIBLE_DEVICES=$i ... --gpu 0`, or you get "invalid device ordinal".
And use `mmap_mode="r"` on the big caches — `cache_*_stride3.npz` is 23.4 GB and OOM-killed a
job already.

---

## STILL CLOSED — do not spend time here

Bounds/E (4 live failures; policy space proved **4-DOF complete**: location, scale, shape,
support — all measured, `W_s` provably unexploitable at +0.00005 sps) · all prediction
rescaling (per-sample oracle only +0.021) · post-hoc residual correctors **as a family**
(§61: information-limited — two DISJOINT feature sets converge on the same ~5% ceiling and
combining them gains nothing) · architectures (CNO/Transolver worse; F-FNO ties the honest
soup; cross-arch ensembling negative; DMD/Koopman negative) · sim pretraining (only 81 of 100
sim files usable) · time efficiency tuning (fp16/CUDA-graphs/grouped-convs/`_BATCH` all ≈0
live — **work removal ~50%, efficiency tuning ~5%**) · tke amplification (global and
per-location) · target quantisation.

---

## Report format

| Config ID | rel_l2 | tke | mvpe | sps | time (ms) | Total | Key levers |

With, for each: how the total was estimated, the falsification test run, and — for anything
measured on a split the model trained on — **which transfer rate you applied and why**
(27.6% leaky vs ~90% honest).

**An honest "no lever found" remains a valuable result.** Two investigations returned empty
tables today and both were among the most useful work done — one proved `W_s` unexploitable,
the other found the leak that reframes the whole record.
