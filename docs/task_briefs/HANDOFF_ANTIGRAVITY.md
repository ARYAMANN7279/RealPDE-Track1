# Handoff — RealPDE Track 1, parallel work

**Banked: 79.378881.** Do not submit anything; a human spends the one daily slot (resets
05:30 IST). A FAILED submission still consumes it.

```
final = 0.46743*rel_l2 + 0.10027*tke + 0.09420*mvpe + 0.09689*time + 0.24737*sps + 0.9119
```
Current subscores: rel_l2 94.074271 · tke 75.998930 · mvpe 93.124176 · time 90.408319 · sps 37.815686
Marginal value per point (incl. the W→sps channel): **rel_l2 0.669 · mvpe 0.170 · tke 0.157 · time 0.097**

Machine: `ssh vm` (key auth, no password). Work dir `/SML_DISK_24TB/rajeshr/Aryamann/UGP`.
Full record: `~/Desktop/sem7/UGP/project_memory.md` (~5970 lines). **Read §45–63 before proposing anything.**

---

## ⛔ FIVE RULES THAT ARE NOT NEGOTIABLE

These come from bugs that each produced a plausible, well-formatted, WRONG table. **None was
caught by the result looking suspicious.** All were caught by a cheap invariant.

1. **Every eval script must print its split sizes and ASSERT its baseline, aborting on mismatch.**
   Honest `re_lohi` baselines for the shipped soup:
   * 664-window cache: `rel 95.3069 / tke 78.7002 / mvpe 96.0540`
   * 900-window protocol: `rel 95.4680 / tke 80.8027 / mvpe 96.3807`
   * shipped-artifact E on the 664-window harvest: `0.66466`
   Mismatch ⇒ your harness is broken. STOP. Do not "explain" it.
2. **NEVER `.float()` an FNO or bounds state dict.** Spectral weights are `complex64`; `.float()`
   silently discards the imaginary part (torch only warns). It once read `rel 86.78` instead of
   `95.47` and produced a fake "+0.088, no retraining". Use
   `v.to(torch.complex64) if v.is_complex() else v.float()`.
3. **Index `RE` via `wt` (per-window), never over `names` (per-trajectory).**
   `RE = np.array([int(names[i].split("_")[0]) for i in wt])`. The wrong form gives
   train 65 / val 8 and a fake `f = 0.151`.
4. **Check processes with `ps -eo args | grep -c "[x]pattern"`, never `pgrep -f x`** — the latter
   matches your own ssh command string and silently reports work as running that is not.
5. **Never write scratch to `/tmp` or the root partition** (47 GB, ~90% full). It truncated a
   201 MB checkpoint mid-unzip and the archive still passed `unzip -t`. Use
   `/SML_DISK_24TB/rajeshr/Aryamann/UGP/`. Also: never patch a file through an ssh heredoc —
   quoting gets eaten; write the patch locally and `rsync` it.

**Two more calibration facts that will mislead you if you do not know them:**
* **Timing transfers BY CATEGORY.** Removing work (deleting a module, fewer passes) → ~50%.
  GPU-efficiency tuning (batch size, precision, kernels, fusion, vmap) → **~5%, treat as ZERO.**
  Measured live: `_BATCH=48` + fp16 measured 2.7%+3.7% locally and delivered 5% of projection.
* **The local harness overstates bound-width sensitivity by 4.9×.** Every "tighten the bounds"
  result is an illusion at that scale.

---

## TASK A — Rebuild the model soup with correct selection  ⚠️ DOWNGRADED — read this first

⛔ **The "27×" premise of this task was a LEAKAGE ARTIFACT.** `finetune.py:32` holds out
`every5`, not `re_lohi`, so every model-vs-model number measured on `re_lohi` is leaky.
Honestly, souping is worth **+0.055 tke over the mean member and −0.215 vs the best** — not
27×. Its real value is rel_l2/mvpe stability (d_acc +0.04…+0.09). **Do Task C first.**
⚠️ Every "honest re_lohi baseline" quoted in this document is LEAKY for model-vs-model work.
It remains valid for correction-vs-correction work (LUT, time-mean head). For model
comparisons use a checkpoint that genuinely held out the eval conditions — honest
counterparts are on the VM in `_tke/`.
★ One correction in your favour: the tke transfer rate is **~90%, not 46%** — an honestly
measured accuracy gain is worth ~2× what the older record implies.

**Why.** Souping is the strongest mechanism in this project: the shipped soup is worth
**+0.6526** vs **+0.0244** for the best single fine-tune — 27×. But the current soup's members
were each selected under **wrong marginal values**: `train_es/ftaug.py:173` used
`MV=dict(rel_l2=0.467, tke=0.208, mvpe=0.290)` where the truth is
`(0.669, 0.157, 0.170)` — tke overweighted 1.9×, mvpe 2.4×, both relative to rel_l2. And
`if d > best: save` **discarded every alternative**, so the better checkpoints are gone.
Separately, `wtke` was never swept below 0.15; the measured optimum is **0.08**.

⇒ Nobody has ever built a soup whose members were selected on the correct objective.

**Do.**
1. Use `train_es/ftmv.py` — it already has the corrected MV. Verify line 97 reads
   `MV=dict(rel_l2=0.669,tke=0.157,mvpe=0.170)`.
2. Train **7 members** from `data/comp_real/sim_real_fno.pth`, honest `re_lohi` split:
   `--lr 1e-5 --wd 1e-6 --steps 16000 --wtke 0.08 --aug none`
   Vary for diversity: lr ∈ {1e-5, 3e-5}, wtke ∈ {0.05, 0.08, 0.10}, and if you add a `--seed`
   argument, different seeds (**the seed is hardcoded 1234 — identical config ⇒ bit-identical
   model, which wastes a GPU; this already happened once**).
3. Launch with `CUDA_VISIBLE_DEVICES=$i ... --gpu 0`. `CUDA_VISIBLE_DEVICES=0` is set in the
   shell profile, so `--gpu 1` dies with "invalid device ordinal".
4. Average the members (see `local_harness/average_soup_v2.py` for the correct complex-safe
   pattern) and evaluate the soup against the shipped one on **both** the 664- and 900-window
   protocols above.

**Report:** per-member and soup `rel_l2 / tke / mvpe`, `d_acc` under the CORRECT MV, and the
soup vs the shipped soup's `+0.6526`. **Do not build a submission zip.**

**Honest expectation:** the per-member wtke gain is +0.010…+0.036, and greedy-souping new
members into the *existing* soup was all-negative (16/16 mixes). A fresh correctly-selected
soup is a different experiment, but price it low.

---

## TASK B — Extend the 100%-data soup (do only AFTER `submission_SV2.zip` scores)

`submission_SV2.zip` (md5 `4c72fe5f7ad34e5c34419390e6498cb2`) puts the **100%-data**
checkpoint `local_harness/soup_v2.pth` into the current machinery. It is a blind bet: soup_v2
trained on **all 81 trajectories** (`finetune_all.py` has no `evaluate()`, no holdout, no
checkpoint selection — 6000 fixed steps), so **no local validation of it is possible, ever.**

* If SV2 **gains**: train 4–6 MORE 100%-data members (vary lr/wtke), soup them, hand back the
  checkpoint. This is the only direction with a live-confirmed mechanism at that point.
* If SV2 **loses or ties**: STOP. The "more data helps" bet is then answered and closed.

Evidence so far (in-sample on held-out Re, therefore inflated by an unknown amount):
soup_v2 `rel 95.5065 / tke 81.9676 / mvpe 96.2958` vs soup_v1 `95.3069 / 78.7002 / 96.0540`.
Notably soup_v2's **rms error is 0.5% LARGER**, which is the opposite of a memorisation
signature — memorising shrinks point error; what improved is the time-variance structure.

---

## TASK C — Price the smoke-test containment risk (cheap, ~1 hour, no GPU)

`starting_kit_v9/realpde_t1_starting_kit_v9/smoke_test_kit.py:280` asserts
`np.all(lo <= pred) and np.all(pred <= up)`. **`scoring.py` does NOT check this** — only shape,
finiteness and `lower <= upper` — which is why five live submissions scored normally with
off-centre bounds. Measured on the banked artifact: **1.264% of elements violate**
(u 2.037%, v 0.490%; median excess 22% of a half-width). Since the assert is `np.all`, it
**fails outright**.

Top-10 entries are **re-trained and re-verified by the organizers in the Decision Phase**, so
this is worth pricing now rather than discovering there.

**Do.** Build a variant that clips the centre shift so `lower <= prediction <= upper`
everywhere (i.e. clamp `alpha*c` to `±h` per element), then measure the **sps cost** on the
664-window honest split using the official `aggregate_sps`. Report ΔE and Δfinal.
If the cost is small (say < 0.05 final), we have a compliant fallback ready for the Decision
Phase at known price. **Do not submit it.**

---

## ⛔ DO NOT SPEND TIME ON THESE — all measured, all closed

* **Bounds/E channel** — cost FOUR live submissions (LUTFIX 78.34, WIDE125 78.30, arcsinh
  76.70, LUTCAL 79.28). The shipped LUT is **bracketed by live failures in both directions**.
  Closed: LUT refits, richer conditioning (trunk probe, frame index, 2-D bins, 64→512 bins),
  asymmetric/highest-density intervals, per-channel fit-g, ensemble weights `cw` (closed by
  algebra), centre strength `alpha` (flat top 0.85–0.95), support truncation (+0.112 vs a
  +0.40 bar), and `W_s` exploitation (**provably separable ⇒ worth +0.00005 sps**).
* **All prediction rescaling** — global γ (optimum 1.0008), per-frame, per-channel,
  per-frame×channel, per-sample (ORACLE only +0.021).
* **Post-hoc residual correctors, as a FAMILY** — capped at ~5% of oracle, and §61 proved it is
  an **information limit, not a feature-engineering shortfall**: two disjoint feature sets
  (288 U-Net trunk channels; 12 input-window temporal features) converge on the same ceiling
  and combining them gains **nothing**. No feature set rescues this family.
* **Architectures** — CNO-FT, Transolver-FT worse; cross-arch ensembling negative; DMD/Koopman
  negative; fused F-FNO at FULL temporal bandwidth, sim-pretrained to convergence then
  fine-tuned, reaches KIT parity (+0.067) but is **−0.736 vs our soup**.
* **Sim pretraining** — the organizers' checkpoint is already baked in; sim-only init −0.118;
  sim mixing ≈0. Also: only **81 of 100** sim files are usable (`mksim.py` moment-matches each
  to its MATCHING real trajectory; the other 19 have no counterpart).
* **Time efficiency tuning** — fp16 (shipped, ≈0 live), CUDA graphs (bit-exact, ≈0.0003),
  grouped convs (0.959×, not bit-identical), `_BATCH` (48 best, ≈0 live).
* **tke amplification** — global (best α=1.10 → +0.145; "pattern, not magnitude") and
  per-location (largest oracle in the project, does not transfer).
* **Target quantisation** — ~5e-08, 6e4× finer than needed. No atoms, no lattice.

---

## Output format

For anything you propose, report:

| Config ID | rel_l2 | tke | mvpe | sps | time (ms) | Total | Key levers |

plus, for each: how the total was estimated, the falsification test you ran, and — for
anything touching bounds or a local optimum — **why it would TRANSFER**. Given four live
failures on that channel, a local optimum is near-worthless as evidence.

**An honest "no lever found, here is the evidence" is a valuable result.** The sps
investigation returned an empty table and was still the most useful work of the day, because
it proved `W_s` unexploitable and caught a wrong number in the record. Do not pad.
