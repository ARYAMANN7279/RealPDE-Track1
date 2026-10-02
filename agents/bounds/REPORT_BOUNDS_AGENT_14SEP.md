# BOUNDS agent final report (14 Sep 2026) — filed by the coordinator
The harness refused the agent's REPORT.md writes ("subagents should return findings as text"). This is its report,
condensed faithfully. `agents/bounds/REPORT.md` dated 12 Sep 10:33 is NOT this agent's and contains two errors:
(1) it says centre shifting gave negligible gains — centre shifting is the ONLY bounds change that ever gained live
(E 0.502 → 0.548); (2) it says further gains need jointly-trained uncertainty — a perfect scale predictor is worth at
most ~+0.019 local E.

## Live-E model (6 variants, fit on 9 live submissions sharing backbone dc935158, 7 bounds policies, leave-one-policy-out)
* LOO miss on the live E LEVEL: 0.018 E (0.30 final).
* On the three shipped WIDTH changes (LUTCAL−FP16, WIDE125−SOUP_v1, LUTFIX−SOUP_v1) the models' difference errors run
  −0.008 … +0.040 E; **17 of 18 positive** ⇒ systematically over-predict the benefit of width changes.
* The unchanged policy's own band is 0.048 E wide (0.81 final) — model uncertainty exceeds any effect sought.

## Width-scale grid on SCREEN/SPEED_SAFE bounds (E_live 0.5503; 17.05 final per unit E)
| k_u × k_v | local ΔE | model point preds | LOO-corrected range | Δfinal |
|---|---:|---|---|---|
| 0.8 × 1.0 | +0.0205 (kit sps 52.92→54.52) | −0.011 … +0.005 | [−0.043, +0.002] | [−0.73, +0.03] |
| 0.7 × 1.0 | +0.0284 | −0.022 … +0.003 | [−0.050, −0.001] | [−0.85, −0.01] |
| 1.0 × 1.0 | 0 | 0 | [−0.040, +0.008] | [−0.67, +0.14] |
| 1.0 × 1.2 | −0.0111 | +0.007 … +0.016 | [−0.024, +0.016] | [−0.40, +0.28] |
| 1.0 × 1.35 | −0.0212 | +0.006 … +0.022 | [−0.019, +0.016] | [−0.33, +0.28] |
* Models disagree on the sign of tightening u; all favour widening v — locally negative, and the exact direction they
  got wrong on WIDE125 and LUTCAL.
* Per-bin LUT re-optimised on local residuals (half/half split; u×0.56, v×0.84): local +0.039, every model predicts
  −0.024 … −0.060 live — the LUTCAL/ARCSINH trap reproduced.
* Per-bin LUTs re-optimised under the live models widen v 1.25–1.50 and are locally negative; the exception
  (uncoverable-fraction model, u×0.81 v×1.25): local +0.006, live [−0.026, +0.009].

## Why LUTCAL lost
Centres identical to TMEAN's (3e-8); only widths changed (median u×0.885, v×1.425). Live ΔE −0.0055 (−0.093 final).
Local: global scaling −0.0142, u part +0.0123, v part −0.0243, full −0.0119. **The local monitor had the right sign
(~0.46× live).** Every model predicts a GAIN (+0.0015 … +0.023), even with LUTCAL inside its fit; the error sits in the
v widening. Scaling vs 64-bin reshape is separable (scaling explains the change to 0.003 E) but the scaling response
itself is wrong. ⇒ **The model is NOT validated for width decisions.** It reproduces the live E level by inflating
residuals ~2×, which automatically rewards widening; live results don't. Our live E deficit is not an inflation of
local residuals.

## Proposal: NONE
No candidate predicts ΔE_live > +0.005 with its band above that; the ones the models like (wider v) are locally
negative.

## Analytic results
* Optimal interval [c−a, c+b]: density at both endpoints = coverage/σ. Gaussian h*/σ = 0.10/0.17/0.27/0.47/0.50 at
  s/σ = 0.05/0.1/0.2/1/5 (reward 0.78/0.65/0.48/0.14/0.03). Laplace: h* = b·ln(1+σ/2b).
* No give-up regime: zero width never optimal. As error scale grows h* → σ/2 = 0.028; our LUT tops out at 0.0274.
* Mis-scaling is second-order: 30% off costs ~3.5%, 2× costs ~10%. Asymmetry = a centre shift; ≤0.2% at realistic skew.
* Kit self-test reproduces local sps 50.8060 vs 50.8059 recorded. Only Gaussian-within-element errors fit the data.
| channel | best constant | best LUT on our feature | perfect scale predictor | knowing actual error |
|---|---:|---:|---:|---:|
| u | 0.595 | 0.644 | 0.662 | 0.834 |
| v | 0.746 | 0.792 | 0.803 | 0.928 |
Our feature captures ρ ≈ 0.91–0.95 of a perfect scale predictor. §69/§71's "0.85 ceiling" assumes knowing the actual
error, which no uncertainty predictor can.
* Live map: no shipped policy ever reached E 0.58. Current bounds 0.5489–0.5510 across all six backbones. Top-60 board:
  **E ≈ −0.694 + 1.806·W (r = 0.60); we sit exactly on the line**; zyangastar gets 0.583 at nearly our W.
* Caches: 12 GB of per-submission bound caches on the VM (`agents/bounds/cache734`), reusable without GPU.
**Analytic E ceiling:** perfect scale predictor +0.011–0.019 local E/channel ⇒ ~+0.012–0.015 E_live ⇒ ≲ +0.25 final.
**Best live E ever:** 0.55105 (ENSEMBLE_FAST_v3); banked 0.5503.
