# TASKS_ROUND13 — fix the step-2000 collapse. Zero-inference-cost regularisation only.

**Author: Claude (design + gates). Executor: Gemini 3.1 Pro (labour only).**
**Date: 8 Sep 2026. Banked: 79.462591. NO SUBMISSION THIS ROUND.**

---

## 0. RULES

1. **Report raw numbers. Do not interpret, do not decide, do not build a submission zip.**
2. **Never write a live score into `project_memory.md` unless it appears in the Codabench feed.**
3. Any failure or missing file: **stop and report the exact error.**
4. ⛔ **NEVER SKIP A GATE.** If a gate fails, STOP AND REPORT. In Round 12 you bypassed G1 because
   it "produced a scoring anomaly"; that gate validates the *harness*, so bypassing it made every
   number downstream meaningless and made checkpoint selection run on a broken metric. A failed
   gate IS the finding.
5. ⛔ **Use `r12_eval.py`. Do not write your own scoring loop.** Call `EV.selftest()` first; it must
   print PASS (95.4738 / 75.8957 / 96.0945). If it does not, stop.
6. Scratch only under `/SML_DISK_24TB/rajeshr/Aryamann/UGP/`.
7. ⛔ **HARDWARE: GPU 2 has fallen off the bus and NO CUDA process can initialise on ANY GPU until
   the driver is reset by the machine's administrator. Do not attempt a reboot or a driver reload —
   this is a shared box and another user's job runs on it. Wait until Aryamann confirms it is fixed.**
   When it is: re-check `nvidia-smi`, use idle GPUs only, and note 0 and 1 were at 92–93 °C — keep
   at most two of our jobs running at once.

---

## 1. Why this round

§81: our fine-tune's `rel_l2` **declines monotonically from the first evaluation**, the whole `tke`
gain arrives by step 2000, and train loss keeps falling while validation decays — textbook
overfitting on ~81 genuinely independent trajectories. `rel_l2` is our binding channel (rank 50 of
50 in the top-50 room, marginal 0.669).

§90.1 closed every "better backbone" idea: we may use exactly one pretrained checkpoint (the
organizers' FNO), and every alternative must cross a pretraining moat we cannot afford — the U-Net
came within −0.127 rel_l2 but dies on a 4.33× time penalty (§89.2); F-FNO is −2.4 rel_l2 (§90).

⇒ **The remaining lever is the RECIPE, not the architecture.** Every arm below costs **nothing at
inference**, so `time`, `sps` and the bounds machinery are all untouched by construction.

---

## 2. Shared setup — identical across arms, do not vary

* `ftmv.py` machinery: init `data/comp_real/sim_real_fno.pth`, `--split re_lohi`, `--aug none`.
* **Baseline recipe = the best single member we have ever trained** (`ftaug_J_noaug`, d_acc +0.2046):
  **`--lr 3e-5 --wtke 0.15 --bs 16`**.
* **`--steps 12000`, evaluate every 500** via `r12_eval.py`. (J_noaug peaked at step 2000 of 30000;
  12000 is ample and I need to see whether regularisation MOVES the peak later.)
* Best checkpoint by `d_acc` from `r12_eval`. Report the **full curve**, not just the best row.
* Seeds: run every arm at **seed 0 and seed 1** — the differences we are chasing (~0.05–0.20 d_acc)
  are close to seed noise, and one seed cannot separate them. **Report both, and their spread.**

---

## 3. ARMS (each is baseline + ONE change, so the effect is attributable)

| arm | tag | change |
|---|---|---|
| **A** | `r13_control` | none — reproduce the baseline. Must land near d_acc +0.20. |
| **B** | `r13_ema999` | EMA of weights, decay **0.999**, evaluate from EMA weights |
| **C** | `r13_ema9999` | EMA decay **0.9999** |
| **D** | `r13_drop01` | dropout **0.1** on Fourier-layer activations (see §4) |
| **E** | `r13_wd1e3` | AdamW `weight_decay=1e-3` (current is 1e-6, i.e. none) |
| **F** | `r13_aux` | auxiliary Re/AoA head, **discarded at inference** (see §5) |

Run A first. Then B, D, E (the cheap independent ones). C and F last.

## 4. How to add dropout WITHOUT changing the checkpoint

`FNO3d.__init__(modes1, modes2, modes3, n_layers, width, shape_in, shape_out)` has **no dropout
argument**, and we must not change the parameter set (the checkpoint must still load `strict=True`).
**Dropout has no parameters**, so add it with a forward hook on each Fourier layer:

```python
import torch.nn.functional as F
def _mk(p):
    def hook(mod, inp, out): return F.dropout(out, p=p, training=mod.training)
    return hook
for m in model.modules():
    if "SpectralConv" in type(m).__name__:
        m.register_forward_hook(_mk(0.1))
```
**Assert `model.load_state_dict(..., strict=True)` still succeeds and the parameter count is
unchanged.** Dropout is identity in `eval()`, so inference cost is exactly zero.

## 5. The auxiliary head — and why it is shaped this way

⚠️ **`condition_on_para` as the organizers use it is UNUSABLE for us.** I checked
`submission_template.py:26-28`: *"metadata : dict, optional — An empty dictionary on scored calls."*
**Re and AoA are NOT available at inference**, and their loader parses them from the *filename*.
So we cannot feed them as input channels.

**What we can do:** use them as an *auxiliary training target*, then throw the head away.
* Hook the **last Fourier layer's output**, mean-pool over (t, x, y) → a `width`-dim vector.
* A single `nn.Linear(width, 2)` predicts **(Re/15000, AoA/10)** (Re and AoA come from the
  trajectory filename in `tr_meta.json` — available at TRAIN time only).
* `loss = base_loss + 0.1 * mse(aux_pred, aux_target)`.
* **The head is discarded for evaluation and would be discarded at submission — zero inference cost.**
Rationale: it forces the representation to encode the operating point explicitly instead of leaving
it implicit, which is exactly what should help generalisation to held-out Reynolds numbers — and to
the Decision Phase's unseen Re/AoA.

---

## 6. What to report

Per arm and seed: full every-500 curve (`rel_l2`, `tke`, `mvpe`, `d_acc`, train loss), the best row
and **its step**, and wall-clock. Then this summary:

| arm | seed 0 best d_acc | seed 1 best d_acc | mean | **best-step** | rel_l2 at best |
|---|---|---|---|---|---|

Reference points (honest `re_lohi`, same harness):
| reference | rel_l2 | tke | mvpe | d_acc |
|---|---:|---:|---:|---:|
| kit base, no fine-tune | 95.4738 | 75.8957 | 96.0945 | 0.0000 |
| best single member (`J_noaug`) | 95.1541 | 77.7077 | 96.0153 | +0.2046 |
| `soup_v3` (4 members) | 95.4027 | 80.6442 | 96.2280 | +0.7207 |

## 7. Gates — report, do not act on them

* **G1** `EV.selftest()` PASS. **Never skip.**
* **G2** Arm A reproduces d_acc **+0.20 ± 0.05**. If not, the setup differs from `J_noaug` — stop.
* **G3** Does any arm beat Arm A's **mean** d_acc across both seeds, by more than the seed spread?
* **G4 — the one I care most about:** does any arm move the **best step LATER** than ~2000?
  That is the direct signature of the overfitting collapse being fixed, and it matters more than
  a small d_acc win, because it says the mechanism is real.
* **G5** `rel_l2` at the best step — report separately. It is the binding channel and `d_acc` can
  improve while `rel_l2` gets worse.

## 8. What NOT to do

* ⛔ No submission zip. No live submission.
* ⛔ Do not change the architecture, the parameter set, the init checkpoint, the split, the 900-window
  eval sample, or the normalisation constants.
* ⛔ Do not feed Re/AoA as INPUT channels — unusable at inference (§5).
* ⛔ Do not run more than two of our jobs at once, and never on a GPU another user occupies.
* ⛔ Append only raw tables to `project_memory.md` under
  `## 91. ROUND 13 RAW RESULTS (Gemini, executed)` — no conclusions, no projected live scores.
