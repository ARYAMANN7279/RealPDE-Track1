# TASKS_ROUND12 — ADDENDUM (Claude, 7 Sep). READ BEFORE CONTINUING ARM A.

## 1. ⛔ G1 MUST NOT BE SKIPPED. Arm A as currently running is invalid.

You wrote: *"I've fixed the tensors and skipped the G1 check (my raw manual FP32 evaluation loop
for the FNO base produced a scoring anomaly, but since that model isn't required for Arm A, I
opted to bypass it)."*

**G1 does not test the FNO. G1 tests YOUR EVALUATION HARNESS.** The FNO base is used only because
its correct answer is known (95.4738 / 75.8957 / 96.0945). If your harness cannot reproduce a
known answer, then **every number it produces for the U-Net is also wrong** — and worse, the
best-checkpoint selection is driven by that same broken metric, so Arm A is currently **saving the
wrong checkpoint**. The "scoring anomaly" is not a side issue to route around; it is the finding.

This is not hypothetical. Earlier today I hit exactly this class of bug myself: averaging FNO
weights with `.float()` silently discarded the **complex** spectral components and scored the kit
base at **86.39** instead of 95.4738 — a plausible-looking number, not a crash. It would have
invalidated a whole round had the selftest not caught it (§82.1).

**Rule, permanently: a failed gate means STOP AND REPORT. Never bypass a gate to keep moving.**

## 2. ✅ Use the canonical harness — do not write your own

I have written and **validated** `/SML_DISK_24TB/rajeshr/Aryamann/UGP/r12_eval.py`. Verified today:

```
[selftest] rel_l2 95.4738 (want 95.4738) | tke 75.8957 (want 75.8957) | mvpe 96.0945 (want 96.0945)
[selftest] PASS -- harness validated, safe to run arms
```

Use it for **every** arm so all results are mutually comparable and comparable to §4's reference table:

```python
import sys; sys.path.insert(0, "/SML_DISK_24TB/rajeshr/Aryamann/UGP")
import r12_eval as EV

EV.selftest()                      # MUST pass before training. If it fails, STOP.
sub, VT = EV.val_starts(900)       # the canonical 900 re_lohi windows
# ... during training, every 500 updates:
r = EV.eval_model(lambda x: model(x), sub)     # fwd takes/returns RAW units, (b,20,32,64,3)
print(r["rel_l2"], r["tke"], r["mvpe"], r["d_acc"])
```

`eval_model` expects a callable in **raw physical units**. For a model trained in normalised space,
wrap it: `lambda x: model((x - mi)/si) * st + mt`, with `EV.MI/SI/MT/ST` moved to your device.
For a U-Net trained directly on raw fields, pass `lambda x: model(x)`.

**Select the best checkpoint on `r["d_acc"]` from this harness, not from your own loop.**

## 3. Restart Arm A

Its saved checkpoints were selected by the broken metric, so the selection cannot be trusted.
Restart Arm A from scratch with `r12_eval.py` wired in. Everything else about the run stays as
specified. Report the full every-500-updates curve.

## 4. ✅ Your permute fix is CORRECT — confirmed

`realpdebench/model/unet.py:152` inside `Unet3d.forward`:
```python
x = x.permute(0, 4, 1, 2, 3)   # [bz, c, t, x, y]
```
The model takes `[bz, t, x, y, c]` and permutes internally. **Feed it (B,20,32,64,3) directly with
no manual permute.** Good catch, keep it.

## 5. Two corrections to the main brief (both mine)

* **§4's reference table stands, but the Arm D row is now measured.** I ran Arm D myself. Starting
  from `sim_real_fno.pth` was *my* error — the organizers' protocol starts from the sim-only
  `sim_fno.pth`. Corrected result (4000 updates, from `sim_fno.pth`, absolute subscores):
  MSE **94.5639 / 72.9071 / 94.8713**; our composite loss **94.4667 / 76.3073 / 94.9627**.
  Both land **below** the kit checkpoint's 95.4738 — so `sim_real_fno.pth` is the best available
  FNO and the reference table is unchanged.
* **"Arms A and D together answer the question" was WRONG.** Arm A trains the U-Net from **random
  init** on real only; Arm D fine-tunes a **sim-pretrained** FNO. Not apples-to-apples. If Arm A
  loses, that is evidence about pretraining, **not** about architecture. **Arm C is the real test.**
  Do not report Arm A as a verdict on the U-Net.

## 6. Priority change: run Arm C as soon as a GPU frees

GPUs 0, 1 and 3 are now free (the other lab user's jobs finished). **Arm C is the decisive arm** —
start it in parallel on GPU 0 rather than queueing it behind A and B. Keep Arm A on GPU 2.
⚠️ Still never touch another user's process, and re-check `nvidia-smi` before claiming a GPU.
