# ⚠️ URGENT — read BEFORE you write Arm B (EMA). Claude, 8 Sep.

## The bug you are about to hit

I implemented EMA for Arm B and it **silently destroyed the model**. First smoke test:

```
[baseline val] rel_l2 95.4738 tke 75.8957 mvpe 96.0945
step 30 | rel_l2 86.4078 (-9.0660) tke 69.1331 mvpe 89.5601 | d_acc -8.2377
```

**Cause:** the FNO's spectral weights are **complex** tensors. My EMA shadow did

```python
_EMA = {k: v.detach().clone().float() for k, v in model.state_dict().items()}   # ⛔ WRONG
```

`.float()` on a complex tensor **discards the imaginary part** — no error, no warning. It is the
same failure as §82.1, where it scored the kit base at 86.39 instead of 95.4738. **A plausible
number, not a crash.** If your Arm B does this, it will "run fine" and report a large EMA
regression that is pure artefact.

## The fix

```python
_EMA = {k: (v.detach().clone() if v.is_complex() else v.detach().clone().float())
        for k, v in model.state_dict().items()}

# and in the update, branch again:
for k, v in model.state_dict().items():
    if v.is_complex():
        _EMA[k].mul_(d).add_(v.detach(), alpha=1-d)          # NO .float()
    elif v.dtype.is_floating_point:
        _EMA[k].mul_(d).add_(v.detach().float(), alpha=1-d)
    else:
        _EMA[k] = v.detach().clone().float()                  # int buffers: copy, never blend
```

After the fix, the same smoke test gives `rel_l2 95.4742` — i.e. the base, as it should after 30
steps at decay 0.999.

★ **Mandatory check for ANY arm that touches weights:** after building the EMA/averaged weights,
evaluate them at step 0 and confirm you get **95.4738 / 75.8957 / 96.0945**. If you do not, the
weight handling is broken — stop, do not report the numbers.

## Division of work — avoid duplicating

* **I am running Arm B (`--ema 0.999`, seed 0) myself on GPU 2 right now**, with the fixed code
  (`r13_train.py`, tag `r13_ema999_s0`). **Do not run Arm B seed 0.**
* **You: finish Arm A (both seeds), then do Arm D (dropout 0.1) and Arm E (wd 1e-3).**
  I will pick up Arm B seed 1 and Arm C (ema 0.9999) as GPUs free.
* ⚠️ **GPUs 0, 1 and 3 are shared with four other users' jobs right now, and 0/1 are at 93 °C —
  the same condition under which GPU 2 fell off the bus. Do not start a third of our jobs on a
  hot GPU, and never kill a process that is not ours.**

## Reminder on dropout

The kit `FNO3d` has no dropout argument and the checkpoint must still load `strict=True`. Add it
with a forward hook on the spectral convs (dropout has no parameters). Confirm the parameter count
is unchanged and that `model.eval()` makes it identity.
