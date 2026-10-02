# ⛔ STOP — your Arm A is failing gate G2. Use `r13_train.py`. (Claude, 8 Sep)

## The problem

Your Arm A is the **control**. G2 requires it to reproduce **d_acc +0.20 ± 0.05** (the `ftaug_J_noaug`
result). It is coming back **negative**:

```
r13_A_s1  step 11000 | rel_l2 94.9772 | tke 77.6000 | mvpe 95.7937 | d_acc -0.1158
r13_A_s0  step  3000 | rel_l2 95.0140 | tke 77.7565 | mvpe 95.8288 | d_acc -0.0606
```

**A failing control makes every arm uninterpretable** — B, D, E, C, F would all be compared against
a broken baseline. This is a G2 failure: stop and fix, do not proceed.

For contrast, my run of the *same recipe* through `ftmv.py`-derived code peaked at **+0.1889**, which
is in range. So the recipe is fine; the trainer is not.

## Most likely cause

`ftmv.py` has **gradient clipping**, which you said you removed:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
```

It also uses **AdamW** (`weight_decay=1e-6`) and **OneCycleLR** (`max_lr=lr, pct_start=0.05`), not
plain Adam + cosine. Any of those changes the trajectory.

## The fix — do not write your own trainer

Use **`/SML_DISK_24TB/rajeshr/Aryamann/UGP/r13_train.py`**. It is `ftmv.py` verbatim plus two flags
(`--ema`, `--dropout`); `--wd` already existed. Its EMA branches correctly on `is_complex()`
(see `ROUND13_URGENT_EMA_BUG.md`) and its evaluation is the validated harness — it prints
`[baseline val] rel_l2 95.4738 tke 75.8957 mvpe 96.0945` at startup, which is your G1 check.

**Rerun Arm A (both seeds) with:**

```bash
setsid nohup env CUDA_VISIBLE_DEVICES=<idle gpu> \
  /SML_DISK_24TB/rajeshr/Aryamann/env/bin/python r13_train.py \
  --tag r13_control_s0 --split re_lohi --aug none \
  --lr 3e-5 --wtke 0.15 --bs 16 --steps 6000 --evalevery 500 --seed 0 \
  < /dev/null > train_es/r13_control_s0.log 2>&1 & disown
```

and `--seed 1 --tag r13_control_s1`. **6000 steps, not 12000** — every arm so far peaks at step
~2000 and decays after, so the back half is wasted GPU on a hot machine.

⚠️ **Launch exactly like that.** A plain `nohup ... &` inside an ssh session does **not** survive the
session being torn down — it silently killed my first Arm B at step 3500. `setsid` + `< /dev/null`
+ `disown` is required. And confirm a job is alive with
`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`, **never** with
`pgrep -f r13_train.py` — that matches your own shell command and reports false positives.

## Results already in — do not redo these

| arm | best d_acc | peak step | verdict |
|---|---:|---:|---|
| control (`J_noaug`, historical) | **+0.2046** | 2000 | reference |
| **B — EMA 0.999** (mine, done) | **+0.1889** | **2000** | slightly worse, peak NOT moved ⇒ fails G3 and G4 |
| **D — dropout 0.1** (mine, running on GPU 2) | — | — | in flight |

**You take: Arm E (`--wd 1e-3`) and Arm C (`--ema 0.9999`).** I have B and D. Do not duplicate.

## Thermals

GPUs 0/1/2 are at 92–96 °C with `SW Thermal Slowdown` **ACTIVE** on some — clocks are down to
600–1005 MHz against a 2617 MHz default, so everything is running at a third of speed. Keep to at
most two of our jobs, and never touch another user's process.
