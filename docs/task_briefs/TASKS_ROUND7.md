# Execution Brief — Round 7: Refund the Time Cost

`submission_TMEAN.zip` scored **79.369466** — a PR, and the banked score. But it paid a
time penalty I failed to price, and **that penalty is recoverable without giving up any of
the gain.** This round is about collecting the refund.

**Your role is execution. Do not submit. Do not mark anything CLOSED.** Where a judgement
call is needed, STOP and report.

---

## 0. What the live result taught us — read this first

| subscore | banked | TMEAN | delta |
|---|---:|---:|---:|
| rel_l2 | 94.045224 | 94.072631 | +0.027407 |
| **tke** | 75.998641 | **75.998641** | **+0.000000** |
| mvpe | 92.873514 | 93.141584 | **+0.268070** |
| **time** | 90.662719 | 90.334470 | **−0.328249** |
| sps | 37.721250 | 37.803117 | +0.081867 |
| **final** | 79.341788 | **79.369466** | **+0.027678** |

**Three calibration facts to carry forward:**

1. **A time-constant correction leaves tke EXACTLY invariant on the live board.** Not
   "approximately" — `+0.000000` to six decimals. The §35 algebra holds on real data. Any
   future correction that is constant along the time axis is *provably* tke-safe.
2. **mvpe transfers at 1.83×** (local +0.1464 → real +0.2681). It is the only lever in this
   project that transfers at better than 1:1, because mvpe *is* the time-averaged probe
   profile. **Never discount an mvpe gain.**
3. **The accuracy gains were fine; the time bill ate 55% of them.** Gross +0.0583, time
   −0.0318, net +0.0265 (actual +0.0277). This round removes the −0.0318.

---

## 1. ★ Where the 0.614 ms went — and it is pure redundancy

Live time went 7.732 → 8.345 ms/sample. Look at `_bld/submission.py` lines ~267–288. The
bounds loop builds `ui` and runs the nets:

```python
for k in range(0, j - i, CH):
    m = min(k + CH, j - i)
    ui = torch.cat([_fold(xb_raw[k:m, :, :, :, :2]),
                    _fold(yb[k:m, :, :, :, :2])], 1)
    o = net(ui)                                  # base bounds net
    ...
    for _e, _wgt in zip(_state["extra"], cw[1:]):
        c = c + _e(ui)[:, :40]...                # 2 ensemble extras
```

and then the TMEAN patch adds a **SECOND COMPLETE LOOP that rebuilds the identical `ui`**:

```python
if _state.get("mp") is not None and _state["mp_alpha"] != 0.0:
    for k in range(0, j - i, CH):                # <-- second full pass over the batch
        m = min(k + CH, j - i)
        ui = torch.cat([_fold(xb_raw[k:m, :, :, :, :2]),
                        _fold(yb[k:m, :, :, :, :2])], 1)   # <-- IDENTICAL tensor, rebuilt
        mc = _state["mp"](ui) * _state["mp_alpha"]
        yb[k:m, :, :, :, :2] += mc.permute(0, 2, 3, 1).unsqueeze(1)
```

Two separate costs: **(a)** `ui` is folded and concatenated twice per chunk, and **(b)** a
full W96 U-Net trunk runs that duplicates work `net` already did on the same input.

---

## ★ TASK A — TIER 1: merge the loops (free, bit-identical, do this first)

Move the `mp` call **inside** the existing `k` loop, reusing the `ui` already in hand.

### A1. Why this is provably safe
`mp` writes only to `yb[k:m]`, the same slice the loop just finished reading. Chunk `k+1`
builds its `ui` from `yb[k+1 slice]`, which `mp` never touched. **The chunk boundaries are
identical (`CH` in both loops), so there is no cross-chunk dependency and merging cannot
change a single output bit.**

Order within the merged loop matters — do it exactly like this:

```python
for k in range(0, j - i, CH):
    m = min(k + CH, j - i)
    ui = torch.cat([_fold(xb_raw[k:m, :, :, :, :2]),
                    _fold(yb[k:m, :, :, :, :2])], 1)     # built ONCE
    o = net(ui)
    ... existing centre / width / LUT / lo / up code, UNCHANGED ...
    lo[k:m, :, :, :, :2] = (ctr - h).permute(0, 1, 3, 4, 2)
    up[k:m, :, :, :, :2] = (ctr + h).permute(0, 1, 3, 4, 2)
    # --- only AFTER lo/up are written for this chunk ---
    if _state.get("mp") is not None and _state["mp_alpha"] != 0.0:
        mc = _state["mp"](ui) * _state["mp_alpha"]
        yb[k:m, :, :, :, :2] += mc.permute(0, 2, 3, 1).unsqueeze(1)
```

⛔ **`yb` must not be modified before `lo`/`up` are written for that chunk.** If you move
the `mp` line above the `ctr = ...` line, the bounds get built on the corrected prediction
and the geometry that scored 79.3418 is destroyed. The acceptance gate below catches this.

### A2. Acceptance — all three must hold vs `submission_TMEAN.zip`
* `max|Δprediction| = 0.000e+00`
* `max|Δlower| = 0.000e+00`
* `max|Δupper| = 0.000e+00`

This is a pure implementation change. **If any is non-zero, stop and report** — it means the
ordering is wrong, not that the numbers are "close enough".

### A3. Measure, don't assume
Report the local A/B ratio vs `TMEAN`, ≥5 interleaved reps, GPU **and** CPU paths. My
estimate is 8.345 → ~8.15 ms (**projected**, ≈ +0.010 final), but that is a guess about how
much of the 0.614 ms is `ui` construction versus the forward pass. **The split is the
deliverable here** — it tells us how much Tier 2 is worth.

---

## ★★ TASK B — TIER 2: a shared-trunk time-mean head (this is the real prize)

### B1. The observation
Compare `_MeanNet` (in `build_tmean.py`) with `_UNet` (`submission.py:93`). **The trunks are
identical** — same `_Blk` encoder, same bottleneck, same decoder, same interpolation. They
differ in exactly one line:

```python
self.out = nn.Conv2d(w, 80, 1)     # _UNet   -> 40 centre + 40 width
s.out    = nn.Conv2d(w, 2,  1)     # _MeanNet -> 2 time-mean channels
```

`net` already runs that trunk, on that same `ui`, one line earlier. **~99.9% of `mp`'s cost
is trunk we have already paid for.** A second 1×1 conv on the `(b, w, 32, 64)` decoder
output is ~0.4 MFLOP/sample — free next to a full U-Net.

### B2. What to build
Add a second head to the existing bounds net, and **freeze everything else**:

```python
class _UNet(nn.Module):
    def __init__(self, ci, co, w, mean_head=False):
        ...
        self.out  = nn.Conv2d(w, co, 1)
        self.mout = nn.Conv2d(w, 2, 1) if mean_head else None
    def forward(self, x):
        ... unchanged through d1 ...
        if self.mout is None:
            return self.out(d1)
        return self.out(d1), self.mout(d1)
```

**Train ONLY `mout`. The trunk and `out` stay frozen at their shipped weights.**

### B3. ★ Why freezing the trunk is the whole safety argument
A frozen trunk and a frozen `out` mean `o = net(ui)` returns **bit-identical** centre and
width tensors. Therefore `lower` and `upper` are **bit-identical**, and `sps` cannot move.
Only `prediction` changes. This is the same property that made `ENSEMBLE_FAST_v3` safe, and
it is why this is a low-risk round.

⛔ **Do not fine-tune the trunk jointly.** It would move the bounds, forfeit that guarantee,
and put the measured +0.0596 sps from the centre ensemble back at risk for no reason.

### B4. ⚠️ Do NOT confuse this with the closed §Round-6 §0 result
Round 6 established that taking `mean_t` of the **existing 40-channel centre output**
captures only **2–7%** of the oracle. **That does not close this route**, and the distinction
is the point of the task:

* Round 6 took `mean_t` of a head trained for the *instantaneous* residual under L1. The
  time-mean of an L1-optimal instantaneous predictor is not the optimal time-mean predictor.
* Here you train a **new head directly on the time-mean target**, reading the same features.

Note the floor this gives you: a `w → 2` linear map can represent the average of 20 rows of
the existing `out` weights exactly, so **the new head can reproduce the 2–7% baseline by
construction and can only improve on it.** The open question is how far above it goes.

### B5. Training
* **Input features:** the frozen trunk's `d1` on the same 80-channel `ui`. Precompute and
  cache `d1` once over the training set — the trunk is frozen, so this is a one-off cost and
  training the head afterwards is minutes, not hours.
* **Target:** `RS.mean(axis=1)` from `cache_lohihonest.npz`, shape `(N, 32, 64, 2)` —
  identical to the Round-6 Task A target. Same data, same split.
* **Loss:** L2 first (the target is a conditional mean), L1 as the check.
* **Split:** honest `re_lohi` (hold out Re 3750 / 5025 / 25425 / 26700). Confirm on `aoa15`.
* Sweep α ∈ {0.5, 0.75, 1.0, 1.25}. The dedicated net's best was α = 0.75.

### B6. ★ Report against THIS bar, in these exact terms
The dedicated W96 net produced **gross +0.0583 final** (rel_l2 + mvpe + sps) at a cost of
−0.0318 in time. **The shared head keeps the full +0.0318 refund regardless of quality**, so:

| capture vs the dedicated W96 net | Δfinal vs shipped TMEAN | projected final |
|---:|---:|---:|
| 40% | −0.0032 | 79.366 |
| **45%** | **≈ 0** | **79.369 ← break-even** |
| 50% | +0.0027 | 79.372 |
| 60% | +0.0085 | 79.378 |
| 75% | +0.0172 | 79.387 |
| **100%** | **+0.0318** | **79.401** |

**The bar is only 45%.** Report capture as
`f_shared / f_dedicated`, where `f = 1 − err_mvpe_corrected / err_mvpe_baseline` on the
honest split — the same definition as Round 6 §A3, so the two are directly comparable.
Report `f` for both nets measured by the **same script in the same run**; do not compare
against a number quoted from an earlier round.

**Also report `|Δtke|` for every variant. It must be ≤ 1e-5.** That is the safety invariant
and it is now confirmed live — anything larger means the correction is not time-constant and
there is a bug. Stop and report.

---

## TASK C — the CUDA-stream fan-out (§37.9), if B lands

Already scoped and measured at **2.02×, bit-identical** for the 3 bounds nets. Worth ~+0.02.
Stack it on top of B only after B's gates pass; do not bundle an unmeasured change with a
measured one.

---

## Decision rule for the next slot

| outcome | ship |
|---|---|
| A passes gates, B ≥ 45% capture | **A + B** (≈ 79.40, or ≈ 79.41 with C) |
| A passes gates, B < 45% capture | **A alone** — bit-identical, ≈ 79.379, a guaranteed PR |
| A fails its bit-identity gate | ship nothing; report the diff |

**Task A alone is already a PR over 79.369466 and cannot lose**, because every output bit is
unchanged and only the clock moves. That is the floor for this round. Task B is the upside.

---

## Rules

* **Do not submit.** A human spends the daily slot. **A FAILED submission still burns it.**
* **Do not mark anything CLOSED.**
* Verify an artifact **exists** and its **per-file sizes match the source** before reporting
  it done. Three rounds have had artifact-level errors (a truncated checkpoint, a file
  reported built that did not exist, a silently-constant bounds net).
* Run the zip's **own `predict()`** as the final gate. Structural checks pass on broken
  artifacts — `ENS_IMPROVED_v3` shipped constant bounds and only this caught it.
* Never write scratch to `/tmp` or the root partition — use `/SML_DISK_24TB/rajeshr/Aryamann/UGP/`.
  Check `df -h /` around any extract.
* Never put a password in a shell command; `ssh vm` uses key auth.
* Never `pkill -f <pattern>` where the pattern also matches your own ssh command string.
* Gates: validator 13/13, extracted < 268,435,456 B, zero `.pyc`, md5 on VM and Mac.
* **Halve every local timing ratio** before quoting a live estimate — apply the halving to
  the *ratio*, not to absolute milliseconds.
* Label every number **measured** or **projected**.
* ★ **New rule from this round: any module added to the inference path is charged against
  `time` before its value is quoted.** The threshold is ≈ 0.032 final per 0.6 ms. TMEAN's
  cost was measured in §41.3, quoted correctly there, and then dropped from the forecast —
  which is exactly the size of the miss.
