# TASK — recover the 30 UNKNOWN checkpoints by SUBSET-MATCHING (CPU only, mechanical)

**Author: Claude. Executor: Gemini 3.1 Pro.**
**Date: 10 Sep 2026. Banked: 79.484440. `$B` = `/SML_DISK_24TB/rajeshr/Aryamann/UGP`.**

## 0. Rules

1. ⛔ **DO NOT USE ANY GPU.** This is weight arithmetic on CPU. Use
   `/SML_DISK_24TB/rajeshr/Aryamann/env/bin/python`.
2. ⛔ Never kill a process that is not yours. Never reboot.
3. ⛔ Scratch only under `$B/scratch/`. **Never `/tmp`** (root partition ~90% full; it once
   truncated a checkpoint that still passed `unzip -t`).
4. **A match counts ONLY if it is bit-exact (`max|Δ| = 0.0`) or within 5e-4 with the fp16 reason
   stated.** Anything else is `NOT RECOVERED`. ⛔ **Do not guess. Do not report near-misses as
   findings.**

## 1. Why — and proof the method works

`soup_v2.pth` was `UNKNOWN` on every column of `CHECKPOINT_REGISTRY.md`, and it is **half of every
artifact we have ever banked**. Yesterday I recovered it bit-exactly by brute-force subset matching:

> **`soup_v2 = mean(ft_all_w15_lr1, ft_all_w15_lr3, ft_all_w33_lr1, ft_all_w33_lr3)`**
> **max|Δ| = 0.000000e+00.** All 15 proper subsets scored 1.4e-3 … 2.0e-2; only the full 4-way
> average is exactly zero. Later confirmed by finding `local_harness/average_soup_v2.py`.

**30 checkpoints remain UNKNOWN** (§118.3) — no sidecar, no recorded recipe. Many are almost
certainly soups of checkpoints we *do* know. Each one recovered turns an unusable file into a usable,
classifiable one. My working script is `$B/idsoup2.py` — read it first; you are generalising it.

## 2. ★ THE EFFICIENCY TRICK — do NOT average full checkpoints in the search loop

Each checkpoint is ~400 MB / 50 tensors. Averaging them for every candidate subset is far too slow.
**Search on a SIGNATURE, confirm on the full tensors.**

**2.1 Build the signature cache (once).**
For every `*.pth` larger than 3e8 bytes under `$B/train_es/`, `$B/local_harness/`,
`$B/data/comp_real/`:
* load it (handle the packed format: if the dict has `state_fp16`, unpack — `complex_keys` entries
  need `torch.view_as_complex`; see `load_sd` in `$B/r47_build.py`);
* take the tensor keys in **sorted order**, and from the **first 8 float (non-complex) tensors**
  extract elements at flat indices `0, 97, 193, 389, ...` — i.e. **use a FIXED index list of 64
  positions per tensor**, giving a **512-element float64 vector**;
* ⚠️ **skip any key containing `num_batches_tracked`** — those are BN counters stored as `float32`
  and they are copied, not averaged (this exact issue produced a spurious `max|Δ| = 2.0` yesterday);
* save all signatures to `$B/scratch/sigs.npz` keyed by file path, plus each file's key-set hash.

**2.2 Search.** For a target `T` with signature `t`, a uniform mean over subset `S` of size `n`
satisfies `sum(sig(S)) = n * t`. Candidate pool = every OTHER checkpoint with the **identical
key-set hash**. Then:
* `n = 2`: all pairs (≈8k) — trivial.
* `n = 3`: all triples — vectorise with numpy.
* `n = 4, 5, 6`: **restrict the pool to files whose mtime is within ±3 days of the target**
  (usually ≤15 candidates), then enumerate all subsets of that restricted pool.
* Accept a signature hit when `max|sum(sig(S)) - n*t| < 1e-5`.

**2.3 Confirm.** For every signature hit, load the full checkpoints, average them properly
(**complex tensors must be averaged as complex — never call `.float()` on them, it silently drops the
imaginary part**), and report `max|Δ|` against the target across all non-counter keys.

## 3. ★ SELF-TEST — run this FIRST and report it before anything else

Point the pipeline at `local_harness/soup_v2.pth`. **It must recover exactly
`ft_all_w15_lr1_final + ft_all_w15_lr3_final + ft_all_w33_lr1_final + ft_all_w33_lr3_final` with
`max|Δ| = 0.0`.** If it does not, the signature or the averaging is wrong — **STOP AND REPORT.**
Do not run the survey on a pipeline that fails its self-test. (§13 of the Round-13 brief: a failed
gate IS the finding; do not bypass it.)

## 4. The survey

Run §2 for every checkpoint currently marked `UNKNOWN` in `$B/CHECKPOINT_REGISTRY.md`. Produce
**`$B/PROVENANCE_RECOVERED.md`**:

| target | recovered? | n | ingredients | max abs delta |

Then update `CHECKPOINT_REGISTRY.md`: for each recovered target, fill `init ROOT` with
`MEAN(<ingredients>)` and set `LEAKAGE` to the **most leaky** class among its ingredients
(`ALL-DATA` > `EVERY5`/`EVERY9` > `INIT-LEAK` > `HONEST`), since a soup inherits the worst leakage
of any member.

## 5. Report back

* The self-test result (§3), first.
* Counts: how many of the 30 recovered, how many still `NOT RECOVERED`.
* The table from §4.
* For the recovered ones, the new `LEAKAGE` counts.

⛔ Do not draw conclusions about which checkpoint is better. ⛔ Do not build or submit anything.
⛔ Do not modify any `.pth` or any zip.
