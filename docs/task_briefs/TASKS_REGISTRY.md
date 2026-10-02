# TASK — build the CHECKPOINT PROVENANCE REGISTRY (pure bookkeeping, CPU only)

**Author: Claude. Executor: Gemini 3.1 Pro.**
**Date: 9 Sep 2026. Banked: 79.484440.**

## 0. Rules

1. **This is a bookkeeping task. No training, no GPU, no submission, no conclusions.**
2. ⛔ **DO NOT USE ANY GPU.** Two of Claude's subagents are running on the box right now. This task
   is pure file reading. If you need Python use `/SML_DISK_24TB/rajeshr/Aryamann/env/bin/python`
   with no CUDA.
3. ⛔ Never kill a process that is not yours. Never reboot.
4. If a field cannot be determined, write `UNKNOWN` — **do not guess.** A wrong provenance entry is
   worse than a missing one, because the whole point of this task is to stop us trusting a
   checkpoint we should not.

## 1. Why this matters (context, not opinion to act on)

On 9 Sep we lost most of a round to an error the record calls the **init-leakage trap** (§99):
`ftaug_sv4_m1..m4` carry `args.split = "re_lohi"` — which looks like an honest condition-disjoint
holdout — but `make_init.py` shows they were **initialised from a 100%-data soup**, so they had
already seen the evaluation trajectories. They scored 3–4× better than any genuinely honest member
and were nearly promoted into a submission.

Separately, `local_harness/soup_v3_honest.py:24` uses `vidx = range(0, ntraj, 5)` — the **`every5`**
split, NOT `re_lohi` — so `soup_v3`, used all day as "the honest reference", is leaky on `re_lohi`.

**A checkpoint is honest only if BOTH its training split AND its entire init chain are disjoint from
the evaluation conditions.** Nobody can currently check that without re-deriving it by hand each
time. That is what this registry fixes.

## 2. What to produce

A single file **`/SML_DISK_24TB/rajeshr/Aryamann/UGP/CHECKPOINT_REGISTRY.md`** containing one table
row per model checkpoint larger than 3e8 bytes found under:
* `$B/train_es/*.pth`
* `$B/local_harness/*.pth`
* `$B/data/comp_real/*.pth`

(`$B` = `/SML_DISK_24TB/rajeshr/Aryamann/UGP`.)

Columns, in this order:

| file | bytes | split | init (raw) | init ROOT | steps | lr | wtke | ema | aug | seed | LEAKAGE | in a scored artifact? |

* **split / steps / lr / wtke / ema / aug / seed** — from the sidecar `ftaug_<tag>.json` (key
  `args`) or `<name>_result.json` if present. `UNKNOWN` if there is no sidecar.
* **init (raw)** — the literal `args.init` string, or `kit` if empty/absent.
* **init ROOT** — ⚠️ **the important column. Follow the chain.** If `init` names another checkpoint,
  look up *that* checkpoint's own sidecar and keep going until you reach either
  `data/comp_real/sim_real_fno.pth` / `sim_fno.pth` (write `KIT`) or a soup (write the soup's name).
  Also grep `$B/*.py` and `$B/local_harness/*.py` for the filename to find the script that produced
  it — e.g. `make_init.py` reveals that `init_sv4.pth` is `soup_fno_fp16.pth` unpacked. Record the
  chain compactly, e.g. `init_sv4.pth -> soup_fno_fp16.pth -> SOUP(100% data)`.
* **LEAKAGE** — exactly one of:
  * `HONEST` — init root is KIT **and** split is `re_lohi`. Safe to evaluate on `re_lohi`.
  * `EVERY5` — split (or the producing script's `vidx`) is `range(0, ntraj, 5)`. Trained on the
    `re_lohi` trajectories ⇒ leaky on our ruler, same regime as `soup_v2`.
  * `ALL-DATA` — split is `none` / trained on all 81 trajectories.
  * `INIT-LEAK` — split looks honest but the init root is a soup or an all-data model.
  * `UNKNOWN`.
* **in a scored artifact?** — yes/no. Cross-check against the backbone md5 inside each zip in
  `$B/submissions/` that we have actually submitted (`submission_SV2.zip`, `submission_FP16.zip`,
  `submission_TMEAN.zip`, `submission_LUTCAL.zip`, `submission_SOUP_v1.zip`,
  `submission_SCREEN.zip`, `submission_BLIND.zip`). Use
  `unzip -p <zip> sim_real_fno_fp16.pth | md5sum` and compare against the packed form where you can;
  where a checkpoint is a *member* of a soup rather than the soup itself, note `member of <zip>`
  based on the build scripts, and mark it `UNKNOWN` if you cannot establish it.

## 3. Also produce, at the top of the same file

* A count of each LEAKAGE class.
* An explicit list of every checkpoint classified `HONEST` — this is the only pool that may be used
  as an honest holdout evaluation, and we currently have to re-derive it by hand every time.
* An explicit list of every `INIT-LEAK` checkpoint — these are the dangerous ones that *look* honest.

## 4. Report back

Paste the summary counts and the two lists (HONEST, INIT-LEAK) in your reply. Do not draw
conclusions about which checkpoint is better — that is not this task. Just the provenance.
