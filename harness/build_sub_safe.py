"""Build a submission zip: fine-tuned FNO (fp16-packed) + per-location SPS bounds.
Adapted from local_harness/build_sub.py (the script that built SOUP_v1) with ONE
addition: the _TIME_BUDGET + persistence-fallback safety mechanism from memory
section 5A, which SOUP_v1's own build did not have. Four submissions were burned
by container timeouts before that fix existed; it costs nothing when inference
finishes on time (which this simple per-location-lookup model should, comfortably)
and prevents a killed-container zero-score if it somehow doesn't.

Layout at the zip ROOT: submission.py  sim_real_fno_fp16.pth  bounds.npy
    load_baseline.py  rpde_baselines/  _vendor/
"""
import argparse, os, shutil, subprocess, sys, zipfile
import numpy as np, torch
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--bounds", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
STAGE = f"{B}/local_harness/_stage_safe"; shutil.rmtree(STAGE, ignore_errors=True); os.makedirs(STAGE)

sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, "_vendor"))
from load_baseline import load_baseline
m, _ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device="cpu")
sd = torch.load(a.ckpt, map_location="cpu")
m.load_state_dict(sd)
tmp = f"{B}/local_harness/_tmp_fp32_safe.pth"
torch.save({"model_state_dict": m.state_dict()}, tmp)
import importlib.util as _ilu
_sp = _ilu.spec_from_file_location("packer", os.path.join(KIT, "pack_ckpt_fp16.py"))
_pk = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_pk)
_pk.pack_fp16(tmp, f"{STAGE}/sim_real_fno_fp16.pth")
os.remove(tmp)
H = np.load(a.bounds).astype(np.float32)
np.save(f"{STAGE}/bounds.npy", H)
print("bounds map", H.shape, "u median %.4f  v median %.4f" % (
    float(np.median(H[..., 0])), float(np.median(H[..., 1]))))
for item in ["load_baseline.py", "rpde_baselines", "_vendor"]:
    src = os.path.join(KIT, item); dst = os.path.join(STAGE, item)
    shutil.copytree(src, dst) if os.path.isdir(src) else shutil.copy2(src, dst)

SUB = '''"""RealPDE Track 1: fine-tuned FNO (model soup) + per-location SPS interval bounds.
Hardened for the 3-minute container limit (memory 5A): tracked per-call so an
untimed warm-up call cannot eat the scored budget; on trip, remaining windows
are filled with persistence (repeat the last input frame) rather than risking
a killed container, which scores zero instead of a low-but-nonzero score."""
from __future__ import annotations
import os, sys, time as _time
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
import torch
from load_baseline import load_baseline

_CKPT = os.path.join(_HERE, "sim_real_fno_fp16.pth")
_BOUNDS = os.path.join(_HERE, "bounds.npy")
_MEAN_IN = np.array([0.154960856, -0.000513992854, 0.0], dtype=np.float32)
_STD_IN = np.array([0.0968056545, 0.015960684, 1.0], dtype=np.float32)
_MEAN_TGT = np.array([0.154962569, -0.000517793698, 0.0], dtype=np.float32)
_STD_TGT = np.array([0.0968104079, 0.0159636438, 1.0], dtype=np.float32)
_BATCH = 64
_TIME_BUDGET = 145.0  # seconds; platform kills the container at 180s INCLUDING load
_state = {"model": None, "device": None, "bounds": None}

def _get():
    if _state["model"] is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model, _meta = load_baseline(_CKPT, device=dev)
        _state["model"] = model.to(dev).eval(); _state["device"] = dev
        _state["bounds"] = np.load(_BOUNDS).astype(np.float32)
    return _state["model"], _state["device"], _state["bounds"]

def predict(input_array, metadata=None):
    _t0 = _time.time()
    x = np.asarray(input_array, dtype=np.float32)
    model, dev, hmap = _get()
    if dev == "cpu":
        try:
            import os as _os
            torch.set_num_threads(max(1, len(_os.sched_getaffinity(0))))
        except Exception:
            pass
    mi = torch.from_numpy(_MEAN_IN).to(dev); si = torch.from_numpy(_STD_IN).to(dev)
    mt = torch.from_numpy(_MEAN_TGT).to(dev); st = torch.from_numpy(_STD_TGT).to(dev)
    n = x.shape[0]
    pred = np.empty((n, 20) + x.shape[2:], dtype=np.float32)
    fallback_from = None
    with torch.no_grad():
        for i in range(0, n, _BATCH):
            if _time.time() - _t0 > _TIME_BUDGET:
                fallback_from = i
                break
            xb = torch.from_numpy(np.ascontiguousarray(x[i:i + _BATCH])).to(dev)
            yb = (model((xb - mi) / si) * st + mt)
            pred[i:i + _BATCH] = yb.float().cpu().numpy()
    if fallback_from is not None:
        last = x[fallback_from:, -1:, :, :, :]
        pred[fallback_from:] = np.repeat(last, 20, axis=1)
    pred[..., 2] = 0.0

    H, W = pred.shape[2], pred.shape[3]
    half = np.zeros_like(pred)
    if hmap.shape[0] == H and hmap.shape[1] == W:
        half[..., 0] = hmap[None, None, :, :, 0]; half[..., 1] = hmap[None, None, :, :, 1]
    else:
        half[..., 0] = 0.0155; half[..., 1] = 0.0090
    half[..., 2] = 0.0
    return {"prediction": pred, "lower": pred - half, "upper": pred + half}
'''
open(f"{STAGE}/submission.py", "w").write(SUB)
if os.path.exists(a.out): os.remove(a.out)
zf = zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED)
tot = 0
for root, _, files in os.walk(STAGE):
    for f in files:
        p = os.path.join(root, f); tot += os.path.getsize(p)
        zf.write(p, os.path.relpath(p, STAGE))
zf.close()
print("zip: %s  (%.1f MB zipped, %.1f MB extracted; cap 256)" % (a.out, os.path.getsize(a.out)/1e6, tot/1e6))
assert tot/1e6 < 256, "OVER THE 256 MB EXTRACTED CAP"
