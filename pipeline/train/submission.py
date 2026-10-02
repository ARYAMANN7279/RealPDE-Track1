"""RealPDE Track 1 Sim2Real submission: official FNO baseline (sim_real_ft, fp16-packed).

This is the exact submission the organizers ran on the Main Development
leaderboard, kept here as a worked example of wrapping a pretrained baseline
behind the Track 1 ``predict()`` API.

How to use it
-------------
1. Download ``sim_real_ft/sim_real_fno_fp16.pth`` from the competition Google
   Drive (see README).
2. Build a zip whose ROOT contains, side by side:

       submission.py                  <- this file, renamed
       sim_real_fno_fp16.pth          <- the checkpoint
       load_baseline.py               <- from this kit
       rpde_baselines/                <- from this kit
       _vendor/                       <- from this kit

3. Submit the zip. Swap ``load_baseline`` / the checkpoint for your own model
   once the round trip works.

What it does
------------
Mirrors the benchmark evaluation pipeline (realpdebench ``eval.py``): inputs are
z-scored with GaussianNormalizer statistics fitted on train_real, the FNO runs in
normalized space, and the prediction is mapped back to physical space with the
target statistics. Skipping this normalization is the single most common reason
a correctly-loaded baseline still scores badly.
"""

from __future__ import annotations

import os
import sys

import time as _time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402

from load_baseline import load_baseline  # noqa: E402  (prepends _vendor/ for einops)

_CKPT_PATH = os.path.join(_HERE, "sim_real_fno_fp16.pth")

# GaussianNormalizer statistics fitted on train_real 32x64 windows, channel
# order [u, v, p]; extracted from the benchmark's cached mean_std_real.pt.
# The p channel is unmeasured in real data (all zeros), so its stored std is 0
# and is replaced by 1 at runtime, exactly as GaussianNormalizer.__init__ does.
_MEAN_IN = np.array([0.154960856, -0.000513992854, 0.0], dtype=np.float32)
_STD_IN = np.array([0.0968056545, 0.015960684, 1.0], dtype=np.float32)
_MEAN_TGT = np.array([0.154962569, -0.000517793698, 0.0], dtype=np.float32)
_STD_TGT = np.array([0.0968104079, 0.0159636438, 1.0], dtype=np.float32)

_BATCH = 64          # fewer python/transfer round-trips than 16
_TIME_BUDGET = 145.0  # seconds; platform kills the container at 180s INCLUDING load
_T_START = _time.time()   # replaced per-call below (warm-up call must not consume the scored budget)


# ---- full-stack additions: temporal spectral correction + learned bounds ----
_ASSETS = os.path.join(_HERE, "head_assets.npz")
_A = {"z": None}

def _assets():
    if _A["z"] is None:
        _A["z"] = np.load(_ASSETS)
    return _A["z"]

class _Head(torch.nn.Module):
    def __init__(self, nf, w=128):
        super().__init__()
        import torch.nn as nn
        self.n = nn.Sequential(
            nn.Conv2d(nf, w, 3, padding=1), nn.GELU(),
            nn.Conv2d(w, w, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv2d(w, w, 3, padding=4, dilation=4), nn.GELU(),
            nn.Conv2d(w, w, 3, padding=8, dilation=8), nn.GELU(),
            nn.Conv2d(w, w, 3, padding=1), nn.GELU(),
            nn.Conv2d(w, 2, 1))
    def forward(self, x):
        return self.n(x)

def _get_head(device):
    if _state.get("head") is None:
        z = _assets()
        nf = int(z["nf"])
        h = _Head(nf)
        sd = {k[2:]: torch.from_numpy(z["w_" + k[2:]]) for k in z.files if k.startswith("w_")}
        h.load_state_dict(sd)
        _state["head"] = h.to(device).eval()
    return _state["head"]

def _feats_t(P_t):
    """Features on-device. CPU is shared across up to 8 concurrent evaluations on
    the host, so the old numpy pass was the bottleneck and the part most exposed
    to contention. torch.gradient mirrors np.gradient's central differences."""
    u = P_t[..., 0]; v = P_t[..., 1]
    def gx(a): return torch.gradient(a, dim=3)[0]
    def gy(a): return torch.gradient(a, dim=2)[0]
    gux, guy, gvx, gvy = gx(u), gy(u), gx(v), gy(v)
    g = torch.sqrt(gux**2 + guy**2 + gvx**2 + gvy**2)
    vo = (gvx - guy).abs()
    lap = (gx(gux) + gy(guy)).abs()
    tv = u.std(dim=1, keepdim=True).expand_as(u)
    tvv = v.std(dim=1, keepdim=True).expand_as(v)
    ke = 0.5 * (u * u + v * v)
    dev = (u - u.mean(dim=1, keepdim=True)).abs()
    dv = (v - v.mean(dim=1, keepdim=True)).abs()
    H, Wd = u.shape[2], u.shape[3]
    yy = torch.linspace(-1, 1, H, device=u.device).view(1, 1, H, 1).expand_as(u)
    xx = torch.linspace(-1, 1, Wd, device=u.device).view(1, 1, 1, Wd).expand_as(u)
    tf = torch.linspace(0, 1, u.shape[1], device=u.device).view(1, -1, 1, 1).expand_as(u)
    L = lambda a, e: torch.log(a + e)
    return torch.stack([L(g,1e-6), L(vo,1e-6), L(lap,1e-6), L(tv,1e-6), L(tvv,1e-6),
                        L(ke,1e-9), L(dev,1e-6), L(dv,1e-6), u, v, xx, yy, tf], dim=-1)

_state = {"model": None, "device": None}


def _get_model():
    if _state["model"] is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _meta = load_baseline(_CKPT_PATH, device=device)
        model = model.to(device)
        model.eval()
        _state["model"] = model
        _state["device"] = device
    return _state["model"], _state["device"]


def predict(input_array, metadata=None):
    """Track 1 API: (N, T_in, H, W, C) raw fields -> (N, T_out, H, W, C).

    Hardened for the 3-minute container limit, which INCLUDES model loading and
    applies whether or not a GPU is allocated. On CPU this model runs ~66x
    slower, so a plain loop can be killed before it writes any output (which is
    scored as Failed, not as a low score). Two guards:
      * use every available CPU thread, and a larger batch
      * track elapsed time and, if the budget would be exceeded, finish the
        remaining windows with persistence (repeat the last input frame).
        Persistence is far from ideal but scores; being killed does not.
    """
    _t0 = _time.time()          # per-call: the untimed warm-up call must not eat the budget
    x = np.asarray(input_array, dtype=np.float32)
    model, device = _get_model()
    if device == "cpu":
        try:
            import os as _os
            torch.set_num_threads(max(1, len(_os.sched_getaffinity(0))))
        except Exception:
            pass

    mean_in = torch.from_numpy(_MEAN_IN).to(device)
    std_in = torch.from_numpy(_STD_IN).to(device)
    mean_tgt = torch.from_numpy(_MEAN_TGT).to(device)
    std_tgt = torch.from_numpy(_STD_TGT).to(device)

    n = x.shape[0]
    prediction = np.empty((n, 20) + x.shape[2:], dtype=np.float32)
    fallback_from = None
    with torch.no_grad():
        for i in range(0, n, _BATCH):
            if _time.time() - _t0 > _TIME_BUDGET:
                fallback_from = i
                break
            xb = torch.from_numpy(np.ascontiguousarray(x[i:i + _BATCH])).to(device)
            xb = (xb - mean_in) / std_in
            yb = model(xb)
            yb = yb * std_tgt + mean_tgt
            prediction[i:i + _BATCH] = yb.float().cpu().numpy()
    if fallback_from is not None:
        last = x[fallback_from:, -1:, :, :, :]
        prediction[fallback_from:] = np.repeat(last, 20, axis=1)

    prediction[..., 2] = 0.0  # p is unmeasured in real data

    # ---- temporal spectral correction: the FNO loses energy at mid temporal
    # frequencies (classic FNO spectral bias); restore it without touching phase.
    half = np.array([0.0129, 0.0098, 0.0], dtype=np.float32).reshape(1, 1, 1, 1, 3)
    try:
        z = _assets()
        gt = z["gt"]
        if prediction.shape[1] == 20 and gt.shape[0] == 11:
            # On GPU: the numpy FFT over the whole array was a CPU bottleneck, and
            # CPU is shared by up to 8 concurrent evaluations on the host.
            gt_t = torch.from_numpy(gt.astype(np.complex64)).to(device)
            for i in range(0, prediction.shape[0], 256):
                pt = torch.from_numpy(np.ascontiguousarray(prediction[i:i + 256, ..., :2])).to(device)
                Fq = torch.fft.rfft(pt, dim=1) * gt_t.view(1, -1, 1, 1, 2)
                prediction[i:i + 256, ..., :2] = torch.fft.irfft(Fq, n=20, dim=1).cpu().numpy()
    except Exception:
        pass

    # ---- learned per-element bounds, computed in CHUNKS so we can bail out ----
    # The numpy feature pass is the expensive part (~17 ms/window) and runs on CPU
    # regardless of GPU. Checking the budget only once before it would risk
    # overrunning the 3-minute limit on a whole large batch, so each chunk is
    # checked and any remainder simply keeps the constant band.
    try:
        z = _assets()
        if (prediction.shape[2], prediction.shape[3]) == (32, 64):
            head = _get_head(device)
            nf = int(z["nf"]); LUT = z["LUT"]; ED = z["ED"]
            _fmu = torch.from_numpy(z["fmu"].astype(np.float32)).to(device)
            _fsd = torch.from_numpy(z["fsd"].astype(np.float32)).to(device)
            hh = np.empty_like(prediction)
            hh[..., 0] = half[0, 0, 0, 0, 0]; hh[..., 1] = half[0, 0, 0, 0, 1]
            hh[..., 2] = 0.0
            CH = 16
            for i in range(0, prediction.shape[0], CH):
                if _time.time() - _t0 > _TIME_BUDGET:
                    break                      # remainder keeps the constant band
                blk = prediction[i:i + CH]
                with torch.no_grad():
                    bt = torch.from_numpy(np.ascontiguousarray(blk[..., :2])).to(device)
                    ft = (_feats_t(bt) - _fmu) / _fsd
                    fb = ft.mean(dim=1).permute(0, 3, 1, 2)
                    m1 = head(fb).permute(0, 2, 3, 1).cpu().numpy()
                mu = np.repeat(m1[:, None], 20, axis=1)
                for ci in (0, 1):
                    hh[i:i + CH, ..., ci] = LUT[np.digitize(mu[..., ci], ED[ci]), ci]
            return {"prediction": prediction, "lower": prediction - hh, "upper": prediction + hh}
    except Exception:
        pass
    return {"prediction": prediction, "lower": prediction - half, "upper": prediction + half}
