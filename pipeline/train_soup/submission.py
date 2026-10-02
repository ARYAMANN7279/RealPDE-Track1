"""RealPDE Track 1 Sim2Real submission: MODEL SOUP + learned per-element bounds.

Differences from submission_FULLSTACK_v7.zip, which scored 77.96:
  * the checkpoint is the model soup (weight-average of 6 fine-tunes of the
    Drive baseline, all sharing that initialization) instead of the untouched
    baseline. Local held-out deltas: rel_l2 -0.30, tke +4.20, mvpe -0.13.
  * the temporal spectral correction is REMOVED. Measured live it was net -0.09
    (accuracy +0.055, time -0.147), and the soup already fixes tke through
    training, so its marginal gain here would be smaller still.
The bounds machinery is unchanged from v7 (clean-LUT per-element head), refitted
on the soup's own errors -- the LUT encodes an error ranking, so it must be fitted
to the model that actually runs.

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

_BATCH = 64          # GPU: fewer python/transfer round-trips than 16
_BATCH_CPU = 8       # CPU: the budget is only checked BETWEEN batches, so a batch
                     # must be small enough that committing to one cannot blow the
                     # deadline. Measured 665 ms/sample on a contended CPU host, so
                     # a 64-window batch is ~43s of unstoppable work; 8 is ~5s.

# ---- time budget -------------------------------------------------------------
# The platform kills the container at 180s and that INCLUDES module import and
# model loading (spec p.3 / FAQ line 308).
#
# The previous version declared exactly that in a comment but then measured the
# budget from a per-call _t0, so load time was effectively free and the real
# worst case was  load(20-40s) + 145s + post-processing  ~= 190s, i.e. OVER the
# limit. It survived on GPU containers (whole run ~40s) and is believed to be
# what killed submission_MAXSOUP_head.zip on a slow/CPU container -- that file
# was byte-identical in code to a submission that had already scored, so the
# archive contents cannot explain the failure (memory 5A diagnostic rule).
#
# Fix: the deadline is now anchored at MODULE IMPORT, so loading is charged
# against it. A per-call cap is kept as well, so that IF the harness makes an
# untimed warm-up call, one call still cannot run away. Whichever binds first
# wins. _RESERVE is held back for building lower/upper and returning, which
# happen after the last budget check and are not free on large N.
_HARD_LIMIT = 180.0
_RESERVE_BASE = 25.0
_CALL_CAP = 145.0
_T_START = _time.time()   # module import: load time is charged against the budget


def _reserve_for(n):
    """Headroom kept free for work that happens after the last budget check:
    allocating lower/upper and filling them with the constant band. Those are two
    full passes over (n,20,32,64,3) float32 -- ~5 GB of writes at n=5140 -- which is
    seconds on an idle host but not on a contended one, so it scales with n."""
    return _RESERVE_BASE + 0.004 * n


def _over_budget(_t0, n, next_cost=0.0):
    """True if the next unit of work should not be started.

    `next_cost` is how long that unit is expected to take, measured from the
    previous one. Checking only elapsed-time was the flaw that let a single
    64-window CPU batch run tens of seconds past an already-expired deadline:
    the check passed, then the batch committed anyway.
    """
    now = _time.time()
    return ((now - _T_START) + next_cost > _HARD_LIMIT - _reserve_for(n)
            or (now - _t0) + next_cost > _CALL_CAP)


# ---- learned per-element bounds ----
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
    _t0 = _time.time()          # per-call cap; the global deadline runs from import
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
    bs = _BATCH if device == "cuda" else _BATCH_CPU
    last_batch = 0.0
    with torch.no_grad():
        for i in range(0, n, bs):
            # 1.5x margin: a contended host can slow down between batches, and
            # overshooting the deadline costs the whole submission (killed, 0-byte
            # result) while stopping early only costs the remaining windows.
            if _over_budget(_t0, n, last_batch * 1.5):
                fallback_from = i
                break
            _tb = _time.time()
            xb = torch.from_numpy(np.ascontiguousarray(x[i:i + bs])).to(device)
            xb = (xb - mean_in) / std_in
            yb = model(xb)
            yb = yb * std_tgt + mean_tgt
            prediction[i:i + bs] = yb.float().cpu().numpy()
            last_batch = _time.time() - _tb
    if fallback_from is not None:
        last = x[fallback_from:, -1:, :, :, :]
        prediction[fallback_from:] = np.repeat(last, 20, axis=1)

    prediction[..., 2] = 0.0  # p is unmeasured in real data

    # ---- learned per-element bounds, computed in CHUNKS so we can bail out ----
    # The feature pass is the expensive part and CPU is shared by up to 8
    # concurrent evaluations on the host. Checking the budget only once before it
    # would risk overrunning the 3-minute limit on a whole large batch, so each
    # chunk is checked and any remainder simply keeps the constant band.
    half = np.array([0.0129, 0.0098, 0.0], dtype=np.float32).reshape(1, 1, 1, 1, 3)

    # lower/upper are written in place. The previous version built a full-size
    # `hh` half-width array and then did `prediction - hh` / `prediction + hh`,
    # which held FIVE arrays of (n,20,32,64,3) live at once (x, prediction, hh,
    # lower, upper) and did two extra full-array passes AFTER the last budget
    # check. Writing straight into lower/upper drops that to four and moves the
    # work inside the budgeted loop.
    lower = np.empty_like(prediction)
    upper = np.empty_like(prediction)
    np.subtract(prediction, half, out=lower)
    np.add(prediction, half, out=upper)
    try:
        z = _assets()
        if (prediction.shape[2], prediction.shape[3]) == (32, 64):
            head = _get_head(device)
            nf = int(z["nf"]); LUT = z["LUT"]; ED = z["ED"]
            _fmu = torch.from_numpy(z["fmu"].astype(np.float32)).to(device)
            _fsd = torch.from_numpy(z["fsd"].astype(np.float32)).to(device)
            CH = 16 if device == "cuda" else 4
            h_chunk = np.empty((CH,) + prediction.shape[1:], dtype=np.float32)
            last_chunk = 0.0
            for i in range(0, prediction.shape[0], CH):
                if _over_budget(_t0, n, last_chunk * 1.5):
                    break                      # remainder keeps the constant band
                _tc = _time.time()
                blk = prediction[i:i + CH]
                with torch.no_grad():
                    bt = torch.from_numpy(np.ascontiguousarray(blk[..., :2])).to(device)
                    ft = (_feats_t(bt) - _fmu) / _fsd
                    fb = ft.mean(dim=1).permute(0, 3, 1, 2)
                    m1 = head(fb).permute(0, 2, 3, 1).cpu().numpy()
                mu = np.repeat(m1[:, None], 20, axis=1)
                hc = h_chunk[:blk.shape[0]]
                for ci in (0, 1):
                    hc[..., ci] = LUT[np.digitize(mu[..., ci], ED[ci]), ci]
                hc[..., 2] = 0.0
                np.subtract(blk, hc, out=lower[i:i + blk.shape[0]])
                np.add(blk, hc, out=upper[i:i + blk.shape[0]])
                last_chunk = _time.time() - _tc
    except Exception:
        pass                      # lower/upper already hold the constant band
    return {"prediction": prediction, "lower": lower, "upper": upper}
