"""finetune2 -- finetune.py plus the three axes that were never varied.

Identical data pipeline, normalisation constants, split rule and eval protocol to
local_harness/finetune.py, so runs are directly comparable to the 20 existing
ft_*_result.json. New knobs, all default-off:

  --wmvpe    weight on the differentiable mvpe term (verified exact vs scoring.py)
  --ema      EMA decay on the weights (0 = off). Distinct from model soup: soup
             averages endpoints ACROSS runs, EMA averages ALONG one trajectory.
  --valstride  5 = the historical every-5th holdout (65 train / 17 val).
             10 keeps ~73 train / 9 val -- the "more data" axis, but unlike
             sec14 Experiment A the step budget is scaled with it, not fixed.

Two checkpoints are saved per run:
  best-by-proj  the historical selection objective (0.306/0.163/0.218), comparable
  best-by-eff   marginal value per subscore point INCLUDING the W->sps channel
                (0.463/0.207/0.277 at our operating point, derived from scoring.py)
"""
import argparse, json, os, sys, time, copy
import numpy as np, torch

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, "_vendor"))
sys.path.insert(0, f"{B}/train_mvpe")
import importlib.util
spec = importlib.util.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
from load_baseline import load_baseline
from mvpe_torch import mvpe_per_sample

ap = argparse.ArgumentParser()
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--steps", type=int, default=8000)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--wtke", type=float, default=0.15)
ap.add_argument("--wmvpe", type=float, default=0.0)
ap.add_argument("--ema", type=float, default=0.0)
ap.add_argument("--valstride", type=int, default=5)
ap.add_argument("--split", default="every5",
                choices=["every5", "re_lohi", "re_int", "aoa15"],
                help="every5 = the historical trajectory-disjoint split. It is NOT "
                     "condition-disjoint: all 18 Reynolds numbers appear in training, "
                     "so each val trajectory has 3-4 same-Re siblings (different AoA) "
                     "in train. The private eval set has UNSEEN Re/angles. re_lohi / "
                     "re_int / aoa15 hold out whole conditions, size-matched to every5.")
ap.add_argument("--wd", type=float, default=1e-6)
ap.add_argument("--evalevery", type=int, default=500)
ap.add_argument("--tag", default="x")
a = ap.parse_args()

DEV = f"cuda:{a.gpu}"; C = 2; IN = 20; OUT = 20
OUTDIR = f"{B}/train_mvpe/runs"; os.makedirs(OUTDIR, exist_ok=True)

MI = torch.tensor([0.154960856, -0.000513992854, 0.0])
SI = torch.tensor([0.0968056545, 0.015960684, 1.0])
MT = torch.tensor([0.154962569, -0.000517793698, 0.0])
ST = torch.tensor([0.0968104079, 0.0159636438, 1.0])
MI, SI, MT, ST = [t.to(DEV) for t in (MI, SI, MT, ST)]

X = np.load(f"{B}/local_harness/tr_frames.npy", mmap_mode="r")
meta = json.load(open(f"{B}/local_harness/tr_meta.json"))
off, lens = meta["off"], meta["lens"]
ntraj = len(lens)
names = meta["names"]
import re as _re
_re_of = lambda t: float(_re.match(r"([0-9.]+)_", t).group(1))
_aoa_of = lambda t: t.split("_")[1].replace(".h5", "")
if a.split == "every5":
    vidx = set(range(0, ntraj, a.valstride))
elif a.split == "re_lohi":            # 2 lowest + 2 highest Re -> 16 traj
    hold = {3750.0, 5025.0, 25425.0, 26700.0}
    vidx = {i for i in range(ntraj) if _re_of(names[i]) in hold}
elif a.split == "re_int":             # 4 interior Re -> 18 traj
    hold = {8850.0, 13950.0, 19050.0, 22875.0}
    vidx = {i for i in range(ntraj) if _re_of(names[i]) in hold}
elif a.split == "aoa15":              # one whole angle of attack -> 16 traj
    vidx = {i for i in range(ntraj) if _aoa_of(names[i]) == "15"}
_vre = sorted({_re_of(names[i]) for i in vidx})
_tre = sorted({_re_of(names[i]) for i in range(ntraj) if i not in vidx})
_leak = sorted(set(_vre) & set(_tre))
print("[split] %s | val Re %s" % (a.split, _vre), flush=True)
print("[split] Re present in BOTH train and val (leak): %s" % (_leak if _leak else "none"), flush=True)
starts_tr, starts_va = [], []
for i in range(ntraj):
    for t0 in range(off[i], off[i] + lens[i] - 39):
        (starts_va if i in vidx else starts_tr).append(t0)
starts_tr = np.array(starts_tr); starts_va = np.array(starts_va)
print("[data] %d trajectories | train %d win (%d traj) | val %d win (%d traj)" % (
    ntraj, len(starts_tr), ntraj - len(vidx), len(starts_va), len(vidx)), flush=True)

rng = np.random.default_rng(0)
va_sub = rng.choice(starts_va, size=min(600, len(starts_va)), replace=False)

def batch(idx):
    w = np.stack([np.asarray(X[s:s + 40]) for s in idx])
    z = np.zeros(w.shape[:-1] + (1,), np.float32)
    w = np.concatenate([w, z], -1)
    return torch.from_numpy(w[:, :IN]).to(DEV), torch.from_numpy(w[:, IN:]).to(DEV)

model, _ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=DEV)
model = model.to(DEV)

def fwd(x): return model((x - MI) / SI) * ST + MT

def rel_l2(p, t):
    p = p[..., :C].reshape(p.shape[0], -1); t = t[..., :C].reshape(t.shape[0], -1)
    return (torch.linalg.norm(p - t, dim=1) / torch.linalg.norm(t, dim=1).clamp(min=1e-8)).mean()

def tke_l2(p, t):
    def ke(z):
        u, v = z[..., 0], z[..., 1]
        return 0.5 * (((u - u.mean(1, keepdim=True)) ** 2).mean(1)
                      + ((v - v.mean(1, keepdim=True)) ** 2).mean(1))
    pk = ke(p[..., :C]).reshape(p.shape[0], -1); tk = ke(t[..., :C]).reshape(t.shape[0], -1)
    return (torch.linalg.norm(pk - tk, dim=1) / torch.linalg.norm(tk, dim=1).clamp(min=1e-8)).mean()

# ---- EMA -------------------------------------------------------------------
ema_sd = None
if a.ema > 0:
    # NOTE: no .float() here -- the FNO carries 16 COMPLEX spectral tensors and
    # .float() silently drops their imaginary part (sec8 complex-tensor trap).
    ema_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

def ema_update():
    sd = model.state_dict()
    for k, v in ema_sd.items():
        src = sd[k].detach()
        if v.is_complex() or v.is_floating_point():
            v.mul_(a.ema).add_(src, alpha=1 - a.ema)
        else:
            v.copy_(src)

class swap_ema:
    """Evaluate under EMA weights, then restore the live ones."""
    def __enter__(self):
        if ema_sd is None: return
        self.live = copy.deepcopy(model.state_dict())
        model.load_state_dict({k: v.clone() for k, v in ema_sd.items()})
    def __exit__(self, *e):
        if ema_sd is None: return
        model.load_state_dict(self.live)

@torch.no_grad()
def evaluate():
    model.eval(); P = []; T = []
    for i in range(0, len(va_sub), 16):
        x, y = batch(va_sub[i:i + 16])
        P.append(fwd(x).cpu().numpy()); T.append(y.cpu().numpy())
    model.train()
    P = np.concatenate(P, 0).astype(np.float32); T = np.concatenate(T, 0).astype(np.float32)
    dm = S.rel_l2_per_sample(P, T, C); tk = S.tke_rel_l2_per_sample(P, T, C)
    mv = S.mvpe_rel_l2_per_sample(P, T)
    return (S.score_error(float(dm.mean())), S.score_error(float(tk.mean())),
            S.score_error(float(mv.mean())))

# historical selection objective, and the marginal-value one (W->sps included)
proj = lambda r: 0.306 * r[0] + 0.163 * r[1] + 0.218 * r[2]
eff  = lambda r: 0.463 * r[0] + 0.207 * r[1] + 0.277 * r[2]

with swap_ema():
    base = evaluate()
b_proj, b_eff = proj(base), eff(base)
print("[base] rel_l2 %.2f tke %.2f mvpe %.2f" % base, flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
best_p = (b_proj, base, 0); best_e = (b_eff, base, 0)
hist = []; t0 = time.time(); model.train()

for step in range(1, a.steps + 1):
    idx = rng.choice(starts_tr, size=a.bs, replace=False)
    x, y = batch(idx)
    p = fwd(x)
    loss = rel_l2(p, y) + a.wtke * tke_l2(p, y)
    if a.wmvpe > 0:
        loss = loss + a.wmvpe * mvpe_per_sample(p, y).mean()
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if ema_sd is not None: ema_update()

    if step % a.evalevery == 0 or step == a.steps:
        with swap_ema():
            r = evaluate()
        dp, de = proj(r) - b_proj, eff(r) - b_eff
        hist.append(dict(step=step, r=r, d_proj=dp, d_eff=de, loss=float(loss)))
        flag = ""
        if proj(r) > best_p[0]:
            best_p = (proj(r), r, step)
            sd = ema_sd if ema_sd is not None else model.state_dict()
            torch.save({k: v.detach().cpu() for k, v in sd.items()},
                       f"{OUTDIR}/{a.tag}_bestproj.pth"); flag += " *proj"
        if eff(r) > best_e[0]:
            best_e = (eff(r), r, step)
            sd = ema_sd if ema_sd is not None else model.state_dict()
            torch.save({k: v.detach().cpu() for k, v in sd.items()},
                       f"{OUTDIR}/{a.tag}_besteff.pth"); flag += " *eff"
        print("step %5d loss %.4f | rel_l2 %.2f tke %.2f mvpe %.2f | dproj %+.3f deff %+.3f | %.0fs%s"
              % (step, float(loss), r[0], r[1], r[2], dp, de, time.time() - t0, flag), flush=True)

print("[best-proj] step %d  rel_l2 %.2f tke %.2f mvpe %.2f  dproj %+.4f" % (
    best_p[2], best_p[1][0], best_p[1][1], best_p[1][2], best_p[0] - b_proj), flush=True)
print("[best-eff ] step %d  rel_l2 %.2f tke %.2f mvpe %.2f  deff  %+.4f" % (
    best_e[2], best_e[1][0], best_e[1][1], best_e[1][2], best_e[0] - b_eff), flush=True)
json.dump(dict(cfg=vars(a), base=base,
               best_proj=dict(step=best_p[2], r=best_p[1], d=best_p[0] - b_proj),
               best_eff=dict(step=best_e[2], r=best_e[1], d=best_e[0] - b_eff),
               hist=hist),
          open(f"{OUTDIR}/{a.tag}_result.json", "w"), indent=1)
