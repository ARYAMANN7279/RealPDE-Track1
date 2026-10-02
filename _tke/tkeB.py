"""tke TRANSFER FORENSICS -- experiment B.

Two questions on ONE ruler (the gated lohi664 / lohi_all protocol):

B1) FINAL WEIGHTS vs BEST-ON-VAL.  ffno_mv.py saves <tag>_final.pth alongside the
    best-on-val <tag>.pth.  If checkpoint-selection pressure is what fails to transfer,
    the final-weights model should lose less between the selection split and elsewhere.

B2) RE-PRICE THE F-FNO HONESTLY.  sec57.6 closed the F-FNO by comparing its HONEST
    re_lohi score (78.0774 tke) against the SHIPPED soup's LEAKY re_lohi score (80.8027).
    Both models are put on the honest ruler here.

MANDATORY: prints split sizes; asserts the honest-soup baseline; ABORTS on mismatch.
"""
import os, sys, json, argparse, importlib.util as iu
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"; LH = f"{B}/local_harness"
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, "_vendor"))
sp = iu.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
S = iu.module_from_spec(sp); sp.loader.exec_module(S)
sys.path.insert(0, f"{B}/_bld")
from load_baseline import load_baseline
lb = iu.spec_from_file_location("lb2", f"{B}/_bld/load_baseline.py")
LB2 = iu.module_from_spec(lb); lb.loader.exec_module(LB2)

ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=0)
a = ap.parse_args(); DEV = f"cuda:{a.gpu}"; C = 2
MI = torch.tensor([0.154960856, -0.000513992854, 0.0]).to(DEV)
SI = torch.tensor([0.0968056545, 0.015960684, 1.0]).to(DEV)
MT = torch.tensor([0.154962569, -0.000517793698, 0.0]).to(DEV)
ST = torch.tensor([0.0968104079, 0.0159636438, 1.0]).to(DEV)

class FactorizedSpectralConv3d(nn.Module):
    def __init__(s, width, m1, m2, m3):
        super().__init__(); s.m = (m1, m2, m3); c = 1.0 / width
        s.w1 = nn.Parameter(c * torch.randn(width, width, m1, dtype=torch.cfloat))
        s.w2 = nn.Parameter(c * torch.randn(width, width, m2, dtype=torch.cfloat))
        s.w3 = nn.Parameter(c * torch.randn(width, width, m3, dtype=torch.cfloat))
    def forward(s, x):
        Bs, Cc, T, H, W = x.shape
        xf = torch.fft.rfft(x, dim=2, norm="ortho"); o = torch.zeros_like(xf); k = min(s.m[0], xf.shape[2])
        o[:, :, :k] = torch.einsum("bckhw,cok->bokhw", xf[:, :, :k], s.w1[:, :, :k])
        y = torch.fft.irfft(o, n=T, dim=2, norm="ortho")
        xf = torch.fft.rfft(x, dim=3, norm="ortho"); o = torch.zeros_like(xf); k = min(s.m[1], xf.shape[3])
        o[:, :, :, :k] = torch.einsum("bctkw,cok->botkw", xf[:, :, :, :k], s.w2[:, :, :k])
        y = y + torch.fft.irfft(o, n=H, dim=3, norm="ortho")
        xf = torch.fft.rfft(x, dim=4, norm="ortho"); o = torch.zeros_like(xf); k = min(s.m[2], xf.shape[4])
        o[:, :, :, :, :k] = torch.einsum("bcthk,cok->bothk", xf[:, :, :, :, :k], s.w3[:, :, :k])
        return y + torch.fft.irfft(o, n=W, dim=4, norm="ortho")

class FFNO3d(nn.Module):
    def __init__(s, width=64, n_layers=8, shape_in=(20, 32, 64, 3), shape_out=(20, 32, 64, 3), pad=4):
        super().__init__()
        s.shape_in, s.shape_out, s.pad, s.width = shape_in, shape_out, pad, width
        s.dim_in = shape_in[-1]; s.dim_out = shape_out[-1] * shape_out[0] // shape_in[0]
        T, H, W = shape_in[0] + pad, shape_in[1] + pad, shape_in[2] + pad
        m1, m2, m3 = T // 2 + 1, H // 2 + 1, W // 2 + 1
        s.fc0 = nn.Linear(s.dim_in + 3, width)
        s.blocks = nn.ModuleList([FactorizedSpectralConv3d(width, m1, m2, m3) for _ in range(n_layers)])
        s.mlps = nn.ModuleList([nn.Sequential(nn.Conv3d(width, width * 2, 1), nn.GELU(),
                                              nn.Conv3d(width * 2, width, 1)) for _ in range(n_layers)])
        s.norms = nn.ModuleList([nn.GroupNorm(8, width) for _ in range(n_layers)])
        s.fc1 = nn.Linear(width, 128); s.fc2 = nn.Linear(128, s.dim_out)
    def get_grid(s, shape, device):
        b, sx, sy, sz = shape[0], shape[1], shape[2], shape[3]
        gx = torch.linspace(0, 1, sx, device=device).reshape(1, sx, 1, 1, 1).expand(b, sx, sy, sz, 1)
        gy = torch.linspace(0, 1, sy, device=device).reshape(1, 1, sy, 1, 1).expand(b, sx, sy, sz, 1)
        gz = torch.linspace(0, 1, sz, device=device).reshape(1, 1, 1, sz, 1).expand(b, sx, sy, sz, 1)
        return torch.cat((gx, gy, gz), dim=-1)
    def forward(s, x):
        x = torch.cat((x, s.get_grid(x.shape, x.device)), dim=-1)
        x = s.fc0(x).permute(0, 4, 1, 2, 3); p = s.pad
        x = F.pad(x, [0, p, 0, p, 0, p])
        for blk, mlp, nrm in zip(s.blocks, s.mlps, s.norms): x = x + mlp(F.gelu(nrm(blk(x))))
        x = x[..., :-p, :-p, :-p].permute(0, 2, 3, 4, 1)
        x = s.fc2(F.gelu(s.fc1(x)))
        x = x.reshape(*x.shape[:-1], s.shape_out[-1], s.shape_out[0] // s.shape_in[0])
        return x.permute(0, 1, 5, 2, 3, 4).reshape(x.shape[0], *s.shape_out)

meta = json.load(open(f"{LH}/tr_meta.json")); off, lens, names = meta["off"], meta["lens"], meta["names"]
X = np.load(f"{LH}/tr_frames.npy", mmap_mode="r")
wins, wt = [], []
for i in range(len(lens)):
    for t0 in range(off[i], off[i] + lens[i] - 39, 10): wins.append(t0); wt.append(i)
wins = np.array(wins); wt = np.array(wt)
RE = np.array([int(names[i].split("_")[0]) for i in wt])
AOA = np.array([int(names[i].split("_", 1)[1].replace(".h5", "")) for i in wt])
assert len(wins) == 6602, "ABORT window construction"
HOLD = [3750, 5025, 25425, 26700]
SPL = {"lohi664": np.where(np.isin(RE, HOLD))[0][::2],
       "lohi_odd": np.where(np.isin(RE, HOLD))[0][1::2],
       "aoa15": np.where(AOA == 15)[0]}
for k, v in SPL.items(): print("split %-9s %5d windows" % (k, len(v)), flush=True)
assert len(SPL["lohi664"]) == 664, "ABORT lohi664"
pad = lambda arr: np.concatenate([arr, np.zeros(arr.shape[:-1] + (1,), np.float32)], -1)

def run(model, idx, bs=8):
    model.eval(); P = np.zeros((len(idx), 20, 32, 64, 2), np.float32); T = np.zeros_like(P)
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            sl = idx[i:i + bs]
            w = np.stack([np.concatenate([np.asarray(X[s:s + 40]),
                          np.zeros((40, 32, 64, 1), np.float32)], -1) for s in wins[sl]]).astype(np.float32)
            xb = torch.from_numpy(w[:, :20]).to(DEV)
            p = (model((xb - MI) / SI) * ST + MT).float().cpu().numpy()
            P[i:i + bs] = p[..., :2]; T[i:i + bs] = w[:, 20:, ..., :2]
    return (S.score_error(float(S.rel_l2_per_sample(pad(P), pad(T), C).mean())),
            S.score_error(float(S.tke_rel_l2_per_sample(pad(P), pad(T), C).mean())),
            S.score_error(float(S.mvpe_rel_l2_per_sample(pad(P), pad(T)).mean())))

MODELS = []
proto, _ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=DEV); proto.eval()
MODELS.append(("kit_fno", proto))
m2, _ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=DEV)
m2.load_state_dict(LB2.unpack_fp16(f"{B}/_bldfp16/sim_real_fno_fp16.pth")); m2.eval()
MODELS.append(("shipped_soup(LEAKY here)", m2))
m3, _ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=DEV)
m3.load_state_dict(torch.load(f"{B}/train_es/soup_lohi_honest.pth", map_location=DEV)); m3.eval()
MODELS.append(("honest_soup", m3))
for tag, p in (("ffno_ft_SELECTED", f"{B}/train_arch/ft_w64l8_lr3e4.pth"),
               ("ffno_ft_FINAL", f"{B}/train_arch/ft_w64l8_lr3e4_final.pth")):
    if not os.path.exists(p): print("MISSING", p); continue
    f = FFNO3d(width=64, n_layers=8, pad=4).to(DEV)
    sd = torch.load(p, map_location=DEV)
    if isinstance(sd, dict) and "model_state_dict" in sd: sd = sd["model_state_dict"]
    f.load_state_dict(sd); f.eval(); MODELS.append((tag, f))

KNOWN = (95.3069, 78.7002, 96.0540); MV = dict(rel_l2=0.669, tke=0.157, mvpe=0.170)
res = {}
for sn, idx in SPL.items():
    print("\n--- %s (%d windows) ---" % (sn, len(idx)), flush=True)
    for mn, mm in MODELS:
        r = run(mm, idx); res[(sn, mn)] = r
        print("  %-26s rel %8.4f  tke %8.4f  mvpe %8.4f" % (mn, *r), flush=True)
        if sn == "lohi664" and mn == "honest_soup":
            dev = max(abs(x - y) for x, y in zip(r, KNOWN))
            print("  GATE vs KNOWN %s -> dev %.4f" % (KNOWN, dev), flush=True)
            assert dev < 0.02, "BASELINE MISMATCH -- ABORT"
            print("  GATE PASSED", flush=True)

print("\n=== B1: FINAL minus SELECTED (F-FNO), per split ===", flush=True)
for sn in SPL:
    s1 = res.get((sn, "ffno_ft_SELECTED")); s2 = res.get((sn, "ffno_ft_FINAL"))
    if s1 and s2:
        print("  %-9s d_rel %+7.4f  d_tke %+7.4f  d_mvpe %+7.4f  d_acc %+7.4f" % (
            sn, s2[0] - s1[0], s2[1] - s1[1], s2[2] - s1[2],
            0.669 * (s2[0] - s1[0]) + 0.157 * (s2[1] - s1[1]) + 0.170 * (s2[2] - s1[2])), flush=True)
print("\n=== B2: F-FNO vs HONEST soup on the honest ruler ===", flush=True)
for sn in SPL:
    h = res[(sn, "honest_soup")]; sh = res[(sn, "shipped_soup(LEAKY here)")]
    for tag in ("ffno_ft_SELECTED", "ffno_ft_FINAL"):
        if (sn, tag) not in res: continue
        f = res[(sn, tag)]
        print("  %-9s %-18s vs honest: d_tke %+7.4f d_acc %+7.4f | vs LEAKY shipped: d_tke %+7.4f d_acc %+7.4f" % (
            sn, tag, f[1] - h[1],
            0.669 * (f[0] - h[0]) + 0.157 * (f[1] - h[1]) + 0.170 * (f[2] - h[2]),
            f[1] - sh[1],
            0.669 * (f[0] - sh[0]) + 0.157 * (f[1] - sh[1]) + 0.170 * (f[2] - sh[2])), flush=True)
json.dump({f"{k[0]}|{k[1]}": v for k, v in res.items()}, open(f"{B}/_tke/tkeB.json", "w"), indent=1)
print("DONE", flush=True)
