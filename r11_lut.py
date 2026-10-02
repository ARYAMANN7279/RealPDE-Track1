import os, sys, time, json, zipfile, io, hashlib
import numpy as np
import torch
import importlib.util as iu

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
LH = f"{B}/local_harness"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"

sys.path.insert(0, KIT)
sys.path.insert(0, os.path.join(KIT, "_vendor"))
sp = iu.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
S = iu.module_from_spec(sp)
sp.loader.exec_module(S)

sys.path.insert(0, f"{B}/_r9")
import submission
from submission import predict, _UNet, _fold

def custom_load_baseline(path, device):
    from load_baseline import load_baseline
    model, _ = load_baseline(path, device="cpu")
    raw = torch.load(f"{B}/train_es/soup_v3_fp16.pth", map_location="cpu")
    complex_keys = set(raw.get("complex_keys", []))
    if "state_fp16" in raw:
        sd = {}
        for k, v in raw["state_fp16"].items():
            if k in complex_keys:
                sd[k] = torch.view_as_complex(v.float())
            elif torch.is_tensor(v) and v.dtype == torch.float16:
                sd[k] = v.float()
            else:
                sd[k] = v
    else:
        sd = raw.get("model_state_dict", raw)
    model.load_state_dict(sd)
    return model, None

submission.load_baseline = custom_load_baseline

meta = json.load(open(f"{LH}/tr_meta.json"))
off, lens, names = meta["off"], meta["lens"], meta["names"]
ntraj = len(lens)
RE = np.array([int(n.split("_")[0]) for n in names])
VT = set(np.where(np.isin(RE, [3750, 5025, 25425, 26700]))[0].tolist())

X32 = np.load(f"{LH}/tr_frames.npy", mmap_mode="r")
starts_va = []
for i in range(ntraj):
    for t0 in range(off[i], off[i]+lens[i]-39):
        if i in VT:
            starts_va.append(t0)
starts_va = np.array(starts_va)
rng = np.random.default_rng(0)
va_sub = rng.choice(starts_va, size=min(900, len(starts_va)), replace=False)

def batch_va(idx):
    w = np.stack([np.asarray(X32[s:s+40]) for s in idx])
    z = np.zeros(w.shape[:-1]+(1,), np.float32)
    w = np.concatenate([w, z], -1).astype(np.float32)
    return w[:,:20], w[:,20:]

chunk_size = 120
P_list, L_list, U_list, T_list, X_list = [], [], [], [], []

print("Running baseline predictions (soup_v3)...", flush=True)
for i in range(0, len(va_sub), chunk_size):
    x, y = batch_va(va_sub[i:i+chunk_size])
    res = predict(x)
    P_list.append(res["prediction"])
    L_list.append(res["lower"])
    U_list.append(res["upper"])
    T_list.append(y)
    X_list.append(x)
    
P = np.concatenate(P_list, 0)
L = np.concatenate(L_list, 0)
U = np.concatenate(U_list, 0)
T = np.concatenate(T_list, 0)
X = np.concatenate(X_list, 0)

c_shipped = (L + U) / 2.0
h_shipped = (U - L) / 2.0
d = np.abs(T[..., :2] - c_shipped[..., :2])
scored_mask = (T[..., :2] != 0.0)

SIGMA = 0.0563870259
DEV = "cuda:0"

z = np.load(f"{B}/_r9/bounds_assets.npz")
LUT_shipped = z["LUT"]
ED = z["ED"]
mid_channels = int(z["w"])

net = _UNet(80, 80, mid_channels).to(DEV).eval()
net.load_state_dict({k[len("w_"):]: torch.from_numpy(z[k].astype(np.float32))
                     for k in z.files if k.startswith("w_")})

w_list = []
with torch.no_grad():
    for i in range(0, len(X), chunk_size):
        xb = torch.from_numpy(X[i:i+chunk_size, ..., :2]).to(DEV)
        yb = torch.from_numpy(P[i:i+chunk_size, ..., :2]).to(DEV)
        ui = torch.cat([_fold(xb), _fold(yb)], 1)
        o = net(ui)
        w = o[:, 40:].reshape(-1, 20, 2, 32, 64).permute(0, 1, 3, 4, 2)
        w_list.append(w.cpu().numpy())
W = np.concatenate(w_list, 0)

with zipfile.ZipFile(f"{B}/submissions/submission_LUTCAL.zip", "r") as zf:
    with zf.open("bounds_assets.npz") as f:
        lutcal_z = np.load(f)
        LUT_cal = lutcal_z["LUT"]

# TASK 1
LUT_b = {b: np.empty_like(LUT_shipped) for b in [1.0, 1.4, 1.8, 2.2]}
unfit_bins = {c: [] for c in range(2)}
bins = np.empty_like(W, dtype=int)

for c in range(2):
    bins[..., c] = np.digitize(W[..., c], ED[c], right=True)
    mask_c = scored_mask[..., c]
    d_c = d[..., c][mask_c]
    med = np.median(d_c)
    
    d_emul = {b: 1.1446 * med * (d_c / med) ** b for b in [1.0, 1.4, 1.8, 2.2]}
    bins_c = bins[..., c][mask_c]
    
    h_grid = torch.linspace(1e-4, 0.12, 4000).cuda()
    
    for bin_idx in range(len(LUT_shipped)):
        bmask = (bins_c == bin_idx)
        n = bmask.sum()
        if n < 200:
            unfit_bins[c].append(bin_idx)
            for b in [1.0, 1.4, 1.8, 2.2]:
                LUT_b[b][bin_idx, c] = LUT_shipped[bin_idx, c]
            continue
            
        for b in [1.0, 1.4, 1.8, 2.2]:
            d_bin = torch.from_numpy(d_emul[b][bmask]).cuda()
            best_g, best_h = -1, -1
            for h in h_grid:
                cov = (d_bin <= h).float().mean()
                g = torch.exp(-2*h / SIGMA) * cov
                if g > best_g:
                    best_g = g
                    best_h = h
            LUT_b[b][bin_idx, c] = best_h.item()

print("Unfit bins (<200 elements):", unfit_bins, flush=True)

res = {"luts": {}}
for c, ch in enumerate(["u", "v"]):
    res["luts"][ch] = []
    for bin_idx in range(len(LUT_shipped)):
        res["luts"][ch].append({
            "bin": bin_idx,
            "shipped": float(LUT_shipped[bin_idx, c]),
            "cal": float(LUT_cal[bin_idx, c]),
            "b1.0": float(LUT_b[1.0][bin_idx, c]),
            "b1.4": float(LUT_b[1.4][bin_idx, c]),
            "b1.8": float(LUT_b[1.8][bin_idx, c]),
            "b2.2": float(LUT_b[2.2][bin_idx, c]),
            "ratio_180_shipped": float(LUT_b[1.8][bin_idx, c] / LUT_shipped[bin_idx, c])
        })
        
res["sensitivity"] = {}
for c, ch in enumerate(["u", "v"]):
    res["sensitivity"][ch] = float(np.median(LUT_b[2.2][:, c] / LUT_b[1.4][:, c]))

def get_g(lut, d_eval):
    ew_g, ew_cov, ew_h = 0, 0, 0
    for c in range(2):
        mask_c = scored_mask[..., c]
        h_new = lut[bins[..., c][mask_c], c]
        dc = d_eval[..., c][mask_c]
        
        inside = (dc <= h_new)
        cov = inside.mean()
        g = (np.exp(-2*h_new/SIGMA) * inside).mean()
        
        ew_g += g / 2.0
        ew_cov += cov / 2.0
        ew_h += h_new.mean() / 2.0
    return ew_g, ew_cov, ew_h

d_emul180 = np.empty_like(d)
for c in range(2):
    mask_c = scored_mask[..., c]
    dc = d[..., c][mask_c]
    med = np.median(dc)
    d_emul180[..., c][mask_c] = 1.1446 * med * (dc / med) ** 1.8

g_shipped_raw, _, _ = get_g(LUT_shipped, d)
g_shipped_emul, _, _ = get_g(LUT_shipped, d_emul180)

res["effects"] = {
    "shipped": {
        "raw": {"g": float(g_shipped_raw)},
        "emul": {"g": float(g_shipped_emul)}
    }
}
for name, lut in zip(["b1.0", "b1.4", "b1.8", "b2.2"], [LUT_b[1.0], LUT_b[1.4], LUT_b[1.8], LUT_b[2.2]]):
    g_raw, cov_raw, h_raw = get_g(lut, d)
    g_emul, cov_emul, h_emul = get_g(lut, d_emul180)
    res["effects"][name] = {
        "raw": {"g": float(g_raw), "R": float(g_raw / g_shipped_raw)},
        "emul": {"g": float(g_emul), "R": float(g_emul / g_shipped_emul)},
        "coverage": float(cov_raw),
        "mean_h": float(h_raw)
    }

G1_u = np.median(LUT_b[1.8][:, 0] / LUT_b[1.0][:, 0])
G1_v = np.median(LUT_b[1.8][:, 1] / LUT_b[1.0][:, 1])
G1 = (G1_u > 1.0) and (G1_v > 1.0)
res["G1"] = {"u": float(G1_u), "v": float(G1_v), "pass": bool(G1)}
res["G2"] = {"u": res["sensitivity"]["u"], "v": res["sensitivity"]["v"], "pass": res["sensitivity"]["u"] <= 1.25 and res["sensitivity"]["v"] <= 1.25}

print(json.dumps(res, indent=2))
with open(f"{B}/r11_results.json", "w") as f:
    json.dump(res, f, indent=2)

if not G1:
    print("G1 FAILED! b=1.8 is narrower than b=1.0. Stop and build nothing.", flush=True)
    sys.exit(0)

print("G1 PASSED! Building artifact...", flush=True)

# Build ZIP
out_zip = f"{B}/submissions/submission_LUTEMU.zip"
with zipfile.ZipFile(f"{B}/_r9/submission_SV2.zip", "r") as zin:
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename == "bounds_assets.npz":
                # replace LUT
                buf = io.BytesIO(zin.read(item.filename))
                z_old = np.load(buf)
                out_buf = io.BytesIO()
                arrays = {k: z_old[k] for k in z_old.files}
                arrays["LUT"] = LUT_b[1.8]
                np.savez(out_buf, **arrays)
                zout.writestr(item, out_buf.getvalue())
            else:
                zout.writestr(item, zin.read(item.filename))

print("Zip built. Running Predict...", flush=True)
# Evaluate zip
sys.path.insert(0, f"{B}/submissions")
import zipfile
# unpack LUTEMU to a tmp dir to run its predict
tmp_dir = f"{B}/tmp_lutemu"
os.makedirs(tmp_dir, exist_ok=True)
with zipfile.ZipFile(out_zip, "r") as zf:
    zf.extractall(tmp_dir)
sys.path.insert(0, tmp_dir)
import submission as sub_lutemu

P_new = sub_lutemu.predict(X[:120])
pred = P_new["prediction"]
lower = P_new["lower"]
upper = P_new["upper"]

assert np.all(np.isfinite(pred)), "Non-finite predictions"
assert np.all(lower <= upper), "Lower > upper"
assert np.all(pred[..., 2] == 0.0), "p != 0"

fallback_u = 0.0129
fallback_v = 0.0098
hu = (upper[..., 0] - lower[..., 0]) / 2.0
hv = (upper[..., 1] - lower[..., 1]) / 2.0
match_u = np.isclose(hu, fallback_u, atol=1e-6)
match_v = np.isclose(hv, fallback_v, atol=1e-6)
assert not np.any(match_u & match_v), "Fallback fired!"

std_hu = np.std(hu)
std_hv = np.std(hv)
res["zip_checks"] = {
    "std_hu": float(std_hu),
    "std_hv": float(std_hv),
    "md5": hashlib.md5(open(out_zip, "rb").read()).hexdigest(),
    "fno_md5": hashlib.md5(open(f"{tmp_dir}/sim_real_fno_fp16.pth", "rb").read()).hexdigest()
}
print(json.dumps(res["zip_checks"], indent=2))

# Timing
t0 = time.time()
for _ in range(5):
    predict(X[:120])
t_old = (time.time() - t0) / 5 / 120 * 1000

t0 = time.time()
for _ in range(5):
    sub_lutemu.predict(X[:120])
t_new = (time.time() - t0) / 5 / 120 * 1000

print(f"Timing SV2: {t_old:.4f} ms, LUTEMU: {t_new:.4f} ms")

