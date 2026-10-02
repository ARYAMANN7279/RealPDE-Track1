import os, sys, json, subprocess
import numpy as np

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT)
sys.path.insert(0, os.path.join(KIT, "_vendor"))
import importlib.util as iu
sp = iu.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
S = iu.module_from_spec(sp)
sp.loader.exec_module(S)

zips = ["submission_M1_SV3.zip"]

LH = f"{B}/local_harness"
meta = json.load(open(f"{LH}/tr_meta.json"))
off, lens, names = meta["off"], meta["lens"], meta["names"]
ntraj = len(lens)
RE = np.array([int(n.split("_")[0]) for n in names])
VT = set(np.where(np.isin(RE, [3750, 5025, 25425, 26700]))[0].tolist())

X32 = np.load(f"{LH}/tr_frames.npy", mmap_mode="r")
starts_va = []
for i in range(ntraj):
    if i in VT:
        for t0 in range(off[i], off[i] + lens[i] - 39):
            starts_va.append(t0)
starts_va = np.array(starts_va)
rng = np.random.default_rng(0)
va_sub = rng.choice(starts_va, size=min(900, len(starts_va)), replace=False)

def batch_va(idx):
    w = np.stack([np.asarray(X32[s:s+40]) for s in idx])
    z = np.zeros(w.shape[:-1] + (1,), np.float32)
    w = np.concatenate([w, z], -1).astype(np.float32)
    return w[:, :20], w[:, 20:]

for zipname in zips:
    d = f"{B}/_tmp/verify_{zipname.split('.')[0]}"
    os.makedirs(d, exist_ok=True)
    subprocess.run(["unzip", "-qo", f"{B}/submissions/{zipname}", "-d", d], check=True)
    
    sys.path.insert(0, d)
    if "submission" in sys.modules: del sys.modules["submission"]
    import submission as M
    
    M.predict(batch_va(va_sub[:2])[0])
    
    P, LO, UP, T = [], [], [], []
    for i in range(0, len(va_sub), 32):
        x, y = batch_va(va_sub[i:i+32])
        out = M.predict(x)
        if isinstance(out, dict):
            P.append(out["prediction"]); LO.append(out["lower"]); UP.append(out["upper"])
        else:
            P.append(out[:, :, :, :, :2]); LO.append(out[:, :, :, :, 3:5]); UP.append(out[:, :, :, :, 6:8])
        T.append(y[:, :, :, :, :2])
    
    P = np.concatenate(P, 0); T = np.concatenate(T, 0)
    LO = np.concatenate(LO, 0); UP = np.concatenate(UP, 0)
    sys.path.pop(0)

    C = 2
    l2 = float(S.rel_l2_per_sample(P, T, C).mean())
    tk = float(S.tke_rel_l2_per_sample(P, T, C).mean())
    mv = float(S.mvpe_rel_l2_per_sample(P, T).mean())
    l2_s = S.score_error(l2); tk_s = S.score_error(tk); mv_s = S.score_error(mv)
    d_acc = 0.669 * (l2_s - 95.4738) + 0.157 * (tk_s - 75.8957) + 0.170 * (mv_s - 96.0945)
    sps, _ = S.aggregate_sps(P, T, c=C, lower=LO, upper=UP)

    print(f"[{zipname}]")
    print(f"  rel_l2: {l2_s:.4f} | tke: {tk_s:.4f} | mvpe: {mv_s:.4f}")
    print(f"  Local d_acc: {d_acc:+.4f} | Local sps: {sps:.4f} | Local TOTAL: {d_acc + sps:+.4f}")
