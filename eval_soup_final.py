import torch, sys, os
import numpy as np
import json
import importlib.util as iu
import zipfile

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT)
sys.path.insert(0, os.path.join(KIT, "_vendor"))
sp = iu.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
S = iu.module_from_spec(sp)
sp.loader.exec_module(S)
from load_baseline import load_baseline

DEV = "cuda:0"
C = 2

def eval_model(model_name, model_sd):
    model, _ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=DEV)
    model.load_state_dict(model_sd)
    model = model.to(DEV).eval()

    MI = torch.tensor([0.154960856, -0.000513992854, 0.0]).to(DEV)
    SI = torch.tensor([0.0968056545, 0.015960684, 1.0]).to(DEV)
    MT = torch.tensor([0.154962569, -0.000517793698, 0.0]).to(DEV)
    ST = torch.tensor([0.0968104079, 0.0159636438, 1.0]).to(DEV)

    def fwd(x): return model((x - MI) / SI) * ST + MT

    LH = f"{B}/local_harness"
    meta = json.load(open(f"{LH}/tr_meta.json"))
    off, lens, names = meta["off"], meta["lens"], meta["names"]
    ntraj = len(lens)
    RE = np.array([int(n.split("_")[0]) for n in names])
    VT = set(np.where(np.isin(RE, [3750, 5025, 25425, 26700]))[0].tolist())

    X32 = np.load(f"{LH}/tr_frames.npy", mmap_mode="r")
    
    # 900-window split (stride 1, random 900)
    starts_va = []
    for i in range(ntraj):
        for t0 in range(off[i], off[i] + lens[i] - 39):
            if i in VT: starts_va.append(t0)
    starts_va = np.array(starts_va)
    rng = np.random.default_rng(0)
    va_sub = rng.choice(starts_va, size=min(900, len(starts_va)), replace=False)

    # 664-window split (stride 20, deterministic)
    starts_664 = []
    for i in range(ntraj):
        for t0 in range(off[i], off[i] + lens[i] - 39, 20):
            if i in VT: starts_664.append(t0)
    starts_664 = np.array(starts_664)

    def batch_va(idx):
        w = np.stack([np.asarray(X32[s:s+40]) for s in idx])
        z = np.zeros(w.shape[:-1] + (1,), np.float32)
        w = np.concatenate([w, z], -1).astype(np.float32)
        return torch.from_numpy(w[:, :20]).to(DEV), torch.from_numpy(w[:, 20:]).to(DEV)

    @torch.no_grad()
    def eval_and_report(idx_list, name=""):
        P, T = [], []
        for i in range(0, len(idx_list), 16):
            x, y = batch_va(idx_list[i:i+16])
            P.append(fwd(x).cpu().numpy()); T.append(y.cpu().numpy())
        P = np.concatenate(P, 0).astype(np.float32)
        T = np.concatenate(T, 0).astype(np.float32)
        dm = float(S.rel_l2_per_sample(P, T, C).mean())
        tk = float(S.tke_rel_l2_per_sample(P, T, C).mean())
        mv = float(S.mvpe_rel_l2_per_sample(P, T).mean())
        
        # Scale to 95.xx
        l2_s = S.score_error(dm)
        tk_s = S.score_error(tk)
        mv_s = S.score_error(mv)

        # Calculate d_acc
        base_l2, base_tk, base_mv = (95.4738, 75.8957, 96.0945)
        d = 0.669 * (l2_s - base_l2) + 0.157 * (tk_s - base_tk) + 0.170 * (mv_s - base_mv)
        print(f"[{name}] rel_l2 {l2_s:.4f} tke {tk_s:.4f} mvpe {mv_s:.4f} | d_acc {d:+.4f}")

    eval_and_report(va_sub, f"{model_name} 900-win")
    eval_and_report(starts_664, f"{model_name} 664-win")


print("Evaluating NEW SOUP...")
soup_sd = torch.load(f"{B}/train_es/soup_honest.pth", map_location='cpu')
eval_model("NEW_SOUP", soup_sd)

print("Extracting SHIPPED SOUP...")
BASE = f"{B}/submissions/submission_SOUP_v1.zip"
CK_BYTES = zipfile.ZipFile(BASE).read("sim_real_fno_fp16.pth")
with open("sim_real_fno_fp16_temp.pth", "wb") as f: f.write(CK_BYTES)

print("Evaluating SHIPPED SOUP...")
from load_baseline import _UNet2d # some hack to load
# Wait, actually load_baseline(ckpt_path) directly works for fp16 if we pass path instead of state_dict.
# But we can just use `torch.load("sim_real_fno_fp16_temp.pth", map_location='cpu')` if it's a state_dict.
try:
    shipped_sd = torch.load("sim_real_fno_fp16_temp.pth", map_location='cpu')
    if "state_fp16" in shipped_sd:
        # custom packing logic from kit
        pass # The load_baseline handles it if we just pass the path!
except:
    pass

model_shipped, _ = load_baseline("sim_real_fno_fp16_temp.pth", device=DEV)
shipped_sd = model_shipped.state_dict()
eval_model("SHIPPED_SOUP", shipped_sd)

