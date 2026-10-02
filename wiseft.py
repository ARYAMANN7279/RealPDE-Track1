import os, sys, time, json
import numpy as np, torch

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
LH = f"{B}/local_harness"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"

sys.path.insert(0, KIT)
sys.path.insert(0, os.path.join(KIT, "_vendor"))
import importlib.util as iu
sp = iu.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
S = iu.module_from_spec(sp)
sp.loader.exec_module(S)

from load_baseline import load_baseline

DEV = "cuda:0"
C = 2

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
    return torch.from_numpy(w[:,:20]).to(DEV), torch.from_numpy(w[:,20:]).to(DEV)

MI = torch.tensor([0.154960856, -0.000513992854, 0.0]).to(DEV)
SI = torch.tensor([0.0968056545, 0.015960684, 1.0]).to(DEV)
MT = torch.tensor([0.154962569, -0.000517793698, 0.0]).to(DEV)
ST = torch.tensor([0.0968104079, 0.0159636438, 1.0]).to(DEV)

model, _ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=DEV)
model = model.to(DEV)

def fwd(x): 
    return model((x - MI)/SI)*ST + MT

@torch.no_grad()
def evaluate():
    model.eval()
    P = []
    T = []
    for i in range(0, len(va_sub), 16):
        x, y = batch_va(va_sub[i:i+16])
        P.append(fwd(x).cpu().numpy())
        T.append(y.cpu().numpy())
    if not P: return (0, 0, 0)
    P = np.concatenate(P, 0).astype(np.float32)
    T = np.concatenate(T, 0).astype(np.float32)
    dm = S.rel_l2_per_sample(P, T, C)
    tk = S.tke_rel_l2_per_sample(P, T, C)
    mv = S.mvpe_rel_l2_per_sample(P, T)
    return (S.score_error(float(dm.mean())), S.score_error(float(tk.mean())), S.score_error(float(mv.mean())))

MV = dict(rel_l2=0.669, tke=0.157, mvpe=0.170)

def unpack_fp16(path):
    raw = torch.load(path, map_location="cpu")
    if "state_fp16" not in raw:
        return raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    complex_keys = set(raw["complex_keys"])
    sd = {}
    for k, v in raw["state_fp16"].items():
        if k in complex_keys:
            sd[k] = torch.view_as_complex(v.float())
        elif torch.is_tensor(v) and v.dtype == torch.float16:
            sd[k] = v.float()
        else:
            sd[k] = v
    return sd

def pack_fp16(sd):
    packed, complex_keys = {}, []
    for k, v in sd.items():
        if torch.is_tensor(v) and v.is_complex():
            packed[k] = torch.view_as_real(v).half()
            complex_keys.append(k)
        elif torch.is_tensor(v) and torch.is_floating_point(v):
            packed[k] = v.half()
        else:
            packed[k] = v
    return {"state_fp16": packed, "complex_keys": complex_keys}

base_path = f"{B}/data/comp_real/sim_real_fno.pth"
base_sd = unpack_fp16(base_path)
base_keys = set(base_sd.keys())

fts = [
    ("soup_v3", f"{B}/train_es/soup_v3_fp16.pth"),
    ("soup_v2", f"{B}/local_harness/soup_v2_fp16.pth")
]

alphas = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00]
results = {}

for name, path in fts:
    print(f"=== Process {name} ===", flush=True)
    ft_sd = unpack_fp16(path)
    ft_keys = set(ft_sd.keys())
    if base_keys != ft_keys:
        print(f"ERROR: Key sets differ for {name}!")
        sys.exit(1)
        
    max_diff = 0.0
    for k in base_keys:
        if base_sd[k].is_floating_point() or base_sd[k].is_complex():
            # Keep the base dtype (complex64 or float32)
            bv = base_sd[k].clone()
            fv = ft_sd[k].to(bv.dtype)
            blend_val = (1.0 - 1.0)*bv + 1.0*fv
            diff = (blend_val - fv).abs().max().item()
            if diff > max_diff:
                max_diff = diff
    
    if max_diff > 1e-6:
        print(f"ERROR: alpha=1.0 self-test failed! Max diff = {max_diff}")
        sys.exit(1)
    
    temp_res = {}
    for a in alphas:
        blend_sd = {}
        for k in base_keys:
            if base_sd[k].is_floating_point() or base_sd[k].is_complex():
                bv = base_sd[k].clone()
                fv = ft_sd[k].to(bv.dtype)
                blend_sd[k] = (1.0 - a)*bv + a*fv
            else:
                blend_sd[k] = ft_sd[k].clone()
        
        model.load_state_dict(blend_sd)
        rel_l2, tke, mvpe = evaluate()
        print(f"  alpha = {a:.2f} | rel_l2: {rel_l2:.4f}, tke: {tke:.4f}, mvpe: {mvpe:.4f}", flush=True)
        temp_res[a] = (rel_l2, tke, mvpe)
        
    base_rel, base_tke, base_mvpe = temp_res[1.0]
    
    run_res = []
    best_dacc = -999.0
    best_alpha = None
    
    for a in alphas:
        r, t, m = temp_res[a]
        d_acc = MV["rel_l2"]*(r - base_rel) + MV["tke"]*(t - base_tke) + MV["mvpe"]*(m - base_mvpe)
        run_res.append({"alpha": a, "rel_l2": r, "tke": t, "mvpe": m, "d_acc_vs_alpha1": d_acc})
        if d_acc > best_dacc:
            best_dacc = d_acc
            best_alpha = a
            
    print(f"  Best alpha for {name} is {best_alpha:.2f} with d_acc_vs_alpha1 = {best_dacc:+.4f}", flush=True)
    
    save_sd = {}
    for k in base_keys:
        if base_sd[k].is_floating_point() or base_sd[k].is_complex():
            bv = base_sd[k].clone()
            fv = ft_sd[k].to(bv.dtype)
            save_sd[k] = (1.0 - best_alpha)*bv + best_alpha*fv
        else:
            save_sd[k] = ft_sd[k].clone()
            
    packed_sd = pack_fp16(save_sd)
    out_path = f"{B}/train_es/wiseft_{name.split('_')[1]}_a{int(best_alpha*100):03d}.pth"
    torch.save(packed_sd, out_path)
    print(f"  Saved best to {out_path}", flush=True)
    results[name] = run_res

with open(f"{B}/wiseft_results.json", "w") as f:
    json.dump(results, f, indent=2)
