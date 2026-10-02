import torch, sys, os
import numpy as np
import json
import importlib.util as iu

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT)
sys.path.insert(0, os.path.join(KIT, "_vendor"))

sp = iu.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
S = iu.module_from_spec(sp)
sp.loader.exec_module(S)
from load_baseline import load_baseline

print('Averaging 7 members...')
paths = [
    f"{B}/train_es/ftaug_mem1_1e5_08.pth",
    f"{B}/train_es/ftaug_mem2_1e5_05.pth",
    f"{B}/train_es/ftaug_mem3_1e5_10.pth",
    f"{B}/train_es/ftaug_mem4_3e5_08.pth",
    f"{B}/train_es/ftaug_mem5_3e5_05.pth",
    f"{B}/train_es/ftaug_mem6_3e5_10.pth",
    f"{B}/train_es/ftaug_mem7_1e5_08.pth"
]
sds = [torch.load(p, map_location='cpu') for p in paths]
soup_sd = {}
for k in sds[0].keys():
    soup_sd[k] = sum(sd[k] for sd in sds) / len(sds)

soup_path = f"{B}/train_es/soup_honest.pth"
torch.save(soup_sd, soup_path)
print('Saved soup_honest.pth')

DEV = "cuda:0"
C = 2
model, _ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=DEV)
model.load_state_dict(soup_sd)
model = model.to(DEV).eval()

# Normalization constants (from ftmv.py)
MI = torch.tensor([0.154960856, -0.000513992854, 0.0]).to(DEV)
SI = torch.tensor([0.0968056545, 0.015960684, 1.0]).to(DEV)
MT = torch.tensor([0.154962569, -0.000517793698, 0.0]).to(DEV)
ST = torch.tensor([0.0968104079, 0.0159636438, 1.0]).to(DEV)

def fwd(x):
    return model((x - MI) / SI) * ST + MT

print('Loading evaluation data...')
LH = f"{B}/local_harness"
meta = json.load(open(f"{LH}/tr_meta.json"))
off = meta["off"]
lens = meta["lens"]
names = meta["names"]
ntraj = len(lens)
RE = np.array([int(n.split("_")[0]) for n in names])
VT = set(np.where(np.isin(RE, [3750, 5025, 25425, 26700]))[0].tolist())

# Load data (like in ftmv.py)
X32 = np.load(f"{LH}/tr_frames.npy", mmap_mode="r")

starts_va = []
for i in range(ntraj):
    for t0 in range(off[i], off[i] + lens[i] - 39):
        if i in VT:
            starts_va.append(t0)
starts_va = np.array(starts_va)

# Evaluate on exactly 900-window protocol (same as ftmv.py eval)
rng = np.random.default_rng(0)
va_sub = rng.choice(starts_va, size=min(900, len(starts_va)), replace=False)

def batch_va(idx):
    w = np.stack([np.asarray(X32[s:s+40]) for s in idx])
    z = np.zeros(w.shape[:-1] + (1,), np.float32)
    w = np.concatenate([w, z], -1).astype(np.float32)
    return torch.from_numpy(w[:, :20]).to(DEV), torch.from_numpy(w[:, 20:]).to(DEV)

@torch.no_grad()
def evaluate_indices(idx_list):
    P = []
    T = []
    for i in range(0, len(idx_list), 16):
        x, y = batch_va(idx_list[i:i+16])
        P.append(fwd(x).cpu().numpy())
        T.append(y.cpu().numpy())
    P = np.concatenate(P, 0).astype(np.float32)
    T = np.concatenate(T, 0).astype(np.float32)
    dm = S.rel_l2_per_sample(P, T, C)
    tk = S.tke_rel_l2_per_sample(P, T, C)
    mv = S.mvpe_rel_l2_per_sample(P, T)
    return float(dm.mean()), float(tk.mean()), float(mv.mean())

def eval_and_report(idx_list, name=""):
    dm, tk, mv = evaluate_indices(idx_list)
    base_l2, base_tk, base_mv = (95.4738, 75.8957, 96.0945) # Taken from ftmv.py baseline log
    
    # Corrected MV
    d = 0.669 * (dm - base_l2) + 0.157 * (tk - base_tk) + 0.170 * (mv - base_mv)
    print(f"[{name}] rel_l2 {dm:.4f} tke {tk:.4f} mvpe {mv:.4f} | d_acc {d:+.4f}")

print("Evaluating SOUP on 900-window subset...")
eval_and_report(va_sub, "SOUP 900-win")

# Evaluate on 664-window split (as per instructions)
# Let's see what the 664-window split is. It's probably the entire `starts_va` for `re_lohi`.
# Wait, len(starts_va) = 664? Let's check!
print(f"len(starts_va) = {len(starts_va)}")
eval_and_report(starts_va, "SOUP 664-win")

# We also need to evaluate the "shipped" soup to compare.
print("Evaluating SHIPPED SOUP...")
shipped_path = f"{B}/train_es/w15lr3_fp16.pth" # Wait, where is the shipped soup?
# "The shipped soup is worth +0.6526 vs +0.0244 for the best single fine-tune"
# Wait, the shipped soup is local_harness/soup.pth or in submission_SOUP_v1.zip.
# We can load soup_v2.pth or soup.pth? The instruction says: "evaluate the soup against the shipped one"
