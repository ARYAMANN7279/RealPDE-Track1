import os, sys, time, json
import numpy as np
import torch
import torch.nn.functional as F
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
from load_baseline import load_baseline
from submission import _UNet, _fold

# 1. Load Data
meta = json.load(open(f"{LH}/tr_meta.json"))
off, lens, names = meta["off"], meta["lens"], meta["names"]
ntraj = len(lens)
RE = np.array([int(n.split("_")[0]) for n in names])
VT = set(np.where(np.isin(RE, [3750, 5025, 25425, 26700]))[0].tolist())

X32 = np.load(f"{LH}/tr_frames.npy", mmap_mode="r")
starts_tr = []
for i in range(ntraj):
    if i not in VT:
        for t0 in range(off[i], off[i]+lens[i]-39):
            starts_tr.append(t0)
starts_tr = np.array(starts_tr)

rng = np.random.default_rng(1234)

def get_batch(batch_size):
    idx = rng.choice(starts_tr, size=batch_size, replace=False)
    w = np.stack([np.asarray(X32[s:s+40]) for s in idx])
    z = np.zeros(w.shape[:-1]+(1,), np.float32)
    w = np.concatenate([w, z], -1).astype(np.float32)
    x = torch.from_numpy(w[:, :20]).cuda()
    y = torch.from_numpy(w[:, 20:]).cuda()
    return x, y

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

DEV = "cuda:0"
model, _ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device="cpu")
model.load_state_dict(unpack_fp16(f"{B}/train_es/soup_v3_fp16.pth"))
model = model.to(DEV).eval()

MI = torch.tensor([0.154960856, -0.000513992854, 0.0]).to(DEV)
SI = torch.tensor([0.0968056545, 0.015960684, 1.0]).to(DEV)
MT = torch.tensor([0.154962569, -0.000517793698, 0.0]).to(DEV)
ST = torch.tensor([0.0968104079, 0.0159636438, 1.0]).to(DEV)

def fwd(x):
    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.float16):
            return (model((x - MI)/SI)*ST + MT).float()

# Load shipped weights
z = np.load(f"{B}/_r9/bounds_assets.npz")
mid_channels = int(z["w"])
shipped_net = _UNet(80, 80, mid_channels)
shipped_net.load_state_dict({k[len("w_"):]: torch.from_numpy(z[k].astype(np.float32))
                             for k in z.files if k.startswith("w_")})
shipped_net = shipped_net.to(DEV)

# TASK A
print("Starting Task A...")
torch.manual_seed(1234)
net_A = _UNet(80, 80, mid_channels).to(DEV)
net_A.load_state_dict(shipped_net.state_dict())

# Freeze all except out[40:80]
for name, p in net_A.named_parameters():
    if name not in ["out.weight", "out.bias"]:
        p.requires_grad = False
    else:
        p.requires_grad = True

opt_A = torch.optim.AdamW(net_A.parameters(), lr=1e-3, weight_decay=1e-6)
sched_A = torch.optim.lr_scheduler.CosineAnnealingLR(opt_A, T_max=8000)

for step in range(8000):
    x, y = get_batch(16)
    yb = fwd(x)
    yb[..., 2] = 0.0
    
    ui = torch.cat([_fold(x[..., :2]), _fold(yb[..., :2])], 1)
    
    e = torch.abs(y[..., :2] - yb[..., :2]) # N, 20, 32, 64, 2
    target_log = torch.log(e + 1e-6)
    target_w = _fold(target_log) # N, 40, 32, 64
    
    o = net_A(ui)
    w = o[:, 40:] # N, 40, 32, 64
    
    mask = _fold((y[..., :2] != 0.0).float())
    
    loss = F.huber_loss(w * mask, target_w * mask, delta=1.0, reduction='sum') / mask.sum().clamp(min=1)
    
    opt_A.zero_grad()
    loss.backward()
    
    # Zero gradients for out[0:40]
    net_A.out.weight.grad[:40] = 0.0
    net_A.out.bias.grad[:40] = 0.0
    
    opt_A.step()
    sched_A.step()
    
    if (step + 1) % 1000 == 0:
        print(f"Task A step {step+1}/8000 loss: {loss.item():.4f}", flush=True)

torch.save(net_A.state_dict(), f"{B}/train_es/r10_taskA.pth")
print("Saved Task A.")

# TASK B
print("Starting Task B...")
torch.manual_seed(1234)
net_B = _UNet(80, 40, mid_channels).to(DEV)
opt_B = torch.optim.AdamW(net_B.parameters(), lr=1e-3, weight_decay=1e-6)
sched_B = torch.optim.lr_scheduler.CosineAnnealingLR(opt_B, T_max=20000)

for step in range(20000):
    x, y = get_batch(16)
    yb = fwd(x)
    yb[..., 2] = 0.0
    
    ui = torch.cat([_fold(x[..., :2]), _fold(yb[..., :2])], 1)
    
    e = torch.abs(y[..., :2] - yb[..., :2])
    target_log = torch.log(e + 1e-6)
    target_w = _fold(target_log)
    
    w = net_B(ui)
    mask = _fold((y[..., :2] != 0.0).float())
    
    loss = F.huber_loss(w * mask, target_w * mask, delta=1.0, reduction='sum') / mask.sum().clamp(min=1)
    
    opt_B.zero_grad()
    loss.backward()
    opt_B.step()
    sched_B.step()
    
    if (step + 1) % 2000 == 0:
        print(f"Task B step {step+1}/20000 loss: {loss.item():.4f}", flush=True)

torch.save(net_B.state_dict(), f"{B}/train_es/r10_taskB.pth")
print("Saved Task B.")
