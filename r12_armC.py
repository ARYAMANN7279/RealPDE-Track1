import os, sys, time, json
import numpy as np
import torch
import torch.nn.functional as F

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
sys.path.insert(0, B)
import r12_eval as EV

sys.path.insert(0, f"{B}/code/RealPDEBench")
from realpdebench.model.unet import Unet3d, EMA

DEV = "cuda:0"
EV.selftest()

meta = json.load(open(f"{B}/local_harness/tr_meta.json"))
off, lens, names = meta["off"], meta["lens"], meta["names"]
ntraj = len(lens)
RE = np.array([int(n.split("_")[0]) for n in names])
VT = set(np.where(np.isin(RE, [3750, 5025, 25425, 26700]))[0].tolist())

starts_tr = []
for i in range(ntraj):
    for t0 in range(off[i], off[i]+lens[i]-39):
        if i not in VT:
            starts_tr.append(t0)
starts_tr = np.array(starts_tr)

sub, _ = EV.val_starts(900)

sim_meta = json.load(open(f"{B}/local_harness/sim_meta.json"))
s_off, s_lens = sim_meta["off"], sim_meta["lens"]
s_starts = []
for i in range(len(s_lens)):
    for t0 in range(s_off[i], s_off[i]+s_lens[i]-39):
        s_starts.append(t0)
s_starts = np.array(s_starts)
SIM32 = np.load(f"{B}/local_harness/sim_frames.npy", mmap_mode="r")

def batch_sim(idx):
    w = np.stack([np.asarray(SIM32[s:s+40]) for s in idx])
    z = np.zeros(w.shape[:-1]+(1,), np.float32)
    w = np.concatenate([w, z], -1).astype(np.float32)
    return w[:,:20], w[:,20:]

model = Unet3d(dim=64, out_channels=3, dim_mults=[1,2,4], channels=3, in_time=20, out_time=20).to(DEV)
ema_model = Unet3d(dim=64, out_channels=3, dim_mults=[1,2,4], channels=3, in_time=20, out_time=20).to(DEV)
ema_model.load_state_dict(model.state_dict())
ema = EMA(0.999)

mi = EV.MI.view(1,1,1,1,3).to(DEV)
si = EV.SI.view(1,1,1,1,3).to(DEV)
mt = EV.MT.view(1,1,1,1,3).to(DEV)
st = EV.ST.view(1,1,1,1,3).to(DEV)

def evaluate():
    ema_model.eval()
    def fwd(x):
        xt_norm = (x - mi) / si
        return ema_model(xt_norm) * st + mt
    return EV.eval_model(fwd, sub, bs=12)

num_update_sim = 5000
batch_size = 12
opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_update_sim)

print("Starting Arm C (Sim Pretrain)...")
tr_rng = np.random.default_rng(42)
t0 = time.time()
for step in range(1, num_update_sim + 1):
    model.train()
    idx = tr_rng.choice(s_starts, size=batch_size, replace=False)
    x, y = batch_sim(idx)
    
    # Domain Randomization mask_prob=0.5
    if tr_rng.random() < 0.5:
        x[..., 2] = 0.0
        y[..., 2] = 0.0
        
    xt = torch.from_numpy(x).to(DEV)
    yt = torch.from_numpy(y).to(DEV)
    
    # mult noise 0.1
    noise_x = 1.0 + 0.1 * torch.randn_like(xt)
    xt = xt * noise_x
    noise_y = 1.0 + 0.1 * torch.randn_like(yt)
    yt = yt * noise_y
    
    xt_norm = (xt - mi) / si
    yt_norm = (yt - mt) / st
    
    opt.zero_grad()
    pred_norm = model(xt_norm)
    loss = F.mse_loss(pred_norm, yt_norm)
    loss.backward()
    opt.step()
    sched.step()
    ema.update_model_average(ema_model, model)
    
    if step % 1000 == 0:
        print(f"Sim Step {step} | Loss: {loss.item():.5f}", flush=True)

print("Starting Arm C (Real Finetune)...")
num_update_real = 10000
opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_update_real)
best_dacc, best_step = -100, -1
results = []

for step in range(1, num_update_real + 1):
    model.train()
    idx = tr_rng.choice(starts_tr, size=batch_size, replace=False)
    x, y = EV.window(idx)
    xt = torch.from_numpy(x).to(DEV)
    yt = torch.from_numpy(y).to(DEV)
    
    xt_norm = (xt - mi) / si
    yt_norm = (yt - mt) / st
    
    opt.zero_grad()
    pred_norm = model(xt_norm)
    loss = F.mse_loss(pred_norm, yt_norm)
    loss.backward()
    opt.step()
    sched.step()
    ema.update_model_average(ema_model, model)
    
    if step % 500 == 0:
        r = evaluate()
        dt = time.time() - t0
        print(f"Real Step {step:5d} | Loss: {loss.item():.5f} | rel_l2: {r['rel_l2']:.4f} | tke: {r['tke']:.4f} | mvpe: {r['mvpe']:.4f} | d_acc: {r['d_acc']:.4f} | time: {dt:.1f}s", flush=True)
        results.append((step, r['rel_l2'], r['tke'], r['mvpe'], r['d_acc']))
        if r['d_acc'] > best_dacc:
            best_dacc = r['d_acc']
            best_step = step
            torch.save(ema_model.state_dict(), f"{B}/train_es/r12_unet_sim2real_best.pth")

print(f"Arm C completed. Best d_acc = {best_dacc:.4f} at step {best_step}")
res_dict = {"results": results, "best_step": best_step, "best_dacc": best_dacc}
with open(f"{B}/r12_armC.json", "w") as f:
    json.dump(res_dict, f)
