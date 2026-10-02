import sys, os, time, argparse, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=2)
ap.add_argument("--width", type=int, default=96)
ap.add_argument("--drop", type=float, default=0.0)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--tag", required=True)
a = ap.parse_args()

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
DEV = f"cuda:{a.gpu}"

class Blk(nn.Module):
    def __init__(self, i, o, drop=0.0):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3, padding=1)
        self.c2 = nn.Conv2d(o, o, 3, padding=1)
        self.n1 = nn.GroupNorm(8, o)
        self.n2 = nn.GroupNorm(8, o)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
    def forward(self, x):
        return self.drop(F.gelu(self.n2(self.c2(F.gelu(self.n1(self.c1(x)))))))

class UNet(nn.Module):
    def __init__(self, ci, co, w=96, drop=0.0):
        super().__init__()
        self.e1 = Blk(ci, w, drop); self.e2 = Blk(w, 2 * w, drop); self.e3 = Blk(2 * w, 4 * w, drop); self.b = Blk(4 * w, 4 * w, drop)
        self.d3 = Blk(8 * w, 2 * w, drop); self.d2 = Blk(4 * w, w, drop); self.d1 = Blk(2 * w, w, drop)
        self.out = nn.Conv2d(w, co, 1); self.pool = nn.AvgPool2d(2)
    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2)); b = self.b(self.pool(e3))
        u = F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False); d3 = self.d3(torch.cat([u, e3], 1))
        u = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False); d2 = self.d2(torch.cat([u, e2], 1))
        u = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False); d1 = self.d1(torch.cat([u, e1], 1))
        return self.out(d1)

net = UNet(80, 2, a.width, a.drop).to(DEV)

d = np.load(f"{B}/train_es/cache_lohihonest_stride3.npz", mmap_mode="r")
XI = d["XI"]; PR = d["PR"]; RS = d["RS"]
import json
meta = json.load(open(f"{B}/local_harness/tr_meta.json"))
RE = np.array([int(meta["names"][i].split("_")[0]) for i in d["wt"]])
VAL = np.isin(RE, [3750, 5025, 25425, 26700])
TR = ~VAL

def flat(z):
    return torch.from_numpy(np.ascontiguousarray(z)).permute(0, 1, 4, 2, 3).reshape(z.shape[0], -1, 32, 64)

idx_tr = np.where(TR)[0]
idx_val = np.where(VAL)[0]

opt = torch.optim.AdamW(net.parameters(), lr=5e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4, total_steps=a.epochs * (len(idx_tr)//32 + 1), pct_start=0.1)

BATCH = 32

for ep in range(1, a.epochs + 1):
    net.train()
    iva = np.random.permutation(idx_tr)
    loss_sum = 0
    t0 = time.time()
    for i in range(0, len(iva), BATCH):
        ix = iva[i:i+BATCH]
        x = torch.cat([flat(XI[ix]), flat(PR[ix])], 1).to(DEV)
        y = torch.from_numpy(RS[ix].mean(axis=1)).permute(0, 3, 1, 2).float().to(DEV)
        o = net(x)
        loss = F.mse_loss(o, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        loss_sum += loss.item()
        
    net.eval()
    val_loss = 0
    with torch.no_grad():
        for i in range(0, min(1000, len(idx_val)), BATCH):
            ix = idx_val[i:i+BATCH]
            x = torch.cat([flat(XI[ix]), flat(PR[ix])], 1).to(DEV)
            y = torch.from_numpy(RS[ix].mean(axis=1)).permute(0, 3, 1, 2).float().to(DEV)
            o = net(x)
            val_loss += F.mse_loss(o, y).item()
            
    print(f"[{a.tag}] Ep {ep} | Train: {loss_sum/(len(iva)//BATCH):.6f} | Val: {val_loss/(1000//BATCH):.6f} | Time: {time.time()-t0:.1f}s", flush=True)

torch.save({"sd": net.state_dict(), "w": a.width, "mu": 0, "sd_": 1}, f"{B}/train_es/mean_predictor_{a.tag}.pth")
print(f"[{a.tag}] Saved mean_predictor_{a.tag}.pth", flush=True)
