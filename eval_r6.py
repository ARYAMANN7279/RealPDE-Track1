import sys, os, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
sys.path.append(f"{B}/train_es")
from scoring import get_score

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

def flat(z): return torch.from_numpy(np.ascontiguousarray(z)).permute(0, 1, 4, 2, 3).reshape(z.shape[0], -1, 32, 64)

d = np.load(f"{B}/train_es/cache_lohihonest_stride3.npz", mmap_mode="r")
import json
meta = json.load(open(f"{B}/local_harness/tr_meta.json"))
RE = np.array([int(meta["names"][i].split("_")[0]) for i in d["wt"]])
VAL = np.isin(RE, [3750, 5025, 25425, 26700])

XI = d["XI"][VAL]
PR = d["PR"][VAL]
RS = d["RS"][VAL]

# TKE requires the actual ground truth and prediction!
# Wait, RS = Ground Truth - Prediction. So Truth = RS + Prediction!
TRU = PR + RS

DEV = "cuda:2"
x = torch.cat([flat(XI), flat(PR)], 1).to(DEV)

for w in [96, 128]:
    try:
        net = UNet(80, 2, w).to(DEV)
        ck = torch.load(f"{B}/train_es/mean_predictor_W{w}_d01.pth", map_location=DEV)
        net.load_state_dict(ck["sd"])
        net.eval()
        with torch.no_grad():
            o = net(x).cpu().numpy() # Shape: (N, 2, 32, 64)
            # o is the time-mean correction!
            # The corrected prediction is PR + o
            o_expand = np.expand_dims(o.transpose(0, 2, 3, 1), axis=1) # (N, 1, 32, 64, 2)
            o_expand = np.broadcast_to(o_expand, PR.shape)
            
            PR_corr = PR + o_expand
            
            # Now calculate TKE and MVPE!
            base_tke = 0; base_mvpe = 0
            corr_tke = 0; corr_mvpe = 0
            
            for i in range(len(PR)):
                sb = get_score(PR[i], TRU[i])
                sc = get_score(PR_corr[i], TRU[i])
                base_tke += sb["tke"]; base_mvpe += sb["mvpe"]
                corr_tke += sc["tke"]; corr_mvpe += sc["mvpe"]
                
            N = len(PR)
            print(f"W{w} | d01")
            print(f"Base TKE: {base_tke/N:.4f}, Corr TKE: {corr_tke/N:.4f}, Delta: {corr_tke/N - base_tke/N:.6f}")
            print(f"Base MVPE: {base_mvpe/N:.4f}, Corr MVPE: {corr_mvpe/N:.4f}")
            f = (base_mvpe/N - corr_mvpe/N) / (base_mvpe/N)
            print(f"f = {f:.4f}")
    except Exception as e:
        print(f"Error on W{w}: {e}")
