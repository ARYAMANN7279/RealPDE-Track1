import json, sys, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
LH = f"{B}/local_harness"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT)
import scoring

class _Blk(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3, padding=1)
        self.c2 = nn.Conv2d(o, o, 3, padding=1)
        self.n1 = nn.GroupNorm(8, o)
        self.n2 = nn.GroupNorm(8, o)

    def forward(self, x):
        x = F.gelu(self.n1(self.c1(x)))
        return F.gelu(self.n2(self.c2(x)))

class _UNet(nn.Module):
    def __init__(self, ci, co, w=64):
        super().__init__()
        self.e1 = _Blk(ci, w)
        self.e2 = _Blk(w, 2 * w)
        self.e3 = _Blk(2 * w, 4 * w)
        self.b = _Blk(4 * w, 4 * w)
        self.d3 = _Blk(8 * w, 2 * w)
        self.d2 = _Blk(4 * w, w)
        self.d1 = _Blk(2 * w, w)
        self.out = nn.Conv2d(w, co, 1)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.b(self.pool(e3))
        u = F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.d3(torch.cat([u, e3], 1))
        u = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([u, e2], 1))
        u = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.d1(torch.cat([u, e1], 1))
        return self.out(d1)

def _fold(a5):
    return a5.permute(0, 1, 4, 2, 3).reshape(a5.shape[0], -1, a5.shape[2], a5.shape[3])

DEV = "cuda:0"
d = np.load(f"{B}/train_es/cache_soup.npz")
wt = d["wt"]
meta = json.load(open(f"{LH}/tr_meta.json"))
RE = np.array([int(meta["names"][i].split("_")[0]) for i in wt])
EV = np.isin(RE, [3750, 5025, 25425, 26700])
XI = d["XI"][EV]
PR = d["PR"][EV]
RS = d["RS"][EV]
TARGET = PR + RS

z = np.load(f"{B}/train_es/bounds_assets_ASYM_W96_a85.npz")
net = _UNet(80, 120, int(z["w"]))
net.load_state_dict({k[2:]: torch.from_numpy(z["w_" + k[2:]].astype(np.float32)) for k in z.files if k.startswith("w_")})
net = net.to(DEV).eval()
lut_d = torch.from_numpy(np.ascontiguousarray(z["LUT_D"])).to(DEV)
ed_d = [torch.from_numpy(np.ascontiguousarray(z["ED_D"][ci])).to(DEV) for ci in (0, 1)]
lut_u = torch.from_numpy(np.ascontiguousarray(z["LUT_U"])).to(DEV)
ed_u = [torch.from_numpy(np.ascontiguousarray(z["ED_U"][ci])).to(DEV) for ci in (0, 1)]
alpha = float(z["alpha"])

print("Predicting bounds for validation set...")
n = XI.shape[0]
lower = np.empty_like(PR)
upper = np.empty_like(PR)
lower_clipped = np.empty_like(PR)
upper_clipped = np.empty_like(PR)

CH = 16
for i in range(0, n, CH):
    j = min(i + CH, n)
    with torch.no_grad():
        xi = torch.from_numpy(np.ascontiguousarray(XI[i:j])).to(DEV)
        pr = torch.from_numpy(np.ascontiguousarray(PR[i:j])).to(DEV)
        o = net(torch.cat([_fold(xi), _fold(pr)], 1))
        c = o[:, :40].reshape(-1, 20, 2, 32, 64).float()
        w_d = o[:, 40:80].reshape(-1, 20, 2, 32, 64).float()
        w_u = o[:, 80:].reshape(-1, 20, 2, 32, 64).float()
        hd = torch.empty_like(w_d)
        hu = torch.empty_like(w_u)
        for ci in (0, 1):
            hd[:, :, ci] = lut_d[torch.bucketize(w_d[:, :, ci], ed_d[ci], right=True), ci]
            hu[:, :, ci] = lut_u[torch.bucketize(w_u[:, :, ci], ed_u[ci], right=True), ci]
        
        # Original logic
        ctr = pr.permute(0, 1, 4, 2, 3) + alpha * c
        lo = (ctr - hd).permute(0, 1, 3, 4, 2).cpu().numpy()
        up = (ctr + hu).permute(0, 1, 3, 4, 2).cpu().numpy()
        lower[i:j] = lo
        upper[i:j] = up

        # Clipped logic
        c_shift = torch.clamp(alpha * c, min=-hu, max=hd)
        ctr_clipped = pr.permute(0, 1, 4, 2, 3) + c_shift
        lo_clipped = (ctr_clipped - hd).permute(0, 1, 3, 4, 2).cpu().numpy()
        up_clipped = (ctr_clipped + hu).permute(0, 1, 3, 4, 2).cpu().numpy()
        lower_clipped[i:j] = lo_clipped
        upper_clipped[i:j] = up_clipped

print("Calculating SPS (Original)...")
E_orig, _ = scoring.aggregate_sps(PR, TARGET, c=2, lower=lower, upper=upper)
print("Calculating SPS (Clipped)...")
E_clip, _ = scoring.aggregate_sps(PR, TARGET, c=2, lower=lower_clipped, upper=upper_clipped)

BANK = 78.4566
W_OURS = 0.684593
dE = E_clip - E_orig
# Wait, what's the formula to go from dE to fin? 
# In stack_eval_asym_v5: BANK + 0.24737*100*W_OURS*dE
dFin = 0.24737 * 100 * W_OURS * dE
print(f"Original SPS: {E_orig:.5f}")
print(f"Clipped SPS:  {E_clip:.5f}")
print(f"dE:           {dE:+.6f}")
print(f"dFinal:       {dFin:+.4f}")
