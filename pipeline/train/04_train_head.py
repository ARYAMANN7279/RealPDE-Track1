"""Train the per-element uncertainty CNN: regress log|error| from features that are
all computable at inference time. Trained only on the released training data.

Note on formulation: optimising the SPS objective directly through a smooth
indicator was tried and learns almost no spatial structure (sd(log h) ~ 0.36,
vanishing gradients at small temperature). Regressing log|error| conditions the
gradients properly; the error -> half-width mapping is then fitted separately in
05_build_lut.py, which is where the SPS trade-off actually lives."""
import json, os, sys
import numpy as np, torch, torch.nn as nn
from importlib.machinery import SourceFileLoader
HERE = os.path.dirname(os.path.abspath(__file__))
C = SourceFileLoader("c", os.path.join(HERE, "00_config.py")).load_module()
C.seed_all()
sys.path.insert(0, HERE)
from head_common import feats, Net, calibrate
dev = "cuda" if torch.cuda.is_available() else "cpu"
d = np.load(os.path.join(C.WORK, "cache.npz"))
P = d["P"].astype(np.float32); Y = d["Y"].astype(np.float32); wt = d["wtraj"]
g = np.load(os.path.join(C.WORK, "spectral_gain.npy"))
Fq = np.fft.rfft(P[..., :2], axis=1) * g[None, :, None, None, :]
P[..., :2] = np.fft.irfft(Fq, n=20, axis=1)
uniq = sorted(set(wt.tolist()))
tr = np.isin(wt, uniq[0::2]); te = ~tr                # trajectory-disjoint
SCM = (Y[..., :2] != 0.0)
ERR = calibrate(P, Y, SCM)
LOGE = np.log(ERR + 1e-6).astype(np.float32)
FT = feats(P); NF = FT.shape[-1]
mu_ = FT[tr].reshape(-1, NF).mean(0); sd_ = FT[tr].reshape(-1, NF).std(0) + 1e-6
FT = (FT - mu_) / sd_
net = Net(NF).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
EP = 26
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EP, eta_min=5e-5)
best = 1e9
for ep in range(EP):
    idx = np.where(tr)[0]; np.random.default_rng(C.SEED + ep).shuffle(idx)
    for i in range(0, len(idx), 8):
        j = idx[i:i+8]
        f = torch.from_numpy(np.ascontiguousarray(FT[j])).to(dev).permute(0,1,4,2,3).reshape(-1,NF,32,64)
        p = net(f).reshape(len(j),20,2,32,64).permute(0,1,3,4,2)
        t = torch.from_numpy(np.ascontiguousarray(LOGE[j])).to(dev)
        m = torch.from_numpy(SCM[j].astype(np.float32)).to(dev)
        loss = ((p - t).abs() * m).sum() / m.sum()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    sch.step()
    if float(loss) < best:
        best = float(loss)
        torch.save({"sd": net.state_dict(), "mu": mu_, "std": sd_, "nf": NF},
                   os.path.join(C.WORK, "head.pth"))
    if ep % 5 == 4: print("  ep %2d  L1 %.4f" % (ep, float(loss)), flush=True)
print("saved head.pth (best L1 %.4f)" % best)
