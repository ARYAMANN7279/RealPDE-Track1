"""Self-test of the cache734 pipeline (CPU only).
1. my SHIFT85 cache vs _agent2's (built by eval_zip.py on 31 Aug from the same zip, same windows)
2. bound identities the record claims: TIER2D == TMEAN (§46.6), FP16 ~ TMEAN, ENSFASTv3 vs TMEAN,
   SCREEN vs SV2 (same bounds file, different backbone -> bounds differ through the U-Net input)
"""
import numpy as np

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
D = f"{B}/agents/bounds/cache734"
A2 = f"{B}/_agent2"


def cmp(name, x, y):
    dd = np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64))
    print(f"{name:38s} max|diff| {dd.max():.3e}  mean {dd.mean():.3e}  frac>1e-6 {(dd > 1e-6).mean():.2e}",
          flush=True)


for k in ("C", "HD", "HU"):
    cmp(f"SHIFT85 {k}: mine vs _agent2", np.load(f"{D}/{k}_SHIFT85.npy"), np.load(f"{A2}/{k}_SHIFT85.npy"))
cmp("SHIFT85 P: mine vs _agent2 PRED", np.load(f"{D}/P_SHIFT85.npy"), np.load(f"{A2}/PRED.npy"))


def lu(t):
    P = np.load(f"{D}/P_{t}.npy")
    c = P + np.load(f"{D}/C_{t}.npy")
    return c - np.load(f"{D}/HD_{t}.npy"), c + np.load(f"{D}/HU_{t}.npy"), P


for a, b in (("TIER2D", "TMEAN"), ("FP16", "TMEAN"), ("ENSFASTv3", "TMEAN"), ("SCREEN", "SV2")):
    la, ua, pa = lu(a)
    lb, ub, pb = lu(b)
    cmp(f"{a} vs {b} lower", la, lb)
    cmp(f"{a} vs {b} upper", ua, ub)
    cmp(f"{a} vs {b} prediction", pa, pb)
    del la, ua, pa, lb, ub, pb
