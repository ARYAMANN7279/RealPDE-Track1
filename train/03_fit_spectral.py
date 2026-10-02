"""Fit the per-frequency TEMPORAL gain that restores the energy the FNO loses.
Amplitude only; phase is untouched. Fitted on the released training data."""
import os, numpy as np
from importlib.machinery import SourceFileLoader
HERE = os.path.dirname(os.path.abspath(__file__))
C = SourceFileLoader("c", os.path.join(HERE, "00_config.py")).load_module()
C.seed_all()
d = np.load(os.path.join(C.WORK, "cache.npz"))
P = d["P"].astype(np.float32); Y = d["Y"].astype(np.float32)
Fp = np.fft.rfft(P[..., :2], axis=1); Fy = np.fft.rfft(Y[..., :2], axis=1)
g = np.sqrt((np.abs(Fy)**2).mean(axis=(0,2,3)) /
            np.maximum((np.abs(Fp)**2).mean(axis=(0,2,3)), 1e-20))
g = np.clip(g, 0.5, 2.0).astype(np.float32)
np.save(os.path.join(C.WORK, "spectral_gain.npy"), g)
print("temporal gain (u):", " ".join("%.3f" % x for x in g[:, 0]))
print("saved spectral_gain.npy  shape", g.shape)
