import os, sys, time
import numpy as np
import torch
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
sys.path.insert(0, f"{B}/_r9")
import submission
from submission import predict

def custom_load_baseline(path, device):
    from load_baseline import load_baseline
    model, _ = load_baseline(path, device="cpu")
    return model, None

submission.load_baseline = custom_load_baseline

x = np.zeros((2, 20, 26, 38, 70), dtype=np.float32)
print("Predicting...", flush=True)
res = predict(x)
print("Success!", flush=True)
