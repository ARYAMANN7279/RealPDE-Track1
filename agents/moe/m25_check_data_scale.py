import numpy as np, json, os
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
X_npy = np.load(f"{B}/local_harness/tr_frames.npy", mmap_mode="r")
print(f"X mean: {X_npy.mean():.6f}, std: {X_npy.std():.6f}")
# Sample a window
w = X_npy[0:40]
x = w[:20, ..., :2]
y = w[20:, ..., :2]
print(f"x mean: {x.mean():.6f}, std: {x.std():.6f}")
print(f"y mean: {y.mean():.6f}, std: {y.std():.6f}")
