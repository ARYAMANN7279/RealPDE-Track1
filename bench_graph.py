import sys, time, json
import numpy as np
import torch

sys.path.insert(0, "/SML_DISK_24TB/rajeshr/Aryamann/UGP/scratch/speed_opt")
import submission

# modify predict to use compile or graphs
dummy = np.random.randn(48, 20, 32, 64, 3).astype(np.float32)

model, dev = submission._get_model()
# wrap model in cudagraphs or compile
# Let's test standard torch.compile first
model = torch.compile(model, mode="reduce-overhead")
submission._state["model"] = model

# Warmup (compiles)
print("Warming up compile...")
out = submission.predict(dummy)

# Measure
print("Measuring...")
t0 = time.time()
for _ in range(10):
    _ = submission.predict(dummy)
torch.cuda.synchronize()
t1 = time.time()
print(f"Time for 10 batches (48): {t1-t0:.4f}s")
