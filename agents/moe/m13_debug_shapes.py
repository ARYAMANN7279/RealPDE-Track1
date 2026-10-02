import torch, sys
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
sys.path.insert(0, f"{B}/agents/speed/screen")
sys.path.insert(0, f"{B}/code/RealPDEBench")
from load_baseline import load_baseline

model, _ = load_baseline(f"{B}/agents/speed/screen/sim_real_fno_fp16.pth")
print(f"shape_in: {model.shape_in}")
print(f"shape_out: {model.shape_out}")
