import torch, numpy as np, os, sys
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
sys.path.insert(0, f"{B}/agents/speed/screen")
sys.path.insert(0, f"{B}/code/RealPDEBench")
from load_baseline import load_baseline

model, _ = load_baseline(f"{B}/agents/speed/screen/sim_real_fno_fp16.pth")
model.eval().cuda()
x = torch.randn(1, 20, 32, 64, 3).cuda()
out_base = model(x)
print(f"Base output mean: {out_base.mean().item():.6f}, std: {out_base.std().item():.6f}, max: {out_base.max().item():.6f}")

# Try a LoRA update manually
# fc0: [64, 6]
A = torch.randn(8, 6) * 0.01
B_mat = torch.zeros(64, 8)
# Simulate the weight update W = W + B@A * 0.125
# For just one layer
W_base = model.fc0.weight.data.clone()
model.fc0.weight.data += (B_mat @ A).cuda() * 0.125
out_lora = model(x)
print(f"LoRA output mean: {out_lora.mean().item():.6f}, std: {out_lora.std().item():.6f}, max: {out_lora.max().item():.6f}")
