import torch, numpy as np, os, sys
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
sys.path.insert(0, f"{B}/agents/speed/screen")
sys.path.insert(0, f"{B}/code/RealPDEBench")
from load_baseline import load_baseline

def get_e_local(model, x, y):
    model.eval().cuda()
    with torch.no_grad():
        p = model(x.cuda())
        # Target is only the first 2 channels
        err = torch.abs(p[..., :2] - y[..., :2].cuda())
        h = 0.05 * torch.abs(p[..., :2])
        num = (torch.exp(-2*h/0.056387) * (err <= h)).sum()
        tot = err.numel()
        return num.item() / tot

model, _ = load_baseline(f"{B}/agents/speed/screen/sim_real_fno_fp16.pth")
data = np.load(f"{B}/agents/moe/experts/expert_AoA_0.npz", allow_pickle=True)
for layer_name, params_arr in data.items():
    params = params_arr.item() if params_arr.ndim == 0 else params_arr
    module = dict(model.named_modules())[layer_name]
    A = torch.from_numpy(params["lora_A"]).float().cuda()
    B_mat = torch.from_numpy(params["lora_B"]).float().cuda()
    model.cuda()
    module.weight.data += (B_mat @ A) * 0.125

X_npy = np.load(f"{B}/local_harness/tr_frames.npy", mmap_mode="r")
x = torch.from_numpy(X_npy[0:20].copy()).float().unsqueeze(0).cuda()
x = torch.cat([x, torch.zeros(x.shape[:-1] + (1,), device="cuda").float()], dim=-1)
y = torch.from_numpy(X_npy[20:40].copy()).float().unsqueeze(0).cuda()

e_loc = get_e_local(model, x, y)
print(f"AoA_0 v3 E_local: {e_loc:.4f}")
