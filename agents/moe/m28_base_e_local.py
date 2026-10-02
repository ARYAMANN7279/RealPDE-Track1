import torch, numpy as np, os, sys
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
sys.path.insert(0, f"{B}/agents/speed/screen")
sys.path.insert(0, f"{B}/code/RealPDEBench")
from load_baseline import load_baseline

def get_e_local(model, x, y):
    model.eval().cuda()
    with torch.no_grad():
        p = model(x.cuda())
        err = torch.abs(p[..., :2] - y.cuda())
        h = 0.05 * torch.abs(p[..., :2])
        num = (torch.exp(-2*h/0.056387) * (err <= h)).sum()
        tot = err.numel()
        return num.item() / tot

model, _ = load_baseline(f"{B}/agents/speed/screen/sim_real_fno_fp16.pth")
X_npy = np.load(f"{B}/local_harness/tr_frames.npy", mmap_mode="r")
x = torch.from_numpy(X_npy[0:20].copy()).float().unsqueeze(0).cuda()
x = torch.cat([x, torch.zeros(x.shape[:-1] + (1,), device="cuda").float()], dim=-1)
y = torch.from_numpy(X_npy[20:40].copy()).float().unsqueeze(0).cuda()

e_loc = get_e_local(model, x, y)
print(f"Base E_local: {e_loc:.4f}")
