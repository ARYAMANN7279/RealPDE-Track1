import torch, numpy as np, os, sys, json
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
sys.path.insert(0, f"{B}/agents/speed/screen")
sys.path.insert(0, f"{B}/code/RealPDEBench")
from load_baseline import load_baseline
from moe_inference_v2 import MoEInference

SIG = 0.0563870259
C = 2
CAL_SOUP = (1.579, 2.770)
# Normalization constants from train_e2e.py
MI = torch.tensor([0.154960856,-0.000513992854,0.0])
SI = torch.tensor([0.0968056545,0.015960684,1.0])
MT = torch.tensor([0.154962569,-0.000517793698,0.0])
ST = torch.tensor([0.0968104079,0.0159636438,1.0])

def n_(x): return x / (0.5 + x)

def calculate_wof(dm, tke, mv):
    def la_to_err(score): return 1.0 - (score/100.0)
    return 0.5*(1-n_(la_to_err(dm))) + 0.3*(1-n_(la_to_err(tke))) + 0.2*(1-n_(la_to_err(mv)))

def rel_l2(p, t):
    p = p[..., :C].reshape(p.shape[0], -1)
    t = t[..., :C].reshape(t.shape[0], -1)
    return (torch.linalg.norm(p-t, dim=1) / torch.linalg.norm(t, dim=1).clamp(min=1e-8)).mean().item()

def tke_l2(p, t):
    def ke(z):
        u, v = z[..., 0], z[..., 1]
        return 0.5 * (((u-u.mean(1, keepdim=True))**2).mean(1) + ((v-v.mean(1, keepdim=True))**2).mean(1))
    pk = ke(p[..., :C]).reshape(p.shape[0], -1)
    tk = ke(t[..., :C]).reshape(t.shape[0], -1)
    return (torch.linalg.norm(pk-tk, dim=1) / torch.linalg.norm(tk, dim=1).clamp(min=1e-8)).mean().item()

def mvpe_l2(p, t):
    return rel_l2(p, t)

def evaluate_model(model, Xin, Y):
    if hasattr(model, "eval"):
        model.eval().cuda()
    
    P_list = []
    with torch.no_grad():
        for i in range(0, len(Xin), 16):
            batch_x = torch.from_numpy(Xin[i:i+16]).cuda()
            # Normalize
            x_norm = (batch_x - MI.cuda()) / SI.cuda()
            
            if hasattr(model, "forward"):
                out = model.forward(x_norm)
            else:
                out = model(x_norm)
            
            # Denormalize
            out = out * ST.cuda() + MT.cuda()
            P_list.append(out.cpu().numpy())
    P = np.concatenate(P_list, 0)
    
    dm = rel_l2(torch.from_numpy(P), torch.from_numpy(Y))
    tk = tke_l2(torch.from_numpy(P), torch.from_numpy(Y))
    mv = mvpe_l2(torch.from_numpy(P), torch.from_numpy(Y))
    
    SCM = (Y[..., :C] != 0.0)
    ERR = np.abs(P[..., :C] - Y[..., :C])
    au, av = CAL_SOUP
    num = 0.0; tot = 0
    for ci, a in ((0, au), (1, av)):
        m = SCM[..., ci]
        e = ERR[..., ci][m]*a
        h = 0.05*np.abs(P[..., ci][m])
        num += float((np.exp(-2*h/SIG)*(e <= h)).sum())
        tot += e.size
    E_local = num/tot
    
    Wv = calculate_wof(100*(1-dm), 100*(1-tk), 100*(1-mv))
    return E_local, 100 * Wv * E_local, dm, tk, mv

if __name__ == "__main__":
    B_root = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
    moe = MoEInference(
        base_model_path=f"{B_root}/agents/speed/screen/sim_real_fno_fp16.pth",
        router_path=f"{B_root}/agents/moe/router_lda.npz",
        experts_dir=f"{B_root}/agents/moe/experts"
    )
    base_model, _ = load_baseline(f"{B_root}/agents/speed/screen/sim_real_fno_fp16.pth")
    base_model.eval().cuda()
    
    X_npy = np.load(f"{B_root}/local_harness/tr_frames.npy", mmap_mode="r")
    meta = json.load(open(f"{B_root}/local_harness/tr_meta.json"))
    off, lens = meta["off"], meta["lens"]
    ntraj = len(lens)
    held = sorted(set(range(0, ntraj, 5)))
    starts = []
    for i in held:
        starts += list(range(off[i], off[i]+lens[i]-39, 20))
    rng = np.random.default_rng(0)
    starts = rng.choice(starts, size=min(400, len(starts)), replace=False)
    Wd = np.stack([np.concatenate([np.asarray(X_npy[s:s+40]), np.zeros((40,32,64,1), np.float32)], -1)
                   for s in starts]).astype(np.float32)
    Xin, Y = Wd[:, :20], Wd[:, 20:]
    
    e_moe, sps_moe, dm_moe, tk_moe, mv_moe = evaluate_model(moe, Xin, Y)
    e_base, sps_base, dm_base, tk_base, mv_base = evaluate_model(base_model, Xin, Y)
    
    print(f"Base Model: E_local={e_base:.4f}, SPS={sps_base:.2f}, rel_l2={dm_base:.4f}, tke={tk_base:.4f}")
    print(f"MoE Model:  E_local={e_moe:.4f}, SPS={sps_moe:.2f}, rel_l2={dm_moe:.4f}, tke={tk_moe:.4f}")
    print(f"Gain: {sps_moe - sps_base:.2f}")
