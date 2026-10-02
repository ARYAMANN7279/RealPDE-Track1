import torch, numpy as np, os, sys, json
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
sys.path.insert(0, f"{B}/agents/speed/screen")
sys.path.insert(0, f"{B}/code/RealPDEBench")
from load_baseline import load_baseline
from realpdebench.model.fno import FNO3d
from moe_inference_v2 import MoEInference

SIG = 0.0563870259
C = 2
CAL_SOUP = (1.579, 2.770)

def n_(x): return x / (0.5 + x)

def calculate_wof(dm, tke, mv):
    # Use proxy inv based on typical la_score values
    # la_score(e) = 100 * (1 - e)
    def inv_proxy(e): return e # This is just a placeholder
    # Since we want a realistic Wof, we should use la_score values
    # For la_score = 90, inv(90) is small.
    # Looking at compare_subs.py: inv = lambda t: brentq(lambda e: S.score_error(e) - t, 1e-9, 50.0)
    # IfL a_score is 90, e is ~0.1.
    # We'll use a simple linear approximation for inv(la_score) for the projection.
    def la_to_err(score): return 1.0 - (score/100.0)
    
    dm_e = la_to_err(dm)
    tke_e = la_to_err(tke)
    mv_e = la_to_err(mv)
    
    return 0.5*(1-n_(dm_e)) + 0.3*(1-n_(tke_e)) + 0.2*(1-n_(mv_e))

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

if __name__ == "__main__":
    B_root = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
    moe = MoEInference(
        base_model_path=f"{B_root}/agents/speed/screen/sim_real_fno_fp16.pth",
        router_path=f"{B_root}/agents/moe/router_lda.npz",
        experts_dir=f"{B_root}/agents/moe/experts"
    )
    
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
    SCM = (Y[..., :C] != 0.0)
    
    moe.model.eval()
    P_list = []
    with torch.no_grad():
        for i in range(0, len(Xin), 16):
            batch_x = torch.from_numpy(Xin[i:i+16]).cuda()
            P_list.append(moe.forward(batch_x).cpu().numpy())
    P = np.concatenate(P_list, 0)
    
    dm = rel_l2(torch.from_numpy(P), torch.from_numpy(Y))
    tk = tke_l2(torch.from_numpy(P), torch.from_numpy(Y))
    mv = mvpe_l2(torch.from_numpy(P), torch.from_numpy(Y))
    
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
    
    score_dm = 100 * (1 - dm)
    score_tk = 100 * (1 - tk)
    score_mv = 100 * (1 - mv)
    
    Wv = calculate_wof(score_dm, score_tk, score_mv)
    pred_sps = 100 * Wv * E_local
    
    print(f"Metrics: rel_l2={dm:.4f}, tke={tk:.4f}, mvpe={mv:.4f}")
    print(f"E_local: {E_local:.4f}")
    print(f"Projected SPS: {pred_sps:.2f}")
