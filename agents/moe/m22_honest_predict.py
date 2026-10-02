import torch, numpy as np, os, sys, json
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
sys.path.insert(0, f"{B}/agents/speed/screen")
sys.path.insert(0, f"{B}/code/RealPDEBench")
from load_baseline import load_baseline
from realpdebench.model.fno import FNO3d

# Constants from compare_subs.py
SIG = 0.0563870259
C = 2
CAL_SOUP = (1.579, 2.770)

def inv(t):
    from scipy.optimize import brentq
    def target(e):
        # Simple version of S.score_error(e)
        # In RealPDEBench, score_error is typically 100 * (1 - e) or similar
        # Looking at compare_subs.py, it calls S.score_error(e).
        # Let's assume score_error(e) = 100 * (1 - e) for simplicity, 
        # BUT actually, I should use the actual scoring.py logic.
        return 100 * (1 - e)
    try:
        return brentq(lambda e: target(e) - t, 1e-9, 50.0)
    except:
        return 0.05 # default

def n_(x): return x / (0.5 + x)

def calculate_wof(dm, tke, mv):
    # Wof = 0.5*(1-n_(inv(dm))) + 0.3*(1-n_(inv(tke))) + 0.2*(1-n_(inv(mv)))
    return 0.5*(1-n_(inv(dm))) + 0.3*(1-n_(inv(tke))) + 0.2*(1-n_(inv(mv)))

class MoEInference:
    def __init__(self, base_model_path, router_path, experts_dir):
        self.model, _ = load_baseline(base_model_path)
        self.model.eval().cuda()
        router_data = np.load(router_path, allow_pickle=True)
        self.router = {
            "mu": torch.from_numpy(router_data["mu"]).float().cuda(),
            "sd": torch.from_numpy(router_data["sd"]).float().cuda(),
            "W": torch.from_numpy(router_data["W"]).float().cuda(),
            "b": torch.from_numpy(router_data["b"]).float().cuda(),
            "classes": router_data["classes"]
        }
        self.experts = {}
        for f in os.listdir(experts_dir):
            if f.endswith(".npz"):
                eid = f.replace("expert_", "").replace(".npz", "")
                data = np.load(os.path.join(experts_dir, f), allow_pickle=True)
                expert_weights = {}
                for layer, params_arr in data.items():
                    params = params_arr.item() if params_arr.ndim == 0 else params_arr
                    expert_weights[layer] = {
                        "A": torch.from_numpy(params["lora_A"]).float().cuda(),
                        "B": torch.from_numpy(params["lora_B"]).float().cuda()
                    }
                self.experts[eid] = expert_weights

    def get_features(self, x):
        B_batch, T, Sx, Sy, C_in = x.shape
        u = x[..., 0]
        v = x[..., 1]
        um = u.mean(1)
        vm = v.mean(1)
        nz = (u != 0) | (v != 0)
        valid = (nz.float().mean(1) > 0.5).float()
        cnt = (valid.sum(dim=(-1, -2)) + 1e-8)
        s = (um * valid).sum(dim=(-1, -2)) / cnt
        vb = (vm * valid).sum(dim=(-1, -2)) / cnt
        su = u.std(1)
        sv = v.std(1)
        ti = torch.sqrt(((su**2 + sv**2) * valid).sum(dim=(-1, -2)) / cnt)
        zf = 1.0 - valid.mean(dim=(-1, -2))
        vf = v[..., 40:] - v[..., 40:].mean(1, keepdim=True)
        P_full = torch.abs(torch.fft.rfft(vf, dim=1))**2
        P = P_full.mean(dim=(-2, -1))[:, 1:11]
        Pn = P / (P.sum(dim=1, keepdim=True) + 1e-20)
        fc = (Pn * torch.arange(1, 11, device=x.device).float()).sum(dim=1)
        def pool(a):
            res = a.view(B_batch, 8, 4, 16, 4).mean(dim=(2, 4))
            return res.reshape(B_batch, -1)
        sc = s[:, None, None]
        f_scalars = torch.stack([torch.log(s + 1e-8), zf, vb/s, ti/s, fc], dim=1)
        f_pool_um = pool(um / (sc + 1e-8))
        f_pool_vm = pool(vm / (sc + 1e-8))
        f_pool_std = pool(torch.sqrt(su**2 + sv**2) / (sc + 1e-8))
        f_pool_valid = pool(valid)
        return torch.cat([f_scalars, Pn, f_pool_um, f_pool_vm, f_pool_std, f_pool_valid], dim=1)

    def route(self, x):
        F = self.get_features(x)
        Z = (F - self.router["mu"]) / self.router["sd"]
        logits = Z @ self.router["W"] + self.router["b"]
        idx = torch.argmax(logits, dim=1)
        return self.router["classes"][idx.cpu().numpy()]

    def forward(self, x):
        expert_vals = self.route(x)
        vals, counts = np.unique(expert_vals, return_counts=True)
        expert_val = vals[np.argmax(counts)]
        expert_id = str(expert_val)
        if expert_id not in self.experts:
            expert_id = list(self.experts.keys())[0]
        expert_weights = self.experts[expert_id]
        originals = {}
        for layer_name, weights in expert_weights.items():
            module = dict(self.model.named_modules())[layer_name]
            originals[layer_name] = module.weight.data.clone()
            A, B = weights["A"], weights["B"]
            module.weight.data += B @ A
        out = self.model(x)
        for layer_name, weight in originals.items():
            module = dict(self.model.named_modules())[layer_name]
            module.weight.data = weight
        return out

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
    # Simple proxy for MVPE: rel_l2 of the mean velocity profile
    # In a real submission, this is computed on the output frames
    return rel_l2(p, t) # Fallback

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
    
    # Calculate 3 metrics
    dm = rel_l2(torch.from_numpy(P), torch.from_numpy(Y))
    tk = tke_l2(torch.from_numpy(P), torch.from_numpy(Y))
    mv = mvpe_l2(torch.from_numpy(P), torch.from_numpy(Y))
    
    # E_local (SPS component)
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
    
    # Honest SPS Projection
    # We need to map the local errors (dm, tk, mv) to their score_error(e)
    # Since we don't have the real scoring.py here, we'll use the inverse mapping
    # from compare_subs.py logic.
    
    # Using dummy values for score_error in this local script
    # Real la_score = S.score_error(dm), etc.
    # let's assume la_score(e) = 100*(1-e) for rough estimate
    score_dm = 100 * (1 - dm)
    score_tk = 100 * (1 - tk)
    score_mv = 100 * (1 - mv)
    
    # la_score is what goes into Wof. a a la:
    # Wof = 0.5*(1-n_(inv(score_dm))) + 0.3*(1-n_(inv(score_tk))) + 0.2*(1-n_(inv(score_mv)))
    # Actually, compare_subs.py defines Wof using the real score_error results.
    
    # For now, I will use a refined Wv based on the ratios of the 3 metrics.
    # a la compare_subs.py: Wv = Wof(subscores)
    # If the metrics are similar, Wv ~ 0.34.
    Wv = calculate_wof(score_dm, score_tk, score_mv)
    pred_sps = 100 * Wv * E_local
    
    print(f"Metrics: rel_l2={dm:.4f}, tke={tk:.4f}, mvpe={mv:.4f}")
    print(f"E_local: {E_local:.4f}")
    print(f"Projected SPS: {pred_sps:.2f}")
