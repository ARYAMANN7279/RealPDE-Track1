import torch, numpy as np, os, sys, json
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
sys.path.insert(0, f"{B}/agents/speed/screen")
sys.path.insert(0, f"{B}/code/RealPDEBench")
from load_baseline import load_baseline
from moe_inference_v3 import MoEInference

SIG = 0.0563870259
C = 2
CAL_SOUP = (1.579, 2.770)
MI = torch.tensor([0.154960856,-0.000513992854,0.0])
SI = torch.tensor([0.0968056545,0.015960684,1.0])
MT = torch.tensor([0.154962569,-0.000517793698,0.0])
ST = torch.tensor([0.0968104079,0.0159636438,1.0])

def evaluate_on_regime(model, Xin, Y):
    model.eval().cuda()
    P_list = []
    with torch.no_grad():
        for i in range(0, len(Xin), 16):
            batch_x = torch.from_numpy(Xin[i:i+16]).cuda()
            x_norm = (batch_x - MI.cuda()) / SI.cuda()
            out = model.forward(x_norm) if hasattr(model, "forward") else model(x_norm)
            out = out * ST.cuda() + MT.cuda()
            P_list.append(out.cpu().numpy())
    P = np.concatenate(P_list, 0)
    
    # E_local calculation
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
    return num/tot

if __name__ == "__main__":
    B_root = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
    base_path = f"{B_root}/agents/speed/screen/sim_real_fno_fp16.pth"
    moe = MoEInference(base_path, f"{B_root}/agents/moe/router_lda.npz", f"{B_root}/agents/moe/experts")
    base_model, _ = load_baseline(base_path)
    base_model.eval().cuda()
    
    X_npy = np.load(f"{B_root}/local_harness/tr_frames.npy", mmap_mode="r")
    meta = json.load(open(f"{B_root}/local_harness/tr_meta.json"))
    off, lens, names = meta["off"], meta["lens"], meta["names"]
    
    print(f"{'Regime':<10} | {'Base E_local':<15} | {'Expert E_local':<15} | {'Gain'}")
    print("-" * 55)
    
    for aoa in [0, 5, 10, 15, 20]:
        # Collect samples for this specific AoA
        starts = []
        for i, name in enumerate(names):
            try:
                if int(name.split("_")[1].split(".")[0]) == aoa:
                    for t0 in range(off[i], off[i]+lens[i]-39, 20):
                        starts.append(t0)
            except: continue
        
        if not starts: continue
        
        # Sample 200 windows for a quick check
        rng = np.random.default_rng(0)
        starts = rng.choice(starts, size=min(200, len(starts)), replace=False)
        Wd = np.stack([np.concatenate([np.asarray(X_npy[s:s+40]), np.zeros((40,32,64,1), np.float32)], -1)
                       for s in starts]).astype(np.float32)
        Xin, Y = Wd[:, :20], Wd[:, 20:]
        
        # Test Base
        e_base = evaluate_on_regime(base_model, Xin, Y)
        
        # Test Expert (Force the MoE to use the specific expert)
        # We can do this by temporarily modifying the MoE router or manually applying weights
        # Simplified: just modify the MoEInference to accept an expert_id override
        
        # Manually apply the expert weights to the base model for a clean test
        eid = f"AoA_{aoa}"
        expert_weights = moe.experts[eid]
        scaling = 0.03125
        originals = {}
        for layer_name, weights in expert_weights.items():
            module = dict(base_model.named_modules())[layer_name]
            originals[layer_name] = module.weight.data.clone()
            A, B = weights["A"], weights["B"]
            module.weight.data += (B @ A) * scaling
        
        e_exp = evaluate_on_regime(base_model, Xin, Y)
        
        # Restore base model
        for layer_name, weight in originals.items():
            module = dict(base_model.named_modules())[layer_name]
            module.weight.data = weight
            
        print(f"AoA_{aoa:<5} | {e_base:<15.4f} | {e_exp:<15.4f} | {e_base - e_exp:.4f}")
