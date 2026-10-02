import torch, numpy as np, os, sys
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
sys.path.insert(0, f"{B}/agents/speed/screen")
sys.path.insert(0, f"{B}/code/RealPDEBench")
from load_baseline import load_baseline
from realpdebench.model.fno import FNO3d

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
                    # LoRA V6 saved weights as objects/dicts
                    params = params_arr.item() if params_arr.ndim == 0 else params_arr
                    if isinstance(params, dict):
                        expert_weights[layer] = {
                            "A": torch.from_numpy(params["lora_A"]).float().cuda(),
                            "B": torch.from_numpy(params["lora_B"]).float().cuda()
                        }
                self.experts[eid] = expert_weights

    def get_features(self, x):
        B_batch, T, Sx, Sy, C_in = x.shape
        u = x[..., 0]; v = x[..., 1]
        um = u.mean(1); vm = v.mean(1)
        nz = (u != 0) | (v != 0)
        valid = (nz.float().mean(1) > 0.5).float()
        cnt = (valid.sum(dim=(-1, -2)) + 1e-8)
        s = (um * valid).sum(dim=(-1, -2)) / cnt
        vb = (vm * valid).sum(dim=(-1, -2)) / cnt
        su = u.std(1); sv = v.std(1)
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
        
        # Rank 32, Alpha 1.0 -> scaling = 1/32 = 0.03125
        scaling = 0.03125
        
        originals = {}
        for layer_name, weights in expert_weights.items():
            module = dict(self.model.named_modules())[layer_name]
            originals[layer_name] = module.weight.data.clone()
            A, B = weights["A"], weights["B"]
            module.weight.data += (B @ A) * scaling
            
        out = self.model(x)
        
        for layer_name, weight in originals.items():
            module = dict(self.model.named_modules())[layer_name]
            module.weight.data = weight
            
        return out
