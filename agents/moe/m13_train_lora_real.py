import torch, numpy as np, os, json, sys
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
sys.path.insert(0, f"{B}/agents/speed/screen")
sys.path.insert(0, f"{B}/code/RealPDEBench")
from load_baseline import load_baseline
from realpdebench.model.fno import FNO3d

class LoRALinear(torch.nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.scaling = alpha / rank
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        device = original_layer.weight.device
        self.lora_A = torch.nn.Parameter(torch.randn(rank, in_features, device=device) * 0.01)
        self.lora_B = torch.nn.Parameter(torch.zeros(out_features, rank, device=device))
        
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

    def forward(self, x):
        res = self.original_layer(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return res + lora_out * self.scaling

def inject_lora(model, rank=8):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(f"Injecting LoRA into {name}")
            parts = name.split(".")
            child_name = parts[-1]
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, child_name, LoRALinear(module, rank=rank))
    return model

class RegimeDataLoader:
    def __init__(self, target_aoa, batch_size=8):
        self.target_aoa = target_aoa
        self.batch_size = batch_size
        self.X_npy = np.load(f"{B}/local_harness/tr_frames.npy", mmap_mode="r")
        with open(f"{B}/local_harness/tr_meta.json", "r") as f:
            self.meta = json.load(f)
        
        self.off = self.meta["off"]
        self.lens = self.meta["lens"]
        self.names = self.meta["names"]
        
        self.starts = []
        for i, name in enumerate(self.names):
            try:
                aoa = int(name.split("_")[1].split(".")[0])
                if aoa == self.target_aoa:
                    for t0 in range(self.off[i], self.off[i] + self.lens[i] - 39):
                        self.starts.append(t0)
            except (IndexError, ValueError):
                continue
        self.starts = np.array(self.starts)
        print(f"Regime AoA={target_aoa}: found {len(self.starts)} windows")

    def __len__(self):
        return len(self.starts) // self.batch_size

    def __iter__(self):
        indices = np.random.permutation(self.starts)
        for i in range(0, len(indices), self.batch_size):
            idx = indices[i : i + self.batch_size]
            if len(idx) < self.batch_size: break
            
            # Load window of 40 frames
            # w: [B, 40, 32, 64, 2]
            w = np.stack([np.asarray(self.X_npy[s : s + 40]) for s in idx])
            
            # Input x: first 20 frames, append zero channel -> [B, 20, 32, 64, 3]
            x_raw = w[:, :20, ...]
            z = np.zeros(x_raw.shape[:-1] + (1,), dtype=np.float32)
            x = np.concatenate([x_raw, z], axis=-1)
            
            # Target y: frames 20 to 39, append zero channel -> [B, 20, 32, 64, 3]
            y_raw = w[:, 20:, ...]
            z_y = np.zeros(y_raw.shape[:-1] + (1,), dtype=np.float32)
            y = np.concatenate([y_raw, z_y], axis=-1)
            
            yield torch.from_numpy(x).float(), torch.from_numpy(y).float()

def train_expert(regime_id, target_aoa, base_model_path):
    print(f"Training expert {regime_id} for AoA={target_aoa}...")
    loader = RegimeDataLoader(target_aoa)
    if len(loader.starts) == 0:
        print(f"No data for AoA={target_aoa}. Skipping.")
        return
    
    model, _ = load_baseline(base_model_path)
    model = model.cuda()
    model = inject_lora(model)
    model.train()
    
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(3):
        total_loss = 0
        for i, (x, y) in enumerate(loader):
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 50 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.6f}")
        print(f"Epoch {epoch} complete. Avg Loss: {total_loss/len(loader):.6f}")
    
    lora_params = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_params[name] = {
                "lora_A": module.lora_A.detach().cpu().numpy(),
                "lora_B": module.lora_B.detach().cpu().numpy(),
            }
    
    out_path = f"{B}/agents/moe/experts/expert_{regime_id}.npz"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **lora_params)
    print(f"Saved expert {regime_id} to {out_path}")

if __name__ == "__main__":
    base_path = f"{B}/agents/speed/screen/sim_real_fno_fp16.pth"
    for aoa in [0, 5, 10, 15, 20]:
        train_expert(f"AoA_{aoa}", aoa, base_path)
