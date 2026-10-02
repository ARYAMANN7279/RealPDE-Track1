import torch, numpy as np, os, json, sys
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
sys.path.insert(0, f"{B}/agents/speed/screen")
sys.path.insert(0, f"{B}/code/RealPDEBench")
from load_baseline import load_baseline
from realpdebench.model.fno import FNO3d

class LoRALinear(torch.nn.Module):
    def __init__(self, original_layer, rank=4, alpha=0.1): # Even smaller rank and alpha
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.scaling = alpha / rank
        device = original_layer.weight.device
        self.lora_A = torch.nn.Parameter(torch.randn(rank, original_layer.in_features, device=device) * 1e-4)
        self.lora_B = torch.nn.Parameter(torch.zeros(original_layer.out_features, rank, device=device))
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

    def forward(self, x):
        return self.original_layer(x) + (x @ self.lora_A.T) @ self.lora_B.T * self.scaling

def inject_lora(model, rank=4):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parts = name.split(".")
            child_name = parts[-1]
            parent = model
            for part in parts[:-1]: parent = getattr(parent, part)
            setattr(parent, child_name, LoRALinear(module, rank=rank))
    return model

class RegimeDataLoader:
    def __init__(self, target_aoa, batch_size=8, split=0.9):
        self.target_aoa = target_aoa
        self.batch_size = batch_size
        self.X_npy = np.load(f"{B}/local_harness/tr_frames.npy", mmap_mode="r")
        with open(f"{B}/local_harness/tr_meta.json", "r") as f:
            self.meta = json.load(f)
        self.off, self.lens, self.names = self.meta["off"], self.meta["lens"], self.meta["names"]
        all_starts = []
        for i, name in enumerate(self.names):
            try:
                if int(name.split("_")[1].split(".")[0]) == target_aoa:
                    for t0 in range(self.off[i], self.off[i] + self.lens[i] - 39):
                        all_starts.append(t0)
            except: continue
        
        rng = np.random.default_rng(0)
        rng.shuffle(all_starts)
        split_idx = int(len(all_starts) * split)
        self.tr_starts = np.array(all_starts[:split_idx])
        self.va_starts = np.array(all_starts[split_idx:])
        print(f"AoA={target_aoa}: train={len(self.tr_starts)}, val={len(self.va_starts)}")

    def _get_batch(self, starts):
        w = np.stack([np.asarray(self.X_npy[s : s + 40]) for s in starts])
        x = np.concatenate([w[:, :20, ...], np.zeros(w[:, :20, ...].shape[:-1] + (1,), dtype=np.float32)], axis=-1)
        y = np.concatenate([w[:, 20:, ...], np.zeros(w[:, 20:, ...].shape[:-1] + (1,), dtype=np.float32)], axis=-1)
        return torch.from_numpy(x).float().cuda(), torch.from_numpy(y).float().cuda()

    def train_iter(self):
        indices = np.random.permutation(self.tr_starts)
        for i in range(0, len(indices), self.batch_size):
            idx = indices[i : i + self.batch_size]
            if len(idx) < self.batch_size: break
            yield self._get_batch(idx)

    def val_iter(self):
        for i in range(0, len(self.va_starts), self.batch_size):
            idx = self.va_starts[i : i + self.batch_size]
            if len(idx) < self.batch_size: break
            yield self._get_batch(idx)

def train_expert(regime_id, target_aoa, base_model_path):
    loader = RegimeDataLoader(target_aoa)
    if len(loader.tr_starts) == 0: return
    model, _ = load_baseline(base_model_path)
    model = inject_lora(model.cuda())
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-6, weight_decay=1e-2)
    criterion = torch.nn.MSELoss()
    
    best_val = float('inf')
    for epoch in range(10):
        # Train
        model.train()
        total_tr = 0
        for x, y in loader.train_iter():
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_tr += loss.item()
        
        # Val
        model.eval()
        total_va = 0
        with torch.no_grad():
            for x, y in loader.val_iter():
                total_va += criterion(model(x), y).item()
        avg_va = total_va / (len(loader.va_starts)//loader.batch_size + 1)
        print(f"Epoch {epoch} | Tr Loss: {total_tr/len(loader.tr_starts)//loader.batch_size:.6f} | Val Loss: {avg_va:.6f}")
        if avg_va < best_val:
            best_val = avg_va
            # Save best
            lora_params = {name: {"lora_A": m.lora_A.detach().cpu().numpy(), "lora_B": m.lora_B.detach().cpu().numpy()} 
                           for name, m in model.named_modules() if isinstance(m, LoRALinear)}
            np.savez(f"{B}/agents/moe/experts/expert_{regime_id}.npz", **lora_params)

if __name__ == "__main__":
    base_path = f"{B}/agents/speed/screen/sim_real_fno_fp16.pth"
    train_expert("AoA_0", 0, base_path)
