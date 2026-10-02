import torch, numpy as np, os, json, sys
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
sys.path.insert(0, f"{B}/agents/speed/screen")
sys.path.insert(0, f"{B}/code/RealPDEBench")
from load_baseline import load_baseline
from realpdebench.model.fno import FNO3d

class LoRALinearV6(torch.nn.Module):
    def __init__(self, original_layer, rank=32, alpha=1.0, device="cuda"):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = torch.nn.Parameter(torch.randn(rank, original_layer.in_features, device=device) * 1e-3)
        self.lora_B = torch.nn.Parameter(torch.zeros(original_layer.out_features, rank, device=device))
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

    def forward(self, x):
        return self.original_layer(x) + (x @ self.lora_A.T) @ self.lora_B.T * self.scaling

def inject_lora(model, rank=32):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parts = name.split(".")
            child_name = parts[-1]
            parent = model
            for part in parts[:-1]: parent = getattr(parent, part)
            # Use the current device of the model for the LoRA parameters
            device = next(model.parameters()).device
            setattr(parent, child_name, LoRALinearV6(module, rank=rank, device=device))
    return model

class RegimeDataLoader:
    def __init__(self, target_aoa, batch_size=8, device="cuda"):
        self.target_aoa = target_aoa
        self.batch_size = batch_size
        self.device = device
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
        self.tr_starts = np.array(all_starts)

    def __len__(self): return len(self.tr_starts) // self.batch_size

    def __iter__(self):
        for i in range(0, len(self.tr_starts), self.batch_size):
            idx = self.tr_starts[i : i + self.batch_size]
            if len(idx) < self.batch_size: break
            w = np.stack([np.asarray(self.X_npy[s : s + 40]) for s in idx])
            x = np.concatenate([w[:, :20, ...], np.zeros(w[:, :20, ...].shape[:-1] + (1,), dtype=np.float32)], axis=-1)
            y = np.concatenate([w[:, 20:, ...], np.zeros(w[:, 20:, ...].shape[:-1] + (1,), dtype=np.float32)], axis=-1)
            yield torch.from_numpy(x).float().to(self.device), torch.from_numpy(y).float().to(self.device)

def train_expert(regime_id, target_aoa, base_model_path, device):
    print(f"Training expert {regime_id} for AoA={target_aoa} on {device}...", flush=True)
    loader = RegimeDataLoader(target_aoa, device=device)
    if len(loader.tr_starts) == 0: return
    model, _ = load_baseline(base_model_path)
    model = model.to(device)
    model = inject_lora(model)
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = torch.nn.MSELoss()
    for epoch in range(20):
        total_loss = 0
        for i, (x, y) in enumerate(loader):
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            if i % 100 == 0: print(f"[{regime_id}] Epoch {epoch}, Batch {i}, Loss: {loss.item():.6f}", flush=True)
        scheduler.step()
        print(f"[{regime_id}] Epoch {epoch} complete. Avg Loss: {total_loss/len(loader):.6f}", flush=True)
    lora_params = {name: {"lora_A": m.lora_A.detach().cpu().numpy(), "lora_B": m.lora_B.detach().cpu().numpy()} 
                   for name, m in model.named_modules() if isinstance(m, LoRALinearV6)}
    out_path = f"{B}/agents/moe/experts/expert_{regime_id}.npz"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **lora_params)
    print(f"Saved expert {regime_id}", flush=True)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <regime_id> <target_aoa> <gpu_id>")
        sys.exit(1)
    regime_id = sys.argv[1]
    target_aoa = int(sys.argv[2])
    gpu_id = sys.argv[3]
    device = f"cuda:{gpu_id}"
    base_path = f"{B}/agents/speed/screen/sim_real_fno_fp16.pth"
    train_expert(regime_id, target_aoa, base_path, device)
