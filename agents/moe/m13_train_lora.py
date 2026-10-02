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

def train_expert(regime_id, data_loader, base_model_path):
    print(f"Training expert {regime_id}...")
    model, _ = load_baseline(base_model_path)
    model = model.cuda()
    model = inject_lora(model)
    model.train()
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=1e-4
    )
    criterion = torch.nn.MSELoss()
    
    for epoch in range(5):
        total_loss = 0
        for i, (x, y) in enumerate(data_loader):
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.6f}")
        print(f"Epoch {epoch} complete. Avg Loss: {total_loss/len(data_loader):.6f}")
    
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
    class DummyLoader:
        def __len__(self): return 10
        def __iter__(self):
            for _ in range(10):
                # Update to match shape_in: [20, 32, 64, 3]
                yield torch.randn(4, 20, 32, 64, 3), torch.randn(4, 20, 32, 64, 3)

    base_path = f"{B}/agents/speed/screen/sim_real_fno_fp16.pth"
    train_expert("AoA_0", DummyLoader(), base_path)
