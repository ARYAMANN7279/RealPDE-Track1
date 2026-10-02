import torch, numpy as np, os, sys
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
sys.path.insert(0, f"{B}/agents/speed/screen")
sys.path.insert(0, f"{B}/code/RealPDEBench")
from load_baseline import load_baseline
from realpdebench.model.fno import FNO3d

class LoRALayer(torch.nn.Module):
    def __init__(self, base_layer, experts_map):
        super().__init__()
        self.base_layer = base_layer
        # experts_map: {expert_id: {"A": tensor, "B": tensor}}
        self.experts = torch.nn.ModuleDict({
            eid: torch.nn.ParameterDict({
                "A": torch.nn.Parameter(val["A"]),
                "B": torch.nn.Parameter(val["B"])
            }) for eid, val in experts_map.items()
        })
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def forward(self, x, expert_id):
        res = self.base_layer(x)
        A = self.experts[expert_id]["A"]
        B = self.experts[expert_id]["B"]
        
        # Scaling alpha/rank = 16/8 = 2.0
        scaling = 2.0
        lora_out = (x @ A.T) @ B.T
        return res + lora_out * scaling

class MoEModel(torch.nn.Module):
    def __init__(self, base_model_path, experts_dir):
        super().__init__()
        base_model, _ = load_baseline(base_model_path)
        self.base_model = base_model
        
        # Load experts
        all_experts = {}
        for f in os.listdir(experts_dir):
            if f.endswith(".npz"):
                eid = f.replace("expert_", "").replace(".npz", "")
                data = np.load(os.path.join(experts_dir, f), allow_pickle=True)
                all_experts[eid] = {
                    layer: {
                        "A": torch.from_numpy(params_arr.item()["lora_A"]).float(),
                        "B": torch.from_numpy(params_arr.item()["lora_B"]).float()
                    } for layer, params_arr in data.items()
                }
        
        # Inject LoRA layers
        for name, module in self.base_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Get all LoRA weights for this layer across all experts
                layer_experts = {eid: exp[name] for eid, exp in all_experts.items() if name in exp}
                
                parts = name.split(".")
                child_name = parts[-1]
                parent = self.base_model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, child_name, LoRALayer(module, layer_experts))
        
        self.to("cuda")
        self.eval()

    def forward(self, x, expert_id):
        # We need to pass expert_id down to the LoRALayer.
        # Since FNO3d.forward doesn't take expert_id, we can set it as a property.
        self.current_expert = expert_id
        # We have to monkey-patch FNO3d.forward or use a wrapper.
        # Actually, I'll just override the forward of the model if it's a wrapper.
        # But FNO3d is the model.
        
        # Let's use a simpler approach: monkey-patch the modules.
        return self.base_model(x, expert_id)

# To make this work, we must modify FNO3d.forward to accept expert_id.
# Instead, I'll just use the weight-swapping approach but optimize it.
