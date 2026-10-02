import torch, numpy as np, os, sys
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
sys.path.insert(0, f"{B}/agents/speed/screen")
sys.path.insert(0, f"{B}/code/RealPDEBench")
from load_baseline import load_baseline

def get_expert_norm(path):
    data = np.load(path, allow_pickle=True)
    norms = {}
    for layer, params_arr in data.items():
        params = params_arr.item() if params_arr.ndim == 0 else params_arr
        A = params["lora_A"]
        B = params["lora_B"]
        norms[layer] = np.linalg.norm(B @ A)
    return norms

experts_dir = f"{B}/agents/moe/experts"
all_norms = {}
for f in os.listdir(experts_dir):
    if f.endswith(".npz"):
        eid = f.replace("expert_", "").replace(".npz", "")
        all_norms[eid] = get_expert_norm(os.path.join(experts_dir, f))

print(f"{'Expert':<12} | {'fc0':<10} | {'fc1':<10} | {'fc2':<10}")
print("-" * 50)
for eid, norms in all_norms.items():
    print(f"{eid:<12} | {norms['fc0']:<10.4f} | {norms['fc1']:<10.4f} | {norms['fc2']:<10.4f}")
