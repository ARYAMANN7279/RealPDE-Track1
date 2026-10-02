import numpy as np, os
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
exp_dir = f"{B}/agents/moe/experts"
experts = [f for f in os.listdir(exp_dir) if f.endswith(".npz")]
for exp in sorted(experts):
    data = np.load(os.path.join(exp_dir, exp), allow_pickle=True)
    print(f"--- {exp} ---")
    for k in data.files:
        layer_data = data[k]
        if isinstance(layer_data, np.ndarray) and layer_data.dtype == object:
            layer_data = layer_data.item()
        
        if isinstance(layer_data, dict):
            for lora_k, w in layer_data.items():
                print(f"{k} | {lora_k} | norm: {np.linalg.norm(w):.4f} | shape: {w.shape}")
        else:
            print(f"{k} | raw norm: {np.linalg.norm(layer_data):.4f} | shape: {layer_data.shape}")
