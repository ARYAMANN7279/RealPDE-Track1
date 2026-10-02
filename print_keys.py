import torch
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
base = torch.load(f"{B}/data/comp_real/sim_real_fno.pth", map_location="cpu")
ft = torch.load(f"{B}/train_es/soup_v3_fp16.pth", map_location="cpu")

if isinstance(base, dict) and "model_state_dict" in base:
    base = base["model_state_dict"]
if isinstance(ft, dict) and "model_state_dict" in ft:
    ft = ft["model_state_dict"]
    
bk = set(base.keys())
fk = set(ft.keys())

print("In base not ft:", bk - fk)
print("In ft not base:", fk - bk)
