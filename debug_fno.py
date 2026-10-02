import sys, os, torch, numpy as np
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
sys.path.insert(0, f"{B}/code/RealPDEBench")
sys.path.insert(0, f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9")
sys.path.insert(0, f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9/_vendor")
from rpde_baselines.model.load_model import load_model
from load_baseline import load_baseline
model, _ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device="cuda:0")
sd = torch.load(f"{B}/data/comp_real/sim_real_fno.pth", map_location="cuda:0")
model.load_state_dict(sd.get("model_state_dict", sd))
x = torch.zeros(1, 20, 32, 64, 3).cuda()
y = model(x)
print(y.shape)
