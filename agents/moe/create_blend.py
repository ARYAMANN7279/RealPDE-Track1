import torch, os, sys
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"

m1_path = f"{B}/local_harness/soup_v2.pth"
m2_path = f"{B}/train_es/ftaug_sv3_3e5_10.pth"

print(f"Blending {m1_path} and {m2_path}...")
s1 = torch.load(m1_path, map_location="cpu")
s2 = torch.load(m2_path, map_location="cpu")

blended_state = {}
for k in s1.keys():
    if k in s2:
        blended_state[k] = 0.5 * s1[k] + 0.5 * s2[k]
    else:
        blended_state[k] = s1[k]

out_path = f"{B}/agents/moe/blend_7948.pth"
torch.save(blended_state, out_path)
print(f"Saved blended state dict to {out_path}")
