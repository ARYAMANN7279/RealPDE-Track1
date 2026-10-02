import numpy as np
import torch
import glob
import os

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
TES = f"{B}/train_es"
build_dir = f"{B}/submissions/build_mega"
os.makedirs(build_dir, exist_ok=True)
os.system(f"unzip -qo {B}/submissions/submission_SV2.zip -d {build_dir}")

z = np.load(f"{build_dir}/bounds_assets.npz")
out = {k: z[k] for k in z.files}

j = 0
while f"e{j}_w" in z.files:
    j += 1

print(f"Original ensemble had {j} extra models. Adding more...")

new_unets = glob.glob(f"{TES}/joint_unet_m*.pth")
for path in new_unets:
    sd = torch.load(path, map_location="cpu")
    out[f"e{j}_w"] = np.array(96, dtype=np.int32)
    for k, v in sd.items():
        out[f"e{j}_p_{k}"] = v.numpy().astype(np.float32)
    j += 1

cw = np.ones(j + 1, dtype=np.float32) / (j + 1)
out["cw"] = cw

np.savez(f"{build_dir}/bounds_assets.npz", **out)
print(f"Packed {j} total extra models into bounds_assets.npz. New center weights: {cw}")

os.system(f"cd {build_dir} && zip -q -0 -r ../submission_MEGA_SV2.zip ./*")
print("Saved submission_MEGA_SV2.zip!")
