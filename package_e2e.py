import torch
import subprocess
import os
import sys
import argparse

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
LH = f"{B}/local_harness"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--tag", required=True)
a = ap.parse_args()

print(f"Loading {a.ckpt}...")
d = torch.load(f"{B}/train_es/{a.ckpt}", map_location="cpu")

fno_path = f"{B}/train_es/fno_{a.tag}.pth"
torch.save(d["fno"], fno_path)

out_unet = {
    "sd": d["unet"],
    "w": 96,
    "mu": -6.0,
    "sd_": 1.5
}
unet_path = f"{B}/train_es/unet_{a.tag}.pth"
torch.save(out_unet, unet_path)
print(f"Saved {fno_path} and {unet_path}")

fno_fp16_path = f"{B}/train_es/fno_{a.tag}_fp16.pth"
r = subprocess.run([sys.executable, f"{KIT}/pack_ckpt_fp16.py", fno_path, fno_fp16_path], capture_output=True, text=True)
print("Pack FNO:", r.stdout.strip(), r.stderr.strip()[:400])
