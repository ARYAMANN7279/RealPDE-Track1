import sys, os, zipfile, io, torch, numpy as np, subprocess

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP/submissions"
src_zip = f"{B}/submission_ENSEMBLE_FAST_v3.zip"
dst_zip = f"{B}/submission_ENS_IMPROVED_v3.zip"

print(f"Loading base zip {src_zip}...")
with zipfile.ZipFile(src_zip, 'r') as z:
    b_assets = z.read("bounds_assets.npz")
    with open(f"/SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es/tmp_v3_bounds.npz", "wb") as f: f.write(b_assets)

d = np.load("/SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es/tmp_v3_bounds.npz")
out_dict = dict(d)

print("Loading new weights...")
T = "/SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es"
sd0 = torch.load(f"{T}/joint_ENS_s100.pth", map_location="cpu")
if "sd" in sd0: sd0 = sd0["sd"]
sd1 = torch.load(f"{T}/joint_ENS_s101.pth", map_location="cpu")
if "sd" in sd1: sd1 = sd1["sd"]
sd2 = torch.load(f"{T}/joint_ENS_s102.pth", map_location="cpu")
if "sd" in sd2: sd2 = sd2["sd"]

for k, v in sd0.items(): out_dict["w_" + k] = v.numpy().astype(np.float16)
for k, v in sd1.items(): out_dict["e0_p" + k] = v.numpy().astype(np.float16)
for k, v in sd2.items(): out_dict["e1_p" + k] = v.numpy().astype(np.float16)

np.savez_compressed(f"{T}/tmp_improved_bounds.npz", **out_dict)

print(f"Building new zip {dst_zip}...")
if os.path.exists(dst_zip): os.remove(dst_zip)
with zipfile.ZipFile(src_zip, 'r') as zin, zipfile.ZipFile(dst_zip, 'w', zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        if item.filename == "bounds_assets.npz":
            zout.writestr(item, open(f"{T}/tmp_improved_bounds.npz", "rb").read())
        else:
            zout.writestr(item, zin.read(item.filename))

print("Evaluating improved ensemble artifact...")
subprocess.run(f"python3 {T}/eval_anchors_fast.py {dst_zip}", shell=True, check=True)
