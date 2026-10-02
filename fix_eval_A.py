import sys, os, zipfile, io, torch, numpy as np, subprocess

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP/submissions"
src_zip = f"{B}/submission_ENSEMBLE_FAST_v3.zip"
dst_zip = f"{B}/submission_DISTILLED_v3.zip"

print(f"Loading base zip {src_zip}...", flush=True)
with zipfile.ZipFile(src_zip, 'r') as z:
    b_assets = z.read("bounds_assets.npz")
    with open(f"/SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es/tmp_v3_bounds.npz", "wb") as f: f.write(b_assets)

d = np.load("/SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es/tmp_v3_bounds.npz")
out_dict = dict(d)

print("Loading distilled weights...", flush=True)
T = "/SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es"
sd_dist = torch.load(f"{T}/distilled_v3_W96.pth", map_location="cpu")
if "sd" in sd_dist: sd_dist = sd_dist["sd"]

if sd_dist["out.weight"].shape[0] == 80:
    w = sd_dist["out.weight"]
    b = sd_dist["out.bias"]
    new_w = torch.zeros((120, *w.shape[1:]), dtype=w.dtype)
    new_w[:80] = w
    new_b = torch.zeros(120, dtype=b.dtype)
    new_b[:80] = b
    sd_dist["out.weight"] = new_w
    sd_dist["out.bias"] = new_b

for k in list(out_dict.keys()):
    if k.startswith("w_") or k.startswith("e0_") or k.startswith("e1_"):
        del out_dict[k]

for k, v in sd_dist.items():
    out_dict["w_" + k] = v.numpy().astype(np.float16)

out_dict["cw"] = np.array([1.0], dtype=np.float32)

np.savez_compressed(f"{T}/tmp_distilled_bounds.npz", **out_dict)

print(f"Building new zip {dst_zip}...", flush=True)
if os.path.exists(dst_zip): os.remove(dst_zip)
with zipfile.ZipFile(src_zip, 'r') as zin, zipfile.ZipFile(dst_zip, 'w', zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        if item.filename == "bounds_assets.npz":
            zout.writestr(item, open(f"{T}/tmp_distilled_bounds.npz", "rb").read())
        else:
            zout.writestr(item, zin.read(item.filename))

print("Evaluating distilled artifact...", flush=True)
subprocess.run(f"python3 {T}/eval_anchors_fast.py {dst_zip}", shell=True, check=True)
