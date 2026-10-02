import torch
LH="/SML_DISK_24TB/rajeshr/Aryamann/UGP/local_harness"
full = torch.load(f"{LH}/soup_v2.pth", map_location="cpu"); full = full.get("model_state_dict", full)
packed = torch.load(f"{LH}/soup_v2_fp16.pth", map_location="cpu")
sf, ck = packed["state_fp16"], set(packed["complex_keys"])
recon = {}
for k, v in sf.items():
    recon[k] = torch.view_as_complex(v.float()) if k in ck else v.float()

rows = []
for k in full:
    a, b = full[k], recon[k]
    if a.is_complex():
        d = float((a - b).abs().max()); mag = float(a.abs().max())
    elif torch.is_floating_point(a):
        d = float((a.float() - b).abs().max()); mag = float(a.float().abs().max())
    else:
        continue
    rows.append((d, k, mag, tuple(a.shape), str(a.dtype)))
rows.sort(reverse=True)
print("worst 8 by absolute diff:")
for d, k, mag, shp, dt in rows[:8]:
    print("  diff %.5f  key=%-30s  max|value| %.5f  shape %s  dtype %s  rel %.2e" % (d, k, mag, shp, dt, d/max(mag,1e-9)))
print("\nworst 8 by RELATIVE diff (diff/max|value|):")
rel = sorted(rows, key=lambda r: r[0]/max(r[2],1e-9), reverse=True)
for d, k, mag, shp, dt in rel[:8]:
    print("  rel %.2e  diff %.5f  key=%-30s  max|value| %.5f" % (d/max(mag,1e-9), d, k, mag))
