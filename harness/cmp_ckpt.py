"""Exhaustive structural comparison of the packed checkpoint inside the WORKING
SOUP_v1 zip vs the FAILING MAXSOUP_v2 zip. submission.py / load_baseline.py /
head_assets.npz are all already ruled out, so any cause left must be here.
Compares keys, dtypes, shapes, container layout -- not values."""
import zipfile, io, torch, numpy as np, sys

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP/submissions"

def load_from_zip(z, name="sim_real_fno_fp16.pth"):
    zf = zipfile.ZipFile(z)
    return torch.load(io.BytesIO(zf.read(name)), map_location="cpu", weights_only=False)

a = load_from_zip(f"{B}/submission_SOUP_v1.zip")
b = load_from_zip(f"{B}/submission_MAXSOUP_v2.zip")

print("=== top-level container ===")
print("  v1     :", type(a).__name__, sorted(a.keys()) if isinstance(a, dict) else "")
print("  maxsoup:", type(b).__name__, sorted(b.keys()) if isinstance(b, dict) else "")

def unpack(d):
    if isinstance(d, dict) and "state_fp16" in d:
        return d["state_fp16"], set(d.get("complex_keys", []))
    return d, set()

sa, cka = unpack(a)
sb, ckb = unpack(b)
print("\n=== state dict ===")
print("  v1     : %d tensors, %d complex_keys" % (len(sa), len(cka)))
print("  maxsoup: %d tensors, %d complex_keys" % (len(sb), len(ckb)))
print("  keys identical:", set(sa.keys()) == set(sb.keys()))
print("  complex_keys identical:", cka == ckb)
ka, kb = set(sa.keys()), set(sb.keys())
if ka != kb:
    print("   only in v1     :", sorted(ka - kb))
    print("   only in maxsoup:", sorted(kb - ka))

print("\n=== per-tensor dtype / shape mismatches ===")
bad = 0
for k in sorted(ka & kb):
    ta, tb = sa[k], sb[k]
    da, db = (ta.dtype, tb.dtype)
    sha, shb = (tuple(ta.shape), tuple(tb.shape))
    if da != db or sha != shb:
        bad += 1
        print("  %-34s v1=%s%s   maxsoup=%s%s" % (k, da, sha, db, shb))
if bad == 0:
    print("  none -- every tensor matches in dtype and shape")

print("\n=== dtype census ===")
from collections import Counter
print("  v1     :", dict(Counter(str(v.dtype) for v in sa.values())))
print("  maxsoup:", dict(Counter(str(v.dtype) for v in sb.values())))

print("\n=== non-finite / anomalous values ===")
for tag, s in (("v1", sa), ("maxsoup", sb)):
    nbad = 0
    worst = None
    for k, v in s.items():
        vv = torch.view_as_real(v) if v.is_complex() else v
        if vv.dtype.is_floating_point:
            f = torch.isfinite(vv)
            if not f.all():
                nbad += 1; print("   %s: NON-FINITE in %s" % (tag, k))
            m = float(vv.abs().max())
            if worst is None or m > worst[1]: worst = (k, m)
    print("  %-8s non-finite tensors: %d   largest |value|: %s=%.6g" % (tag, nbad, worst[0], worst[1]))

print("\n=== does it actually load into the architecture? (what the eval does) ===")
sys.path.insert(0, "/SML_DISK_24TB/rajeshr/Aryamann/UGP/starting_kit_v9/realpde_t1_starting_kit_v9")
sys.path.insert(0, "/SML_DISK_24TB/rajeshr/Aryamann/UGP/starting_kit_v9/realpde_t1_starting_kit_v9/_vendor")
import warnings, tempfile, os, shutil
from load_baseline import load_baseline
for tag, z in (("v1", "submission_SOUP_v1.zip"), ("maxsoup", "submission_MAXSOUP_v2.zip")):
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "sim_real_fno_fp16.pth")
    with open(p, "wb") as f:
        f.write(zipfile.ZipFile(f"{B}/{z}").read("sim_real_fno_fp16.pth"))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            m, meta = load_baseline(p, device="cpu")
            print("  %-8s load OK   meta=%s   warnings=%d" % (tag, type(meta).__name__, len(w)))
            for wi in w[:5]:
                print("      WARN:", str(wi.message)[:110])
        except Exception as e:
            print("  %-8s LOAD FAILED: %s: %s" % (tag, type(e).__name__, e))
    shutil.rmtree(tmp, ignore_errors=True)
