"""GATE 2 + GATE 3 on the ACTUAL zips, candidate vs the artifact that scored 78.4566.

Only LUT *values* changed -- no shape, no code, no checkpoint -- so timing should be
identical and any difference in the output contract would be a red flag.
"""
import importlib.util, os, shutil, subprocess, sys, time
import numpy as np
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
X = np.load(f"{B}/local_harness/tr_frames.npy", mmap_mode="r")
import json
meta = json.load(open(f"{B}/local_harness/tr_meta.json")); off, lens = meta["off"], meta["lens"]
starts = [off[i] for i in range(0, len(lens), 7)][:64]
inp = np.stack([np.concatenate([np.asarray(X[s:s+20]), np.zeros((20,32,64,1), np.float32)], -1)
                for s in starts]).astype(np.float32)
print("probe batch:", inp.shape)

def run(zipname, tag):
    d = f"{B}/_tmp/gate23_{tag}"
    if os.path.exists(d): shutil.rmtree(d)
    os.makedirs(d); subprocess.run(["unzip","-q",f"{B}/submissions/{zipname}","-d",d], check=True)
    sys.path.insert(0, d)
    for m in list(sys.modules):
        if m in ("submission","load_baseline") or m.startswith("rpde_baselines") or m.startswith("einops"):
            del sys.modules[m]
    spec = importlib.util.spec_from_file_location(f"sub_{tag}", f"{d}/submission.py")
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    M.predict(inp[:4])                       # untimed warm-up, as the platform does
    t0 = time.time(); out = M.predict(inp); dt = time.time() - t0
    sys.path.remove(d)
    return out, dt

res = {}
for zipname, tag in (("submission_SOUP_v1.zip","ref"), ("submission_LUTFIX.zip","cand")):
    out, dt = run(zipname, tag)
    p, lo, up = out["prediction"], out["lower"], out["upper"]
    h = (up - lo)/2.0
    res[tag] = dict(dt=dt, p=p, h=h)
    print("\n=== %s ===" % zipname)
    print("  shapes      pred %s lower %s upper %s" % (p.shape, lo.shape, up.shape))
    print("  finite      pred %s lower %s upper %s" % (np.isfinite(p).all(), np.isfinite(lo).all(), np.isfinite(up).all()))
    print("  lower<=upper %s" % bool((lo <= up).all()))
    print("  p channel zero %s" % bool((p[...,2] == 0).all()))
    print("  bounds vary per element: h_u std %.5f  h_v std %.5f" % (h[...,0].std(), h[...,1].std()))
    print("  h_u  min %.4f med %.4f max %.4f" % (h[...,0].min(), np.median(h[...,0]), h[...,0].max()))
    print("  h_v  min %.4f med %.4f max %.4f" % (h[...,1].min(), np.median(h[...,1]), h[...,1].max()))
    print("  time %.3fs for %d samples = %.3f ms/sample" % (dt, len(inp), 1000*dt/len(inp)))

print("\n=== GATE 3: ratio to the artifact that really scored ===")
r = res["cand"]["dt"]/res["ref"]["dt"]
print("  candidate / SOUP_v1 runtime ratio: %.4f" % r)
print("  predictions byte-identical: %s" % np.array_equal(res["cand"]["p"], res["ref"]["p"]))
print("  (only LUT values changed, so predictions MUST be identical and time ~1.00)")
dh = res["cand"]["h"] - res["ref"]["h"]
print("  half-width change: u %+.5f median, v %+.5f median" % (np.median(dh[...,0]), np.median(dh[...,1])))
