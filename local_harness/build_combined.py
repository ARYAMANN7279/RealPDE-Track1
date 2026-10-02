"""Build the combined candidate: proven FNO (72.78) + moving-average temporal
smoothing (window W) along the 20-frame output + SPS-optimized bounds.
Usage: python build_combined.py <W> <hu> <hv>
"""
import os, re, sys, zipfile, shutil
W=int(sys.argv[1]); HU=float(sys.argv[2]); HV=float(sys.argv[3])
SUB="/Users/aryamannsrivastava/Desktop/sem7/UGP/submissions"
BUILD=f"/private/tmp/claude-501/-Users-aryamannsrivastava-Desktop/b65f478a-789b-4719-973f-1d580a53ff7d/scratchpad/fno_combined"
if os.path.exists(BUILD): shutil.rmtree(BUILD)
os.makedirs(BUILD)
with zipfile.ZipFile(f"{SUB}/submission_fno_calibrated.zip") as z: z.extractall(BUILD)
sp=os.path.join(BUILD,"submission.py"); s=open(sp).read()
# 1) new bound constant
s=re.sub(r"_HALF_BOUND = np\.array\(\[[^\]]*\]", f"_HALF_BOUND = np.array([{HU}, {HV}, 0.0]", s)
# 2) insert a temporal-smoothing helper + apply it right after `pred = np.concatenate(...)`
helper=f'''
_SMOOTH_W = {W}  # moving-average window along the 20-frame time axis (denoise FNO jitter)

def _smooth_time(a):
    w = _SMOOTH_W
    if w <= 1:
        return a
    k = w // 2
    ap = np.pad(a, ((0, 0), (k, k), (0, 0), (0, 0), (0, 0)), mode="edge")
    out = np.zeros_like(a)
    for i in range(w):
        out += ap[:, i:i + a.shape[1]]
    return (out / w).astype(np.float32)
'''
s=s.replace("def predict(input_array, metadata=None):", helper+"\ndef predict(input_array, metadata=None):",1)
s=s.replace("    pred = np.concatenate(outs, axis=0).astype(np.float32)\n",
            "    pred = np.concatenate(outs, axis=0).astype(np.float32)\n    pred = _smooth_time(pred)\n",1)
assert str(HU) in s and "_smooth_time(pred)" in s and f"_SMOOTH_W = {W}" in s, "patch failed"
open(sp,"w").write(s)
# repackage
for r,_,fs in os.walk(BUILD):
    for f in fs:
        if f.endswith(".pyc") or f==".DS_Store": os.remove(os.path.join(r,f))
for r,ds,_ in os.walk(BUILD):
    for dd in ds:
        if dd=="__pycache__": shutil.rmtree(os.path.join(r,dd))
out=f"{SUB}/submission_fno_smooth_sps.zip"
if os.path.exists(out): os.remove(out)
with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED) as z:
    for r,_,fs in os.walk(BUILD):
        for f in fs: z.write(os.path.join(r,f),os.path.relpath(os.path.join(r,f),BUILD))
print(f"wrote {out} ({os.path.getsize(out)/1e6:.1f} MB)  W={W} bounds=[{HU},{HV}]")
