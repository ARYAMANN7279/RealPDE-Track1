import sys, numpy as np, os, shutil, subprocess, importlib.util, time, json
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
X=np.load(f"{B}/local_harness/tr_frames.npy",mmap_mode="r")
meta=json.load(open(f"{B}/local_harness/tr_meta.json")); off,lens=meta["off"],meta["lens"]
starts=[]
for i in range(len(lens)):
    starts += list(range(off[i], off[i]+lens[i]-20, 120))
starts=starts[:256]
inp=np.stack([np.concatenate([np.asarray(X[s:s+20]),np.zeros((20,32,64,1),np.float32)],-1)
              for s in starts]).astype(np.float32)

def run(zipname,tag):
    d=f"{B}/_tmp/g23_{tag}_test"
    if os.path.exists(d): shutil.rmtree(d)
    os.makedirs(d)
    subprocess.run(["unzip","-q",f"{B}/submissions/{zipname}","-d",d],check=True)
    sys.path.insert(0,d)
    for m in list(sys.modules):
        if m in ("submission","load_baseline") or m.startswith("rpde_baselines") or m.startswith("einops"):
            del sys.modules[m]
    spec=importlib.util.spec_from_file_location(f"sub_{tag}",f"{d}/submission.py")
    M=importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    M.predict(inp[:4])
    t0=time.time(); out=M.predict(inp); dt=time.time()-t0
    sys.path.remove(d)
    return out, dt

try:
    print("Running REF...")
    ref_out, ref_dt = run("submission_SOUP_v1.zip", "ref")
    print("Running CAND...")
    cand_out, cand_dt = run("submission_ASYM_W96_a85.zip", "cand")
    
    h_ref = (ref_out["upper"] - ref_out["lower"])/2.0
    h_cand = (cand_out["upper"] - cand_out["lower"])/2.0
    
    r_u = np.median(h_cand[...,0])/np.median(h_ref[...,0])
    r_v = np.median(h_cand[...,1])/np.median(h_ref[...,1])
    print(f"h_u ratio med {r_u:.4f} | h_v ratio med {r_v:.4f}")
except Exception as e:
    import traceback
    traceback.print_exc()
