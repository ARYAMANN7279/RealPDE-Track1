"""Is the per-window normalisation wrapper SAFE on data that is already
in-distribution? If yes it is a low-risk submission change; if it degrades
in-distribution data it must not be submitted.

Build an in-distribution surrogate by permanently mapping our windows (input AND
target) into the official distribution, then compare the model with and without
the wrapper on it.
"""
import glob, os, sys
import numpy as np, torch, pyarrow as pa, pyarrow.ipc as ipc
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
SD=f"{B}/data/real_hf/foil/hf_dataset/real"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))
import importlib.util
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
from load_baseline import load_baseline
DEV="cuda:3"
MI=np.array([0.154960856,-0.000513992854,0.0],np.float32); SI=np.array([0.0968056545,0.015960684,1.0],np.float32)
MT=np.array([0.154962569,-0.000517793698,0.0],np.float32); ST=np.array([0.0968104079,0.0159636438,1.0],np.float32)
model=load_baseline(f"{B}/local_harness/fno_model/sim_real_fno_fp16.pth",device=DEV)[0]; model.eval()
mi,si,mt,st=[torch.tensor(a).to(DEV) for a in (MI,SI,MT,ST)]
def read(sp):
    with pa.memory_map(sp,"rb") as src:
        tb=ipc.open_stream(src).read_all()
        t,h,w=[tb.column(k)[0].as_py() for k in ("shape_t","shape_h","shape_w")]
        u=np.frombuffer(tb.column("u")[0].as_py(),np.float32).reshape(t,h,w)
        v=np.frombuffer(tb.column("v")[0].as_py(),np.float32).reshape(t,h,w)
    return u[:,::4,::4][:,:32,:64],v[:,::4,::4][:,:32,:64]
IN=HOR=20; HOR=40
xs,ys=[],[]
for sp in sorted(glob.glob(f"{SD}/*.arrow")):
    u,v=read(sp)
    for k in range((u.shape[0]-HOR)//HOR+1):
        t0=k*HOR; tr=np.stack([u[t0:t0+HOR],v[t0:t0+HOR],np.zeros_like(u[t0:t0+HOR])],-1)
        xs.append(tr[:IN]); ys.append(tr[IN:])
X=np.stack(xs).astype(np.float32); Y=np.stack(ys).astype(np.float32); C=2
@torch.no_grad()
def run(A):
    o=[]
    for i in range(0,len(A),32):
        xb=torch.from_numpy(A[i:i+32]).to(DEV).float(); o.append((model((xb-mi)/si)*st+mt).cpu().numpy())
    return np.concatenate(o,0).astype(np.float32)
def rep(P,T,tag):
    dm=S.rel_l2_per_sample(P,T,C); tk=S.tke_rel_l2_per_sample(P,T,C); mv=S.mvpe_rel_l2_per_sample(P,T)
    n=lambda x:x/(0.5+x)
    W=float((0.5*(1-n(dm))+0.3*(1-np.nan_to_num(n(tk)))+0.2*(1-np.nan_to_num(n(mv)))).mean())
    print("  %-34s rl %6.2f  tke %6.2f  mvpe %6.2f"%(tag,S.score_error(float(dm.mean())),
          S.score_error(float(tk.mean())),S.score_error(float(mv.mean()))),flush=True)
    return S.score_error(float(dm.mean())),S.score_error(float(tk.mean())),S.score_error(float(mv.mean()))

def wrap(A):
    m=A[...,:C].mean(axis=(1,2,3),keepdims=True); s=A[...,:C].std(axis=(1,2,3),keepdims=True).clip(1e-8)
    Aa=A.copy(); Aa[...,:C]=(A[...,:C]-m)/s*SI[:C]+MI[:C]
    P=run(Aa); Pi=P.copy(); Pi[...,:C]=(P[...,:C]-MT[:C])/ST[:C]*s+m
    return Pi

print("=== 1. OUR data (out-of-distribution) ===")
rep(run(X),Y,"no wrapper"); rep(wrap(X),Y,"WITH wrapper")

print("\n=== 2. IN-DISTRIBUTION surrogate (X and Y both mapped to official stats) ===")
m=X[...,:C].mean(axis=(1,2,3),keepdims=True); s=X[...,:C].std(axis=(1,2,3),keepdims=True).clip(1e-8)
Xi=X.copy(); Yi=Y.copy()
Xi[...,:C]=(X[...,:C]-m)/s*SI[:C]+MI[:C]
Yi[...,:C]=(Y[...,:C]-m)/s*SI[:C]+MI[:C]
a=rep(run(Xi),Yi,"no wrapper (baseline)")
b=rep(wrap(Xi),Yi,"WITH wrapper")
print("\n  wrapper effect on ALREADY in-distribution data: rl %+.2f  tke %+.2f  mvpe %+.2f"%(
    b[0]-a[0],b[1]-a[1],b[2]-a[2]))
print("  => if these are ~0 the wrapper is safe to submit; if negative it is not.")
