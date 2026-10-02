"""Is the kit FNO simply OUT-OF-DISTRIBUTION on our downloaded data?

The model normalizes with mi/si = (0.15496, 0.09681) for u. Our shards are
u ~ 0.30 +/- 0.009, so (x-mi)/si is a near-constant +1.50 everywhere instead of a
structured field around 0. If that is the whole story, mapping our data into the
official distribution before inference should collapse the error.

Diagnostic metric: mean|err| / (target's own within-window std). Currently 6.8x
for u -- i.e. the FNO is ~7x worse than predicting the window mean. In-distribution
behaviour would be well under 1.
"""
import glob, os, sys, json
import numpy as np, torch, pyarrow as pa, pyarrow.ipc as ipc
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
SD=f"{B}/data/real_hf/foil/hf_dataset/real"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))
import importlib.util
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
from load_baseline import load_baseline
DEV="cuda:3"
MI=np.array([0.154960856,-0.000513992854,0.0],np.float32)
SI=np.array([0.0968056545,0.015960684,1.0],np.float32)
MT=np.array([0.154962569,-0.000517793698,0.0],np.float32)
ST=np.array([0.0968104079,0.0159636438,1.0],np.float32)
model=load_baseline(f"{B}/local_harness/fno_model/sim_real_fno_fp16.pth",device=DEV)[0]; model.eval()
mi,si,mt,st=[torch.tensor(a).to(DEV) for a in (MI,SI,MT,ST)]

def read(sp):
    with pa.memory_map(sp,"rb") as src:
        tb=ipc.open_stream(src).read_all()
        sid=tb.column("sim_id")[0].as_py()
        t,h,w=[tb.column(k)[0].as_py() for k in ("shape_t","shape_h","shape_w")]
        u=np.frombuffer(tb.column("u")[0].as_py(),np.float32).reshape(t,h,w)
        v=np.frombuffer(tb.column("v")[0].as_py(),np.float32).reshape(t,h,w)
    return sid,u[:,::4,::4][:,:32,:64],v[:,::4,::4][:,:32,:64]

IN=OUT=20; HOR=40
xs,ys=[],[]
for sp in sorted(glob.glob(f"{SD}/*.arrow")):
    sid,u,v=read(sp)
    for k in range((u.shape[0]-HOR)//HOR+1):
        t0=k*HOR; tr=np.stack([u[t0:t0+HOR],v[t0:t0+HOR],np.zeros_like(u[t0:t0+HOR])],-1)
        xs.append(tr[:IN]); ys.append(tr[IN:])
X=np.stack(xs).astype(np.float32); Y=np.stack(ys).astype(np.float32)
print("windows",X.shape,flush=True)
C=2

@torch.no_grad()
def run(Xin):
    o=[]
    for i in range(0,len(Xin),32):
        xb=torch.from_numpy(Xin[i:i+32]).to(DEV).float()
        o.append((model((xb-mi)/si)*st+mt).cpu().numpy())
    return np.concatenate(o,0).astype(np.float32)

def report(P,Y,tag):
    dm=S.rel_l2_per_sample(P,Y,C); tk=S.tke_rel_l2_per_sample(P,Y,C); mv=S.mvpe_rel_l2_per_sample(P,Y)
    n=lambda x:x/(0.5+x)
    W=float((0.5*(1-n(dm))+0.3*(1-np.nan_to_num(n(tk)))+0.2*(1-np.nan_to_num(n(mv)))).mean())
    h=np.zeros((1,1,1,1,C),np.float32); h[...,0]=0.030; h[...,1]=0.010
    Ec=S.score_sps(S.aggregate_sps(P,Y,C,lower=P[...,:C]-h,upper=P[...,:C]+h)[0])/(100*W)
    iv=0.1*np.abs(P[...,:C])
    Ep=S.score_sps(S.aggregate_sps(P,Y,C,lower=P[...,:C]-iv/2,upper=P[...,:C]+iv/2)[0])/(100*W)
    r=[]
    for ci,cn in enumerate("uv"):
        e=np.abs(P[...,ci]-Y[...,ci]).mean()
        sg=Y[...,ci].reshape(len(Y),-1).std(axis=1).mean()
        r.append(e/sg)
    print("%-26s rl %6.2f tke %6.2f mvpe %6.2f | E_c %.4f E_p %.4f | err/signal u=%5.2f v=%5.2f"%(
        tag,S.score_error(float(dm.mean())),S.score_error(float(tk.mean())),S.score_error(float(mv.mean())),
        Ec,Ep,r[0],r[1]),flush=True)

print("TARGET (real leaderboard)   rl  94.17 tke  74.03 mvpe  92.84 | E_c 0.4399 E_p 0.2076\n")
report(run(X),Y,"raw (current harness)")

# per-window standardisation into the official distribution, then invert
for name,axes in [("per-window",(1,2,3)),("per-window-per-frame",(2,3))]:
    m=X[...,:C].mean(axis=axes,keepdims=True); s=X[...,:C].std(axis=axes,keepdims=True).clip(1e-8)
    Xa=X.copy(); Xa[...,:C]=(X[...,:C]-m)/s*SI[:C]+MI[:C]
    Pa=run(Xa)
    Pi=Pa.copy(); Pi[...,:C]=(Pa[...,:C]-MT[:C])/ST[:C]*s+m
    report(Pi,Y,"adapt %s"%name)

# scale-only sweep: x' = (x-b)/a*si+mi  with b = window mean, a swept
m=X[...,:C].mean(axis=(1,2,3),keepdims=True)
for a in [0.002,0.004,0.006,0.009,0.015,0.03,0.06]:
    Xa=X.copy(); Xa[...,:C]=(X[...,:C]-m)/a*SI[:C]+MI[:C]
    Pa=run(Xa); Pi=Pa.copy(); Pi[...,:C]=(Pa[...,:C]-MT[:C])/ST[:C]*a+m
    report(Pi,Y,"adapt scale a=%.3f"%a)
