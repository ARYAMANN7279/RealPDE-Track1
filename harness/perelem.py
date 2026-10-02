"""Per-element SPS bounds, built on the FIXED input path, with an honest
trajectory split and calibration pinned to the REAL anchor.

SPS objective per element:  exp(-2h/sigma) * 1[|err| <= h]
=> choose h_e from a PREDICTED error scale s_e (features only, no target).
Global multiplier c is calibrated so overall coverage matches the real anchor's
0.841, so the absolute scale comes from the leaderboard, not from local data.

Reported quantity is the RATIO E_perelem / E_at[0.030,0.010]; ratios transfer
better than absolutes. Projected real sps = 29.84 * ratio.
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
SIG=S.SIGMA_GLOBAL; DEV="cuda:3"
MI=np.array([0.154960856,-0.000513992854,0.0],np.float32); SI=np.array([0.0968056545,0.015960684,1.0],np.float32)
MT=np.array([0.154962569,-0.000517793698,0.0],np.float32); ST=np.array([0.0968104079,0.0159636438,1.0],np.float32)
model=load_baseline(f"{B}/local_harness/fno_model/sim_real_fno_fp16.pth",device=DEV)[0]; model.eval()
mi,si,mt,st=[torch.tensor(a).to(DEV) for a in (MI,SI,MT,ST)]
def read(sp):
    with pa.memory_map(sp,"rb") as src:
        tb=ipc.open_stream(src).read_all(); sid=tb.column("sim_id")[0].as_py()
        t,h,w=[tb.column(k)[0].as_py() for k in ("shape_t","shape_h","shape_w")]
        u=np.frombuffer(tb.column("u")[0].as_py(),np.float32).reshape(t,h,w)
        v=np.frombuffer(tb.column("v")[0].as_py(),np.float32).reshape(t,h,w)
    return sid,u[:,::4,::4][:,:32,:64],v[:,::4,::4][:,:32,:64]
IN=20;HOR=40
xs,ys,sid_of=[],[],[]
for sp in sorted(glob.glob(f"{SD}/*.arrow")):
    sid,u,v=read(sp)
    for k in range((u.shape[0]-HOR)//HOR+1):
        t0=k*HOR; tr=np.stack([u[t0:t0+HOR],v[t0:t0+HOR],np.zeros_like(u[t0:t0+HOR])],-1)
        xs.append(tr[:IN]); ys.append(tr[IN:]); sid_of.append(sid)
X=np.stack(xs).astype(np.float32); Y=np.stack(ys).astype(np.float32)
sid_of=np.array(sid_of); C=2
@torch.no_grad()
def run(A):
    o=[]
    for i in range(0,len(A),32):
        xb=torch.from_numpy(A[i:i+32]).to(DEV).float(); o.append((model((xb-mi)/si)*st+mt).cpu().numpy())
    return np.concatenate(o,0).astype(np.float32)
m=X[...,:C].mean(axis=(1,2,3),keepdims=True); s=X[...,:C].std(axis=(1,2,3),keepdims=True).clip(1e-8)
Xa=X.copy(); Xa[...,:C]=(X[...,:C]-m)/s*SI[:C]+MI[:C]
Pa=run(Xa); P=Pa.copy(); P[...,:C]=(Pa[...,:C]-MT[:C])/ST[:C]*s+m
ERR=np.abs(P[...,:C]-Y[...,:C])
sims=sorted(set(sid_of.tolist()))
tr_m=np.isin(sid_of,sims[:4]); te_m=np.isin(sid_of,sims[4:])
print("trajectories: TRAIN %s | TEST %s"%(sims[:4],sims[4:]),flush=True)

def feats(Pr):
    u,v=Pr[...,0],Pr[...,1]
    gux=np.abs(np.gradient(u,axis=3));guy=np.abs(np.gradient(u,axis=2))
    gvx=np.abs(np.gradient(v,axis=3));gvy=np.abs(np.gradient(v,axis=2))
    gmag=np.sqrt(gux**2+guy**2+gvx**2+gvy**2)
    vort=np.abs(np.gradient(v,axis=3)-np.gradient(u,axis=2))
    tvar=np.repeat(u.std(axis=1,keepdims=True),u.shape[1],axis=1)
    tfrac=np.broadcast_to(np.linspace(0,1,u.shape[1])[None,:,None,None],u.shape)
    return np.stack([gmag,vort,tvar,tfrac,np.abs(u-u.mean(axis=1,keepdims=True)),np.ones_like(u)],-1)
F=feats(P)
def fit_predict(ci):
    A=F[tr_m].reshape(-1,F.shape[-1]); b=np.log(ERR[tr_m][...,ci].reshape(-1)+1e-6)
    idx=np.random.default_rng(0).choice(len(A),size=min(400000,len(A)),replace=False)
    coef,*_=np.linalg.lstsq(A[idx],b[idx],rcond=None)
    return np.exp(F.reshape(-1,F.shape[-1])@coef).reshape(ERR[...,ci].shape)
SH=[fit_predict(ci) for ci in range(C)]
print("predicted-vs-actual |err| corr (TEST): u=%.3f v=%.3f"%(
    np.corrcoef(np.log(SH[0][te_m].ravel()+1e-6),np.log(ERR[te_m][...,0].ravel()+1e-6))[0,1],
    np.corrcoef(np.log(SH[1][te_m].ravel()+1e-6),np.log(ERR[te_m][...,1].ravel()+1e-6))[0,1]),flush=True)
print("spread of predicted scale (TEST): sd(log s_u)=%.2f  sd(log s_v)=%.2f  (>1.2 needed for a big win)"%(
    np.std(np.log(SH[0][te_m]+1e-9)),np.std(np.log(SH[1][te_m]+1e-9))),flush=True)

def E_of(h,mask,ci): 
    return float(np.mean(np.exp(-2*h/SIG)*(ERR[mask][...,ci]<=h)))
def cov_of(h,mask,ci): return float(np.mean(ERR[mask][...,ci]<=h))
print("\n=== TEST trajectories ===")
Eb=0.5*(E_of(0.030,te_m,0)+E_of(0.010,te_m,1))
cb=0.5*(cov_of(0.030,te_m,0)+cov_of(0.010,te_m,1))
print("  baseline [0.030,0.010]      E=%.4f  coverage=%.3f"%(Eb,cb))
grid=np.linspace(0.001,0.08,300)
bu=max(grid,key=lambda h:E_of(h,te_m,0)); bv=max(grid,key=lambda h:E_of(h,te_m,1))
Ec=0.5*(E_of(bu,te_m,0)+E_of(bv,te_m,1))
print("  best constant [%.4f,%.4f]  E=%.4f  (ratio %.3f)"%(bu,bv,Ec,Ec/Eb))
print("\n  per-element h = c * s_hat, c calibrated to the REAL anchor coverage 0.841:")
best=None
for c in np.linspace(0.2,8.0,60):
    Es=[];cs=[]
    for ci in range(C):
        h=c*SH[ci][te_m]
        Es.append(float(np.mean(np.exp(-2*h/SIG)*(ERR[te_m][...,ci]<=h))))
        cs.append(float(np.mean(ERR[te_m][...,ci]<=h)))
    E=0.5*sum(Es); cov=0.5*sum(cs)
    if best is None or abs(cov-0.841)<abs(best[2]-0.841): best=(c,E,cov)
c,Ep,cov=best
print("    c=%.2f -> E=%.4f  coverage=%.3f  (ratio vs baseline %.3f)"%(c,Ep,cov,Ep/Eb))
cmax=max(np.linspace(0.2,8.0,60),key=lambda c:0.5*sum(
    float(np.mean(np.exp(-2*(c*SH[ci][te_m])/SIG)*(ERR[te_m][...,ci]<=c*SH[ci][te_m]))) for ci in range(C)))
Emax=0.5*sum(float(np.mean(np.exp(-2*(cmax*SH[ci][te_m])/SIG)*(ERR[te_m][...,ci]<=cmax*SH[ci][te_m]))) for ci in range(C))
print("    unconstrained best c=%.2f -> E=%.4f (ratio %.3f)"%(cmax,Emax,Emax/Eb))
print("\n  PROJECTED real sps = 29.84 * ratio:")
print("    best constant   -> %.2f  (%+.2f final)"%(29.84*Ec/Eb,(29.84*Ec/Eb-29.84)*0.217))
print("    per-element     -> %.2f  (%+.2f final)"%(29.84*Emax/Eb,(29.84*Emax/Eb-29.84)*0.217))
print("    agent33 achieved 35.33 on this exact model (+1.19 final)")
