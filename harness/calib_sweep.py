"""Find an eval-set construction that reproduces the kit FNO's REAL anchors.

Falsifiable target (all from the live leaderboard, not the proxy):
    rel_l2 94.168   tke 74.026   mvpe 92.836
    E = 0.4399 at constant half-widths [0.030, 0.010]
    E = 0.2076 at proportional half-width 0.05*|pred|
where E = sps/(100*W) is the interval-efficiency term of SPS.

Sweeps window construction (temporal stride, spatial reduction) over the real
shards and scores the kit FNO on each. A config matching all five is a
trustworthy proxy; anything less is not.
"""
import glob, json, os, sys, itertools
import numpy as np, torch
import pyarrow as pa, pyarrow.ipc as ipc

B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
SD=f"{B}/data/real_hf/foil/hf_dataset/real"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))
import importlib.util
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
scoring=importlib.util.module_from_spec(spec); spec.loader.exec_module(scoring)
from load_baseline import load_baseline
SIG=scoring.SIGMA_GLOBAL
DEV="cuda:3"

ANCH=dict(rl=94.168150,tke=74.025866,mv=92.836278,E_const=0.4399,E_prop=0.2076)

mi=torch.tensor([0.154960856,-0.000513992854,0.0]);si=torch.tensor([0.0968056545,0.015960684,1.0])
mt=torch.tensor([0.154962569,-0.000517793698,0.0]);st=torch.tensor([0.0968104079,0.0159636438,1.0])
model=load_baseline(f"{B}/local_harness/fno_model/sim_real_fno_fp16.pth",device=DEV)[0]
model.eval()
mi,si,mt,st=[t.to(DEV) for t in (mi,si,mt,st)]

def read(sp):
    with pa.memory_map(sp,"rb") as src:
        tb=ipc.open_stream(src).read_all()
        sid=tb.column("sim_id")[0].as_py()
        t,h,w=[tb.column(k)[0].as_py() for k in ("shape_t","shape_h","shape_w")]
        u=np.frombuffer(tb.column("u")[0].as_py(),dtype=np.float32).reshape(t,h,w)
        v=np.frombuffer(tb.column("v")[0].as_py(),dtype=np.float32).reshape(t,h,w)
    return sid,u,v

def reduce_sp(a,mode):
    if mode=="stride4":  return a[:,::4,::4][:,:32,:64]
    if mode=="avgpool4":
        T=a.shape[0]; return a[:,:128,:256].reshape(T,32,4,64,4).mean(axis=(2,4))
    if mode=="stride2crop": return a[:,::2,::2][:,:32,:64]
    raise ValueError(mode)

@torch.no_grad()
def predict(X):
    out=[]
    for i in range(0,len(X),32):
        xb=torch.from_numpy(X[i:i+32]).to(DEV).float()
        out.append((model((xb-mi)/si)*st+mt).cpu().numpy())
    return np.concatenate(out,0).astype(np.float32)

def E_of(P,Y,C,lower,upper,W):
    s,_=scoring.aggregate_sps(P,Y,C,lower=lower,upper=upper)
    return scoring.score_sps(s)/(100*W)

shards=sorted(glob.glob(f"{SD}/*.arrow"))
print("shards:",len(shards),flush=True)
cache={}
for sp in shards:
    sid,u,v=read(sp); cache[sid]=(u,v)
    print("  loaded",sid,u.shape,flush=True)

IN=OUT=20
results=[]
for tstride,spmode in itertools.product([1,2,4,8],["stride4","avgpool4","stride2crop"]):
    HOR=(IN+OUT)*tstride
    xs,ys=[],[]
    for sid,(u,v) in cache.items():
        us=reduce_sp(u,spmode); vs=reduce_sp(v,spmode)
        T=us.shape[0]
        for k in range((T-HOR)//HOR+1):
            t0=k*HOR
            iu=us[t0:t0+HOR:tstride][:IN+OUT]; iv=vs[t0:t0+HOR:tstride][:IN+OUT]
            if iu.shape[0]<IN+OUT: break
            tr=np.stack([iu,iv,np.zeros_like(iu)],axis=-1)
            xs.append(tr[:IN]); ys.append(tr[IN:])
    if not xs: continue
    X=np.stack(xs).astype(np.float32); Y=np.stack(ys).astype(np.float32)
    P=predict(X); C=2
    n=lambda x:x/(0.5+x)
    dm=scoring.rel_l2_per_sample(P,Y,C); tk=scoring.tke_rel_l2_per_sample(P,Y,C); mv=scoring.mvpe_rel_l2_per_sample(P,Y)
    rl_s=scoring.score_error(float(dm.mean())); tk_s=scoring.score_error(float(tk.mean())); mv_s=scoring.score_error(float(mv.mean()))
    W=float((0.5*(1-n(dm))+0.3*(1-np.nan_to_num(n(tk)))+0.2*(1-np.nan_to_num(n(mv)))).mean())
    h=np.zeros((1,1,1,1,C),np.float32); h[...,0]=0.030; h[...,1]=0.010
    Ec=E_of(P,Y,C,P[...,:C]-h,P[...,:C]+h,W)
    iv2=0.1*np.abs(P[...,:C]); Ep=E_of(P,Y,C,P[...,:C]-iv2/2,P[...,:C]+iv2/2,W)
    err=(abs(rl_s-ANCH["rl"])+abs(tk_s-ANCH["tke"])+abs(mv_s-ANCH["mv"])
         +100*abs(Ec-ANCH["E_const"])+100*abs(Ep-ANCH["E_prop"]))
    r=dict(tstride=tstride,spmode=spmode,n=len(X),rl=rl_s,tke=tk_s,mvpe=mv_s,W=W,E_const=Ec,E_prop=Ep,anchor_err=err)
    results.append(r)
    print("t=%d %-11s n=%4d | rl %6.2f tke %6.2f mvpe %6.2f | E_c %.4f E_p %.4f | anchor_err %7.2f"%(
        tstride,spmode,len(X),rl_s,tk_s,mv_s,Ec,Ep,err),flush=True)

results.sort(key=lambda r:r["anchor_err"])
json.dump(results,open(f"{B}/local_harness/calib_sweep_results.json","w"),indent=1)
print("\nBEST:",json.dumps(results[0],indent=1))
print("TARGET: rl 94.17 tke 74.03 mvpe 92.84 E_c 0.4399 E_p 0.2076")
