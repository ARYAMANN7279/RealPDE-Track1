"""With the input path fixed, determine the SIGN of the FNO's TKE error and test
whether rescaling the predicted temporal fluctuation improves tke.

pred = mean_t(pred) + alpha*(pred - mean_t(pred))
alpha>1 amplifies temporal fluctuation, alpha<1 damps it.
Memory 6 called this a dead end, but that was measured through the broken input path.
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
IN=20;HOR=40
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
m=X[...,:C].mean(axis=(1,2,3),keepdims=True); s=X[...,:C].std(axis=(1,2,3),keepdims=True).clip(1e-8)
Xa=X.copy(); Xa[...,:C]=(X[...,:C]-m)/s*SI[:C]+MI[:C]
Pa=run(Xa); P=Pa.copy(); P[...,:C]=(Pa[...,:C]-MT[:C])/ST[:C]*s+m

def ke(a):
    u,v=a[...,0],a[...,1]
    return 0.5*(np.mean((u-u.mean(1,keepdims=True))**2,1)+np.mean((v-v.mean(1,keepdims=True))**2,1))
kp,kt=ke(P),ke(Y)
print("=== SIGN of the temporal-fluctuation error (input path FIXED) ===")
print("  predicted KE mean = %.4e"%kp.mean())
print("  target    KE mean = %.4e"%kt.mean())
r=kp.mean()/kt.mean()
print("  ratio pred/target = %.3f  => FNO %s temporal fluctuation by %.2fx in std"%(
    r,"OVER-predicts" if r>1 else "UNDER-predicts (too smooth)",np.sqrt(max(r,1/r))))
print("  (through the BROKEN path this ratio read 15.01 -> the opposite conclusion)\n")

print("=== alpha sweep: pred = mean_t + alpha*(pred - mean_t) ===")
print("  %-7s %7s %7s %7s   %s"%("alpha","rel_l2","tke","mvpe","net final vs alpha=1"))
base=None
for al in [0.6,0.8,0.9,1.0,1.1,1.25,1.5,1.8,2.2,3.0]:
    Q=P.copy()
    mu=Q[...,:C].mean(axis=1,keepdims=True)
    Q[...,:C]=mu+al*(Q[...,:C]-mu)
    dm=S.rel_l2_per_sample(Q,Y,C);tk=S.tke_rel_l2_per_sample(Q,Y,C);mv=S.mvpe_rel_l2_per_sample(Q,Y)
    rl,tks,mvs=(S.score_error(float(dm.mean())),S.score_error(float(tk.mean())),S.score_error(float(mv.mean())))
    if al==1.0: base=(rl,tks,mvs)
    net="" if base is None else "%+.3f"%(0.306*(rl-base[0])+0.163*(tks-base[1])+0.218*(mvs-base[2]))
    print("  %-7.2f %7.2f %7.2f %7.2f   %s"%(al,rl,tks,mvs,net))
