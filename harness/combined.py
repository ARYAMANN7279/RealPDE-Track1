"""End-to-end validation of BOTH levers together, then export deployable maps.

Order matters: the tke gain changes the predictions, so the bounds must be
refit on the MODIFIED errors. Everything is fit on TRAIN trajectories and
scored on TEST trajectories (disjoint by trajectory).
"""
import numpy as np, json, sys, os
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT)
import importlib.util
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
SIG=S.SIGMA_GLOBAL; C=2
LAM=float(np.load(f"{B}/local_harness/calib_lambda.npy")[0])
d=np.load(f"{B}/local_harness/comp_eval_cache.npz")
P=d["P"].astype(np.float32); Y=d["Y"].astype(np.float32)
rows=json.load(open(f"{B}/local_harness/comp_anchor_rows.json"))
sim=np.concatenate([[r["sim"]]*r["n"] for r in rows])
sims=sorted(set(sim.tolist())); tr=np.isin(sim,sims[0::2]); te=np.isin(sim,sims[1::2])
SCM=(Y[...,:C]!=0.0)
n=lambda x:x/(0.5+x)
def subs(Q,m):
    dm=S.rel_l2_per_sample(Q[m],Y[m],C); tk=S.tke_rel_l2_per_sample(Q[m],Y[m],C); mv=S.mvpe_rel_l2_per_sample(Q[m],Y[m])
    return S.score_error(float(dm.mean())),S.score_error(float(tk.mean())),S.score_error(float(mv.mean()))

# ---- lever 2 fit on TRAIN: per-pixel temporal-variance gain ----
def ke(a):
    u,v=a[...,0],a[...,1]
    return 0.5*(np.mean((u-u.mean(1,keepdims=True))**2,1)+np.mean((v-v.mean(1,keepdims=True))**2,1))
kp,kt=ke(P),ke(Y)
num=(kp[tr]*kt[tr]).sum(0); den=(kp[tr]*kp[tr]).sum(0).clip(1e-20)
GAIN=np.sqrt(np.clip(num/den,0.25,4.0)).astype(np.float32)
Q=P.copy(); mu=Q[...,:C].mean(axis=1,keepdims=True)
Q[...,:C]=mu+GAIN[None,None,:,:,None]*(Q[...,:C]-mu)

a=subs(P,te); b=subs(Q,te)
print("=== accuracy on TEST ===")
print("  before: rel_l2 %.2f tke %.2f mvpe %.2f"%a)
print("  after : rel_l2 %.2f tke %.2f mvpe %.2f   (d_rl %+.2f d_tke %+.2f d_mvpe %+.2f)"%(
    b[0],b[1],b[2],b[0]-a[0],b[1]-a[1],b[2]-a[2]),flush=True)

# ---- lever 1 refit on TRAIN using MODIFIED predictions ----
ERR=(np.abs(Q[...,:C]-Y[...,:C])/LAM).astype(np.float32)
def best_h(v):
    v=np.sort(v)
    if v.size==0: return 0.0
    k=np.arange(1,v.size+1)/v.size
    return float(v[np.argmax(np.exp(-2*v/SIG)*k)])
Hh,W_=P.shape[2],P.shape[3]
HL=np.zeros((Hh,W_,C),np.float32)
for ci in range(C):
    E=ERR[tr][...,ci]; M=SCM[tr][...,ci]
    for i in range(Hh):
        for j in range(W_):
            HL[i,j,ci]=best_h(E[:,:,i,j][M[:,:,i,j]][::3])
u,v=Q[...,0],Q[...,1]
g=np.sqrt(np.gradient(u,axis=3)**2+np.gradient(u,axis=2)**2
         +np.gradient(v,axis=3)**2+np.gradient(v,axis=2)**2)
QT=np.quantile(g[tr][::11],np.linspace(0,1,11)[1:-1]).astype(np.float32)
bid=np.digitize(g,QT)
MULT=np.ones((10,C),np.float32)
for ci in range(C):
    for bb in range(10):
        sel=(bid[tr]==bb)&SCM[tr][...,ci]
        if sel.sum()<1000: continue
        base=np.broadcast_to(HL[None,None,:,:,ci],ERR[...,ci].shape)[tr][sel]
        e=ERR[tr][...,ci][sel]
        best=(1.0,-1.0)
        for m_ in np.linspace(0.3,3.0,28):
            h=base*m_; val=float((np.exp(-2*h/SIG)*(e<=h)).mean())
            if val>best[1]: best=(m_,val)
        MULT[bb,ci]=best[0]
def score_bounds(m,Hmap,mult):
    e=ERR[m]; s=SCM[m]
    h=np.broadcast_to(Hmap[None,None],e.shape).copy()
    for ci in range(C): h[...,ci]*=mult[bid[m],ci]
    ok=(e<=h)&s; ns=s.sum()
    return float((np.exp(-2*h/SIG)*ok).sum()/ns), float(ok.sum()/ns)
Hb=np.zeros((Hh,W_,C),np.float32); Hb[...,0]=0.030; Hb[...,1]=0.010
E0,c0=score_bounds(te,Hb,np.ones((10,C),np.float32))
E1,c1=score_bounds(te,HL,MULT)
W_new=0.5*(1-n((100/b[0]-1)*2))+0.3*(1-n((100/b[1]-1)*2))+0.2*(1-n((100/b[2]-1)*2))
WR=0.6784
print("\n=== bounds on TEST (refit on modified predictions) ===")
print("  baseline [0.030,0.010] E=%.4f cov=%.3f -> real sps %.2f"%(E0,c0,100*WR*E0))
print("  per-loc x grad-decile  E=%.4f cov=%.3f -> real sps %.2f"%(E1,c1,100*WR*E1))
RL,TK,MV,TM=94.168150,74.025866,92.836278,91.32
fin=lambda rl,tk,mv,sps:0.306*rl+0.163*tk+0.218*mv+0.100*TM+0.217*sps
base_f=fin(RL,TK,MV,29.84)
rl2,tk2=RL+(b[0]-a[0]),TK+(b[1]-a[1]); mv2=MV+(b[2]-a[2])
Wn=0.5*(1-n((100/rl2-1)*2))+0.3*(1-n((100/tk2-1)*2))+0.2*(1-n((100/mv2-1)*2))
sps_new=100*Wn*E1
print("\n=== projected on the real leaderboard (deltas vs formula baseline) ===")
print("  rel_l2 %.2f  tke %.2f  mvpe %.2f  sps %.2f"%(rl2,tk2,mv2,sps_new))
print("  delta final = %+.2f   (77.20 -> ~%.2f)"%(fin(rl2,tk2,mv2,sps_new)-base_f,
                                                  77.20+fin(rl2,tk2,mv2,sps_new)-base_f))
np.savez(f"{B}/local_harness/deploy_maps.npz",gain=GAIN,hmap=HL,mult=MULT,qt=QT)
print("\nexported -> deploy_maps.npz  gain%s hmap%s mult%s qt%s"%(GAIN.shape,HL.shape,MULT.shape,QT.shape))
