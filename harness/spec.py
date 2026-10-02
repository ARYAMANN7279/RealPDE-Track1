"""SPECTRAL CORRECTION for the tke gap.

Literature (Spectral-Refiner, DINO, adv-NO) is unanimous: FNO under-penalises
high-wavenumber error under an Lp loss and so LOSES ENERGY at high k. We measured
exactly that: predicted temporal KE / target = 0.637.

Global amplitude scaling already failed (it is 'pattern, not magnitude'), but that
is a single scalar. A PER-WAVENUMBER gain is the targeted version: restore the
spectrum without touching phase. Fit g(k) on TRAIN trajectories, apply to TEST.
Tests spatial-radial, temporal, and combined.
"""
import json,os,sys
import numpy as np
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT)
import importlib.util
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
C=2
d=np.load(f"{B}/local_harness/comp_eval_cache.npz")
P=d["P"].astype(np.float32); Y=d["Y"].astype(np.float32)
rows=json.load(open(f"{B}/local_harness/comp_anchor_rows.json"))
sim=np.concatenate([[r["sim"]]*r["n"] for r in rows]); sims=sorted(set(sim.tolist()))
tr=np.isin(sim,sims[0::2]); te=~tr
def sub(Q,m):
    dm=S.rel_l2_per_sample(Q[m],Y[m],C);tk=S.tke_rel_l2_per_sample(Q[m],Y[m],C);mv=S.mvpe_rel_l2_per_sample(Q[m],Y[m])
    return (S.score_error(float(dm.mean())),S.score_error(float(tk.mean())),S.score_error(float(mv.mean())))
a0=sub(P,te)
print("baseline (held-out): rel_l2 %.2f  tke %.2f  mvpe %.2f"%a0,flush=True)
net=lambda a: 0.306*(a[0]-a0[0])+0.163*(a[1]-a0[1])+0.218*(a[2]-a0[2])

# ---------- 1. TEMPORAL spectral gain ----------
print("\n=== 1. temporal per-frequency gain (fit on TRAIN) ===",flush=True)
def tfft(A): return np.fft.rfft(A[...,:C],axis=1)
Ft_p,Ft_y=tfft(P[tr]),tfft(Y[tr])
pw_p=(np.abs(Ft_p)**2).mean(axis=(0,2,3)); pw_y=(np.abs(Ft_y)**2).mean(axis=(0,2,3))
gt=np.sqrt(pw_y/np.maximum(pw_p,1e-20))
print("  temporal gain by freq (u): ", " ".join("%.2f"%x for x in gt[:,0][:8]))
def apply_t(Q,g,cap):
    F=np.fft.rfft(Q[...,:C],axis=1); gg=np.clip(g,1/cap,cap)
    F=F*gg[None,:,None,None,:]
    out=Q.copy(); out[...,:C]=np.fft.irfft(F,n=Q.shape[1],axis=1); return out
for cap in (1.5,2.0,3.0,6.0):
    a=sub(apply_t(P,gt,cap),te)
    print("  cap %.1f -> rel_l2 %.2f tke %.2f mvpe %.2f | net %+.3f"%(cap,a[0],a[1],a[2],net(a)),flush=True)

# ---------- 2. SPATIAL radial gain ----------
print("\n=== 2. spatial radial per-wavenumber gain (fit on TRAIN) ===",flush=True)
H,Wd=32,64
ky=np.fft.fftfreq(H)[:,None]*H; kx=np.fft.rfftfreq(Wd)[None,:]*Wd
KR=np.sqrt(ky**2+kx**2); NB=18
edges=np.linspace(0,KR.max()+1e-6,NB+1); bid=np.clip(np.digitize(KR,edges)-1,0,NB-1)
def sfft(A): return np.fft.rfft2(A[...,:C],axes=(2,3))
Fp,Fy=sfft(P[tr]),sfft(Y[tr])
gs=np.ones((NB,C),np.float32)
for ci in range(C):
    pp=(np.abs(Fp[...,ci])**2).mean(axis=(0,1)); yy=(np.abs(Fy[...,ci])**2).mean(axis=(0,1))
    for b in range(NB):
        m=bid==b
        if m.sum()>0: gs[b,ci]=np.sqrt(yy[m].mean()/max(pp[m].mean(),1e-20))
print("  radial gain (u), low->high k:", " ".join("%.2f"%x for x in gs[:,0][::2]))
def apply_s(Q,g,cap):
    F=np.fft.rfft2(Q[...,:C],axes=(2,3)); gg=np.clip(g,1/cap,cap)
    G=gg[bid]                      # (H,W/2+1,C)
    F=F*G[None,None,:,:,:]
    out=Q.copy(); out[...,:C]=np.fft.irfft2(F,s=(H,Wd),axes=(2,3)); return out
best=None
for cap in (1.2,1.5,2.0,3.0):
    a=sub(apply_s(P,gs,cap),te); n=net(a)
    print("  cap %.1f -> rel_l2 %.2f tke %.2f mvpe %.2f | net %+.3f"%(cap,a[0],a[1],a[2],n),flush=True)
    if best is None or n>best[0]: best=(n,cap)
# ---------- 3. combined ----------
print("\n=== 3. spatial then temporal ===",flush=True)
for cs in (1.5,2.0):
    for ct in (1.5,2.0):
        a=sub(apply_t(apply_s(P,gs,cs),gt,ct),te)
        print("  spatial %.1f + temporal %.1f -> rel_l2 %.2f tke %.2f mvpe %.2f | net %+.3f"%(
            cs,ct,a[0],a[1],a[2],net(a)),flush=True)
print("\n(net = change in final from accuracy alone; sps gains would come on top)")
