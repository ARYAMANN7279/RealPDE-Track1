"""Rebuild the eval set on the OFFICIAL window index and test it against the
REAL leaderboard anchors.

data/hf_meta/foil/hf_dataset/test_index_real.json gives 5140 exact
(sim_id, time_id) pairs across all 98 trajectories. A window is
input = frames[t : t+20], target = frames[t+20 : t+40], at 32x64 via [::4,::4].
This is the official construction; the old harness invented its own windows on
9 trajectories.

ANCHORS (kit FNO, live leaderboard):
    rel_l2 94.168   tke 74.026   mvpe 92.836   coverage@[0.030,0.010] 0.841
Runs on whatever shards have downloaded so far; re-runnable as more arrive.
"""
import glob, json, os, sys
import numpy as np, torch, pyarrow as pa, pyarrow.ipc as ipc
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
SD=f"{B}/data/real_hf/foil/hf_dataset/real"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))
import importlib.util
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
from load_baseline import load_baseline
SIG=S.SIGMA_GLOBAL; DEV="cuda:3"; C=2; IN=20; HOR=40
MI=np.array([0.154960856,-0.000513992854,0.0],np.float32); SI=np.array([0.0968056545,0.015960684,1.0],np.float32)
MT=np.array([0.154962569,-0.000517793698,0.0],np.float32); ST=np.array([0.0968104079,0.0159636438,1.0],np.float32)
model=load_baseline(f"{B}/local_harness/fno_model/sim_real_fno_fp16.pth",device=DEV)[0]; model.eval()
mi,si,mt,st=[torch.tensor(a).to(DEV) for a in (MI,SI,MT,ST)]

idx=json.load(open(f"{B}/data/hf_meta/foil/hf_dataset/test_index_real.json"))
bysim={}
for e in idx: bysim.setdefault(e["sim_id"],[]).append(e["time_id"])
ind=set(json.load(open(f"{B}/data/foil/in_dist_test_params_real.json")))
outd=set(json.load(open(f"{B}/data/foil/out_dist_test_params_real.json")))
TEST20=ind|outd

@torch.no_grad()
def run(A):
    o=[]
    for i in range(0,len(A),64):
        xb=torch.from_numpy(A[i:i+64]).to(DEV).float()
        o.append((model((xb-mi)/si)*st+mt).cpu().numpy())
    return np.concatenate(o,0).astype(np.float32)

n=lambda x:x/(0.5+x)
rows=[]; allP=[]; allY=[]; allsim=[]
files=sorted(glob.glob(f"{SD}/*.arrow"))
print("shards available: %d ; official index covers %d trajectories"%(len(files),len(bysim)),flush=True)
for sp in files:
    try:
        with pa.memory_map(sp,"rb") as src:
            tb=ipc.open_stream(src).read_all(); sid=tb.column("sim_id")[0].as_py()
            t,h,w=[tb.column(k)[0].as_py() for k in ("shape_t","shape_h","shape_w")]
            if sid not in bysim: continue
            u=np.frombuffer(tb.column("u")[0].as_py(),np.float32).reshape(t,h,w)[:,::4,::4][:,:32,:64]
            v=np.frombuffer(tb.column("v")[0].as_py(),np.float32).reshape(t,h,w)[:,::4,::4][:,:32,:64]
    except Exception as ex:
        print("  skip %s (%s)"%(os.path.basename(sp),ex),flush=True); continue
    ts=[x for x in sorted(bysim[sid]) if x+HOR<=u.shape[0]]
    if not ts: continue
    xs=[];ys=[]
    for t0 in ts:
        tr=np.stack([u[t0:t0+HOR],v[t0:t0+HOR],np.zeros_like(u[t0:t0+HOR])],-1)
        xs.append(tr[:IN]); ys.append(tr[IN:])
    X=np.stack(xs).astype(np.float32); Y=np.stack(ys).astype(np.float32)
    P=run(X)
    dm=S.rel_l2_per_sample(P,Y,C); tk=S.tke_rel_l2_per_sample(P,Y,C); mv=S.mvpe_rel_l2_per_sample(P,Y)
    e=np.abs(P[...,:C]-Y[...,:C])
    cov=0.5*(float((e[...,0]<=0.030).mean())+float((e[...,1]<=0.010).mean()))
    rows.append(dict(sim=sid,n=len(X),intest20=sid in TEST20,
        dm=float(dm.mean()),tk=float(tk.mean()),mv=float(mv.mean()),cov=cov,
        rl_s=S.score_error(float(dm.mean())),tk_s=S.score_error(float(tk.mean())),mv_s=S.score_error(float(mv.mean()))))
    allP.append(P.astype(np.float16)); allY.append(Y.astype(np.float16)); allsim += [sid]*len(X)
    print("  %-16s n=%3d %s | rl %6.2f tke %6.2f mvpe %6.2f cov %.3f"%(
        sid,len(X),"TEST20" if sid in TEST20 else "      ",rows[-1]["rl_s"],rows[-1]["tk_s"],rows[-1]["mv_s"],cov),flush=True)
    del u,v,X,Y,P

def agg(sel,tag):
    if not sel: print("  %s: none"%tag); return
    N=sum(r["n"] for r in sel)
    dm=sum(r["dm"]*r["n"] for r in sel)/N; tk=sum(r["tk"]*r["n"] for r in sel)/N
    mv=sum(r["mv"]*r["n"] for r in sel)/N; cov=sum(r["cov"]*r["n"] for r in sel)/N
    print("  %-22s n=%5d | rl %6.2f  tke %6.2f  mvpe %6.2f  cov %.3f"%(
        tag,N,S.score_error(dm),S.score_error(tk),S.score_error(mv),cov),flush=True)

print("\n=== AGGREGATE on the official window index ===",flush=True)
print("  %-22s %13s %6s   %6s   %6s   %s"%("ANCHOR (real)","","94.17","74.03","92.84","0.841"),flush=True)
agg(rows,"all available")
agg([r for r in rows if r["intest20"]],"official TEST-20 only")
json.dump(rows,open(f"{B}/local_harness/official_eval_rows.json","w"),indent=1)
if allP:
    np.savez_compressed(f"{B}/local_harness/official_eval_cache.npz",
        P=np.concatenate(allP,0),Y=np.concatenate(allY,0),sim=np.array(allsim))
    print("\ncached predictions -> official_eval_cache.npz",flush=True)
