"""Head input is the TIME-MEAN of per-frame features. Does averaging over a
subsample of frames (stride 4 -> 5 frames) give the same mu, and hence the same E?
If so the feature pass gets ~4x cheaper and we recover part of the time_score loss."""
import json,os,sys,numpy as np,torch,torch.nn as nn
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT)
import importlib.util as iu
sp=iu.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=iu.module_from_spec(sp); sp.loader.exec_module(S); SIG=S.SIGMA_GLOBAL; C=2; DEV="cuda"
d=np.load(f"{B}/local_harness/comp_eval_cache.npz")
P0=d["P"].astype(np.float32); Y=d["Y"].astype(np.float32)
rows=json.load(open(f"{B}/local_harness/comp_anchor_rows.json"))
sim=np.concatenate([[r["sim"]]*r["n"] for r in rows]); sims=sorted(set(sim.tolist()))
tr=np.isin(sim,sims[0::2]); te=~tr; SCM=(Y[...,:C]!=0.0)
z=np.load(f"{B}/local_harness/head_assets.npz")
gt=z["gt"]; Fq=np.fft.rfft(P0[...,:C],axis=1)*gt[None,:,None,None,:]
P=P0.copy(); P[...,:C]=np.fft.irfft(Fq,n=20,axis=1)
src=open("stack.py").read(); exec(src[src.index("def feats"):src.index("class Net")])
class Net(nn.Module):
    def __init__(s,nf,w=128):
        super().__init__()
        s.n=nn.Sequential(nn.Conv2d(nf,w,3,padding=1),nn.GELU(),
            nn.Conv2d(w,w,3,padding=2,dilation=2),nn.GELU(),
            nn.Conv2d(w,w,3,padding=4,dilation=4),nn.GELU(),
            nn.Conv2d(w,w,3,padding=8,dilation=8),nn.GELU(),
            nn.Conv2d(w,w,3,padding=1),nn.GELU(),
            nn.Conv2d(w,2,1))
    def forward(s,x): return s.n(x)
nf=int(z["nf"]); net=Net(nf).to(DEV)
net.load_state_dict({k[2:]:torch.from_numpy(z["w_"+k[2:]]) for k in z.files if k.startswith("w_")}); net.eval()
FT=(feats(P)-z["fmu"])/z["fsd"]
def mu_sub(mask,stride):
    o=[];idx=np.where(mask)[0]
    with torch.no_grad():
        for i in range(0,len(idx),8):
            j=idx[i:i+8]
            f=torch.from_numpy(np.ascontiguousarray(FT[j][:,::stride].mean(axis=1))).to(DEV).permute(0,3,1,2)
            m=net(f).permute(0,2,3,1).cpu().numpy(); o.append(np.repeat(m[:,None],20,axis=1))
    return np.concatenate(o,0)
EL=np.abs(P[...,:C]-Y[...,:C]); AP=np.abs(P[...,:C]); ERR=(EL*(2.80-8.00*AP)).astype(np.float32)
def best_h(v):
    v=np.sort(v)
    if v.size==0: return 0.012
    k=np.arange(1,v.size+1)/v.size
    return float(v[np.argmax(np.exp(-2*v/SIG)*k)])
def evalE(MUtr,MUte):
    NB=24; LUT=np.zeros((NB,C),np.float32); ED=[]
    for ci in range(C):
        m=SCM[tr][...,ci]; mu=MUtr[...,ci][m]; er=ERR[tr][...,ci][m]
        q=np.quantile(mu,np.linspace(0,1,NB+1)[1:-1]); ED.append(q); b=np.digitize(mu,q)
        for k in range(NB):
            s_=b==k; LUT[k,ci]=best_h(er[s_][::3]) if s_.sum()>200 else 0.012
    tot=0.0;cnt=0.0
    for ci in range(C):
        m=SCM[te][...,ci]; mu=MUte[...,ci][m]; er=ERR[te][...,ci][m]
        h=LUT[np.digitize(mu,ED[ci]),ci]
        tot+=float((np.exp(-2*h/SIG)*(er<=h)).sum()); cnt+=float(m.sum())
    return tot/cnt
for st in (1,2,4,5):
    E=evalE(mu_sub(tr,st),mu_sub(te,st))
    print("  frame stride %d (%2d frames): E %.4f"%(st,len(range(0,20,st)),E))
