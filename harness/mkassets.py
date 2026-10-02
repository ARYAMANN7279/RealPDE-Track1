"""Export everything the submission needs: temporal spectral gain, head weights,
feature normalisation, and the mu->h lookup table."""
import json,os,sys,numpy as np,torch
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT)
import importlib.util
sp=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=importlib.util.module_from_spec(sp); sp.loader.exec_module(S); SIG=S.SIGMA_GLOBAL; C=2
d=np.load(f"{B}/local_harness/comp_eval_cache.npz")
P=d["P"].astype(np.float32); Y=d["Y"].astype(np.float32)
rows=json.load(open(f"{B}/local_harness/comp_anchor_rows.json"))
sim=np.concatenate([[r["sim"]]*r["n"] for r in rows]); SCM=(Y[...,:C]!=0.0)
# spectral gain fitted on ALL trajectories for deployment
Fp=np.fft.rfft(P[...,:C],axis=1); Fy=np.fft.rfft(Y[...,:C],axis=1)
gt=np.sqrt((np.abs(Fy)**2).mean(axis=(0,2,3))/np.maximum((np.abs(Fp)**2).mean(axis=(0,2,3)),1e-20))
gt=np.clip(gt,0.5,2.0).astype(np.float32)
print("temporal gain shape",gt.shape,"u:", " ".join("%.2f"%x for x in gt[:6,0]))
Fq=np.fft.rfft(P[...,:C],axis=1)*gt[None,:,None,None,:]
P1=P.copy(); P1[...,:C]=np.fft.irfft(Fq,n=P.shape[1],axis=1)
# calibrate + rebuild LUT on the CORRECTED predictions, all trajectories
EL=np.abs(P1[...,:C]-Y[...,:C]); AP=np.abs(P1[...,:C])
m0,m1=SCM[...,0],SCM[...,1]; SB=40
eu=EL[...,0][m0][::SB]; ev=EL[...,1][m1][::SB]; au=AP[...,0][m0][::SB]; av=AP[...,1][m1][::SB]
NT=eu.size+ev.size; best=None
for al in np.linspace(1.0,6.0,26):
    for be in np.linspace(-12,2,29):
        su,sv=eu*(al+be*au),ev*(al+be*av)
        Ec=(np.exp(-2*0.030/SIG)*(su<=0.030).sum()+np.exp(-2*0.010/SIG)*(sv<=0.010).sum())/NT
        hu,hv=0.05*au,0.05*av
        Ep=((np.exp(-2*hu/SIG)*(su<=hu)).sum()+(np.exp(-2*hv/SIG)*(sv<=hv)).sum())/NT
        r=abs(Ec-0.4399)+abs(Ep-0.2076)
        if best is None or r<best[0]: best=(r,al,be)
_,AL,BE=best; ERR=(EL*(AL+BE*AP)).astype(np.float32)
print("calibration alpha %.2f beta %.2f"%(AL,BE))
exec(open("stack.py").read().split("def feats")[1].split("class Net")[0].join(["def feats",""]))
import torch.nn as nn, torch.nn.functional as F
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
ck=torch.load(f"{B}/local_harness/unc3_best.pth",map_location="cuda:2",weights_only=False)
NF=ck["nf"]; net=Net(NF).to("cuda:2"); net.load_state_dict(ck["sd"]); net.eval()
FT=(feats(P1)-ck["mu"])/ck["std"]
MU=[]
with torch.no_grad():
    for i in range(0,len(P1),8):
        f=torch.from_numpy(np.ascontiguousarray(FT[i:i+8])).to("cuda:2").permute(0,1,4,2,3).reshape(-1,NF,32,64)
        MU.append(net(f).reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy())
MU=np.concatenate(MU,0)
def best_h(v):
    v=np.sort(v)
    if v.size==0: return 0.012
    k=np.arange(1,v.size+1)/v.size
    return float(v[np.argmax(np.exp(-2*v/SIG)*k)])
NB=24; LUT=np.zeros((NB,C),np.float32); ED=np.zeros((C,NB-1),np.float32)
for ci in range(C):
    m=SCM[...,ci]; mu=MU[...,ci][m]; er=ERR[...,ci][m]
    q=np.quantile(mu,np.linspace(0,1,NB+1)[1:-1]); ED[ci]=q
    b=np.digitize(mu,q)
    for k in range(NB):
        s_=b==k; LUT[k,ci]=best_h(er[s_][::3]) if s_.sum()>200 else 0.012
print("LUT h_u range %.4f .. %.4f"%(LUT[:,0].min(),LUT[:,0].max()))
sd={k:v.cpu().numpy() for k,v in net.state_dict().items()}
np.savez_compressed(f"{B}/local_harness/head_assets.npz",gt=gt,LUT=LUT,ED=ED,
                    fmu=ck["mu"],fsd=ck["std"],nf=np.int32(NF),**{"w_"+k:v for k,v in sd.items()})
print("saved head_assets.npz  %.2f MB"%(os.path.getsize(f"{B}/local_harness/head_assets.npz")/1e6))
