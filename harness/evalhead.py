import json,os,sys,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT)
import importlib.util
sp=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=importlib.util.module_from_spec(sp); sp.loader.exec_module(S)
SIG=S.SIGMA_GLOBAL; C=2; DEV="cuda:2"; W=0.6784
exec(open("unc_head2.py").read().split("class Head")[0].split("print(\"features")[0].replace("print(","#print("))
class Head(nn.Module):
    def __init__(s,nf,w=112):
        super().__init__()
        s.net=nn.Sequential(nn.Conv2d(nf,w,3,padding=1),nn.GELU(),
            nn.Conv2d(w,w,3,padding=1,dilation=1),nn.GELU(),
            nn.Conv2d(w,w,3,padding=2,dilation=2),nn.GELU(),
            nn.Conv2d(w,w,3,padding=4,dilation=4),nn.GELU(),
            nn.Conv2d(w,w,3,padding=1),nn.GELU(),
            nn.Conv2d(w,2,1))
        s.b=nn.Parameter(torch.zeros(2))
    def forward(s,x): return F.softplus(s.net(x)+s.b.view(1,2,1,1))+1e-5
ck=torch.load(f"{B}/local_harness/unc_head_best.pth",map_location=DEV,weights_only=False)
NF=ck["nf"]; net=Head(NF).to(DEV); net.load_state_dict(ck["sd"]); net.eval()
FT2=(feats(P)-ck["mu"])/ck["std"]
def to_t(a): return torch.from_numpy(np.ascontiguousarray(a)).to(DEV)
tot=0.0;cnt=0.0;hs=[]
idx=np.where(te)[0]
with torch.no_grad():
    for i in range(0,len(idx),8):
        j=idx[i:i+8]
        f=to_t(FT2[j]).permute(0,1,4,2,3).reshape(-1,NF,32,64)
        h=net(f).reshape(len(j),20,2,32,64).permute(0,1,3,4,2)
        e=to_t(ERR[j]); m=to_t(SCM[j].astype(np.float32))
        ok=(e<=h).float()*m
        tot+=float((torch.exp(-2*h/SIG)*ok).sum()); cnt+=float(m.sum())
        if i==0: hs=h.cpu().numpy()
E=tot/cnt
fin=lambda s_:0.306*94.168150+0.163*74.025866+0.218*92.836278+0.100*91.362375+0.217*s_
RATIO=0.4876/0.5132
print("E_test(model) %.4f -> real E ~%.4f -> sps ~%.2f -> final ~%.2f  (%+.2f vs 78.07)"%(
    E,E*RATIO,100*W*E*RATIO,fin(100*W*E*RATIO),fin(100*W*E*RATIO)-78.068714))
print("half-width u med %.5f v med %.5f | sd(log h_u) %.2f  (needed >~1.2)"%(
    float(np.median(hs[...,0])),float(np.median(hs[...,1])),float(np.std(np.log(hs[...,0]+1e-9)))))
print("top-20 real E is 0.5896; ours(real,est) %.4f"%(E*RATIO))
