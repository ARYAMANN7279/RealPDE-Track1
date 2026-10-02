import time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
DEV="cuda:2"
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
ck=torch.load("/SML_DISK_24TB/rajeshr/Aryamann/UGP/local_harness/unc3_best.pth",map_location="cpu",weights_only=False)
net=Net(ck["nf"])
np_=sum(p.numel() for p in net.parameters())
print("head params %.2fM -> %.1f MB fp32"%(np_/1e6,np_*4/1e6))
for dev in ["cuda:2","cpu"]:
    n=net.to(dev).eval()
    x=torch.randn(160,ck["nf"],32,64).to(dev)   # 8 windows x 20 frames
    with torch.no_grad():
        n(x[:16])
        t=time.time(); n(x); dt=time.time()-t
    print("  %-6s head: %.1f ms per 8 windows = %.2f ms/window"%(dev,dt*1000,dt*1000/8))
