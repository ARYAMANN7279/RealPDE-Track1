"""FITBASE CPU weight comparison (no GPU).  python fb_wcmp.py A.pth B.pth [C.pth]
Prints max|A-B| and ||A-B||/||B|| (complex as real pairs, BN counters skipped); if C is given, also
||C-B||/||B|| as a yardstick (e.g. how far a DIFFERENT-lr member sits from B)."""
import sys, math, torch
def load_sd(p):   # r50_build.py verbatim
    sd=torch.load(p,map_location="cpu")
    if isinstance(sd,dict) and "model_state_dict" in sd: sd=sd["model_state_dict"]
    if isinstance(sd,dict) and "state_fp16" in sd:
        ck=set(sd.get("complex_keys",[])); out={}
        for k,v in sd["state_fp16"].items():
            v=v.float()
            if k in ck: v=torch.view_as_complex(v)
            out[k]=v
        return out
    return sd
def cmp(a,b):
    mx=0.0; num=den=0.0
    for k,y in b.items():
        if "num_batches_tracked" in k or not (y.is_complex() or y.dtype.is_floating_point): continue
        x=a[k]
        if y.is_complex(): x=torch.view_as_real(x.to(torch.complex64)); y=torch.view_as_real(y.to(torch.complex64))
        x=x.double(); y=y.double(); dd=x-y
        mx=max(mx,float(dd.abs().max())); num+=float((dd*dd).sum()); den+=float((y*y).sum())
    return mx, math.sqrt(num/max(den,1e-300))
A=load_sd(sys.argv[1]); Bsd=load_sd(sys.argv[2])
print("keys identical: %s (%d)"%(set(A)==set(Bsd),len(Bsd)))
mx,rd=cmp(A,Bsd); print("A vs B : max|d| %.4e   rel-L2 %.4e   [%s vs %s]"%(mx,rd,sys.argv[1].split("/")[-1],sys.argv[2].split("/")[-1]))
if len(sys.argv)>3:
    Csd=load_sd(sys.argv[3]); mx2,rd2=cmp(Csd,Bsd)
    print("C vs B : max|d| %.4e   rel-L2 %.4e   [yardstick %s]"%(mx2,rd2,sys.argv[3].split("/")[-1]))
print("[done]")
