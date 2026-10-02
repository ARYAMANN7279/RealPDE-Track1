"""Verify soup_best.pth integrity: complex tensors intact, and it IS the all-6 average."""
import torch, os, numpy as np
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
LH=f"{B}/local_harness"

sd=torch.load(f"{LH}/soup_best.pth",map_location="cpu")
sd=sd.get("model_state_dict",sd)
print("=== soup_best.pth ===")
print("keys:",len(sd))
cplx=[k for k in sd if torch.is_tensor(sd[k]) and sd[k].is_complex()]
print("complex tensors:",len(cplx))
bad=0
for k in cplx:
    im=sd[k].imag.abs().max().item()
    if im==0.0:
        bad+=1; print("  !! ZERO IMAG:",k)
print("complex tensors with zero imaginary part:",bad)
if cplx:
    print("  sample imag magnitudes:",[round(sd[k].imag.abs().max().item(),6) for k in cplx[:5]])

# Is it the average of all 6, or of w010+w015 (the buggy run's pick)?
tags=["w005","w010","w015","lr1e5","w060","lr3e5"]
mem={t:torch.load(f"{LH}/ft_{t}_best.pth",map_location="cpu") for t in tags
     if os.path.exists(f"{LH}/ft_{t}_best.pth")}
mem={t:v.get("model_state_dict",v) for t,v in mem.items()}
print("\nmembers found:",list(mem.keys()))

def maxdiff(a,b):
    m=0.0
    for k in a:
        if not (torch.is_tensor(a[k]) and (a[k].is_complex() or torch.is_floating_point(a[k]))): continue
        d=(a[k]-b[k]).abs().max().item()
        m=max(m,d)
    return m

import itertools
for name,keys in [("all6",list(mem.keys())),("w010+w015",["w010","w015"]),
                  ("w005+w010+w015",["w005","w010","w015"])]:
    ks=[k for k in keys if k in mem]
    if len(ks)<2: continue
    avg={}
    for k in mem[ks[0]]:
        v0=mem[ks[0]][k]
        if torch.is_tensor(v0) and (v0.is_complex() or torch.is_floating_point(v0)):
            avg[k]=(sum(mem[t][k] for t in ks)/len(ks)).to(v0.dtype)
        else: avg[k]=v0
    print("  vs %-16s max|diff| = %.3e"%(name,maxdiff(sd,avg)))
