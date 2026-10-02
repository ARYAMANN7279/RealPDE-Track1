import torch
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP/local_harness"
sd=torch.load(f"{B}/soup_v2.pth",map_location="cpu")
sd=sd.get("model_state_dict",sd)
cplx=[k for k in sd if torch.is_tensor(sd[k]) and sd[k].is_complex()]
print("keys:",len(sd),"complex tensors:",len(cplx))
bad=sum(1 for k in cplx if sd[k].imag.abs().max().item()==0.0)
print("complex tensors with ZERO imaginary part:",bad)
if cplx:
    print("sample imag magnitudes:",[round(sd[k].imag.abs().max().item(),6) for k in cplx[:5]])
print("any NaN/Inf:", any(torch.is_tensor(v) and not torch.isfinite(v.float() if v.is_complex() else v).all() for v in sd.values() if torch.is_tensor(v)))

members=['ft_all_w15_lr1_final.pth','ft_all_w15_lr3_final.pth','ft_all_w33_lr1_final.pth','ft_all_w33_lr3_final.pth']
mem={p:torch.load(f"{B}/{p}",map_location="cpu") for p in members}
mem={p:v.get("model_state_dict",v) for p,v in mem.items()}
avg={}
for k in mem[members[0]]:
    avg[k]=sum(mem[p][k] for p in members)/len(members)
maxdiff=max(float((sd[k]-avg[k]).abs().max()) for k in avg if torch.is_tensor(avg[k]))
print("max|soup_v2.pth - recomputed avg(4 members)| =",maxdiff)

fp16=torch.load(f"{B}/soup_v2_fp16.pth",map_location="cpu")
fp16=fp16.get("model_state_dict",fp16)
cplx16=[k for k in fp16 if torch.is_tensor(fp16[k]) and fp16[k].is_complex()]
print("\nsoup_v2_fp16.pth: complex tensors",len(cplx16),
      "zero-imag:",sum(1 for k in cplx16 if fp16[k].imag.abs().max().item()==0.0))
print("dtypes present:",sorted(set(str(v.dtype) for v in fp16.values() if torch.is_tensor(v))))
