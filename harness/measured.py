exec(open("disagreement.py").read().split("=== signal quality")[0].rsplit("print(",1)[0])
soup_sd=torch.load(f"{LH}/soup_final_candidate.pth",map_location=DEV)
P0=P_soup[...,:C]
def quality(sig,tag):
    m=SCM[ev]; a=np.log(sig[ev][m]+1e-8); b=np.log(ERR[ev][m]+1e-8)
    c=float(np.corrcoef(a,b)[0,1]); s=float(a.std())
    print("  %-34s corr %.3f  sd %.3f  %s"%(tag,c,s,"<== CLEARS BAR" if (c>0.48 and s>1.2) else ""))
    return c,s
def run_sd(sd):
    base.load_state_dict(sd); m=base.to(DEV).eval(); o=[]
    with torch.no_grad():
        for i in range(0,len(Xin),32):
            xb=torch.from_numpy(np.ascontiguousarray(Xin[i:i+32])).to(DEV)
            o.append((m((xb-mi)/si)*st+mt).cpu().numpy())
    return np.concatenate(o,0).astype(np.float32)[...,:C]
print("=== MEASURED signals that could fit in 256MB (bar: corr>0.48 AND sd>1.2) ===")
print("-- A: |soup - ONE member| (needs 1 extra ckpt; 201+50MB int8 delta = 251MB, tight)")
for k in (0,4,8):
    quality(np.abs(P0-PS[k])+1e-8,"|soup - %s|"%MEM[k].replace("ft_","").replace("_best.pth",""))
print("-- B: weight-noise ensemble (ZERO extra storage, K forwards)")
for sd_n in (0.01,0.02,0.05):
    outs=[]
    for r in range(3):
        g=torch.Generator(device="cpu").manual_seed(100+r)
        pert={}
        for k,v in soup_sd.items():
            if torch.is_tensor(v) and (v.is_complex() or torch.is_floating_point(v)):
                if v.is_complex():
                    n=(torch.randn(v.shape,generator=g,dtype=torch.float32)+1j*torch.randn(v.shape,generator=g,dtype=torch.float32)).to(v.dtype).to(v.device)
                else:
                    n=torch.randn(v.shape,generator=g,dtype=torch.float32).to(v.dtype).to(v.device)
                pert[k]=v+n*v.abs().mean()*sd_n
            else: pert[k]=v
        outs.append(run_sd(pert))
    quality(np.stack(outs).std(axis=0)+1e-8,"weight-noise sd=%.2f (K=3)"%sd_n)
print()
print("reference: ensemble K=3 corr 0.558 sd 1.277 (BEST, but needs 600MB)")
print("           learned head corr 0.605 sd 0.811 | input-pert best corr 0.420")
