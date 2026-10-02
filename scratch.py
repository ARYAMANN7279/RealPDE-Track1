import re
with open('/SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es/build_asym.py', 'r') as f:
    text = f.read()

# Replace the network and inference part
text = text.replace("net=UNet(80,80,W)", "net=UNet(80,120,W)")

inference_old = """with torch.no_grad():
    for i in range(0,len(RS),16):
        o=net(torch.cat([flat(XI[i:i+16]),flat(PR[i:i+16])],1).to(DEV))
        Cc[i:i+16]=o[:,:40].reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy()
        Ww[i:i+16]=o[:,40:].reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy()"""

inference_new = """Ww_d=np.zeros_like(RS); Ww_u=np.zeros_like(RS)
with torch.no_grad():
    for i in range(0,len(RS),16):
        o=net(torch.cat([flat(XI[i:i+16]),flat(PR[i:i+16])],1).to(DEV))
        Cc[i:i+16]=o[:,:40].reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy()
        Ww_d[i:i+16]=o[:,40:80].reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy()
        Ww_u[i:i+16]=o[:,80:].reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy()"""

text = text.replace(inference_old, inference_new)

# Replace the fitting part
fit_old = """EFF=np.abs(RS-a.alpha*Cc).astype(np.float32); del RS,Cc
NB=a.nb; LUT=np.zeros((NB,2),np.float32); ED=np.zeros((2,NB-1),np.float32); SCALE=(a.au,a.av)
print("[fit] %d windows (%s) | NB %d | err scale u x%.4f v x%.4f"%(FITW.sum(),a.fitset,NB,*SCALE),flush=True)
for ci in range(2):
    m=np.zeros(EFF.shape[:-1],bool); m[FITW]=True; m&=SC[...,ci]
    s=Ww[...,ci][m][::a.sub].astype(np.float64); e=EFF[...,ci][m][::a.sub].astype(np.float64)*SCALE[ci]
    q=np.quantile(s,np.linspace(0,1,NB+1)[1:-1]); ED[ci]=q
    b=np.digitize(s,q); o=np.argsort(b,kind="stable"); bs=b[o]; es=e[o]
    cut=np.searchsorted(bs,np.arange(NB+1))
    for k in range(NB):
        v=np.sort(es[cut[k]:cut[k+1]])
        if v.size<200: LUT[k,ci]=0.012; continue
        kk=np.arange(1,v.size+1)/v.size
        LUT[k,ci]=float(v[np.argmax(np.exp(-2*v/SIG)*kk)])
    print("   ch%d %d elem | LUT min %.5f med %.5f max %.5f | %d/%d non-monotone"
          %(ci,s.size,LUT[:,ci].min(),np.median(LUT[:,ci]),LUT[:,ci].max(),
            int((np.diff(LUT[:,ci])<0).sum()),NB-1),flush=True)
assert np.isfinite(LUT).all() and np.isfinite(ED).all() and (LUT>0).all() and LUT.max()<0.10
sd={k:v.cpu().numpy().astype(np.float32) for k,v in net.state_dict().items()}
OUT_NPZ=f"{B}/train_es/bounds_assets_{a.tag}.npz"
np.savez_compressed(OUT_NPZ,LUT=LUT,ED=ED,w=np.int32(W),alpha=np.float32(a.alpha),
                    mu=np.float32(ck["mu"]),sd_=np.float32(ck["sd_"]),
                    **{"w_"+k:v for k,v in sd.items()})"""

fit_new = """EFF_D=np.maximum(-(RS-a.alpha*Cc), 0).astype(np.float32)
EFF_U=np.maximum(RS-a.alpha*Cc, 0).astype(np.float32)
del RS,Cc
NB=a.nb; LUT_D=np.zeros((NB,2),np.float32); ED_D=np.zeros((2,NB-1),np.float32); SCALE=(a.au,a.av)
LUT_U=np.zeros((NB,2),np.float32); ED_U=np.zeros((2,NB-1),np.float32)
print("[fit] %d windows (%s) | NB %d | err scale u x%.4f v x%.4f"%(FITW.sum(),a.fitset,NB,*SCALE),flush=True)

def fit_lut(Ww, EFF):
    LUT=np.zeros((NB,2),np.float32); ED=np.zeros((2,NB-1),np.float32)
    for ci in range(2):
        m=np.zeros(EFF.shape[:-1],bool); m[FITW]=True; m&=SC[...,ci]
        s=Ww[...,ci][m][::a.sub].astype(np.float64); e=EFF[...,ci][m][::a.sub].astype(np.float64)*SCALE[ci]
        q=np.quantile(s,np.linspace(0,1,NB+1)[1:-1]); ED[ci]=q
        b=np.digitize(s,q); o=np.argsort(b,kind="stable"); bs=b[o]; es=e[o]
        cut=np.searchsorted(bs,np.arange(NB+1))
        for k in range(NB):
            v=np.sort(es[cut[k]:cut[k+1]])
            if v.size<200: LUT[k,ci]=0.012; continue
            kk=np.arange(1,v.size+1)/v.size
            LUT[k,ci]=float(v[np.argmax(np.exp(-2*v/SIG)*kk)])
        print("   ch%d %d elem | LUT min %.5f med %.5f max %.5f | %d/%d non-monotone"
              %(ci,s.size,LUT[:,ci].min(),np.median(LUT[:,ci]),LUT[:,ci].max(),
                int((np.diff(LUT[:,ci])<0).sum()),NB-1),flush=True)
    return LUT, ED

print("Fitting LUT_D:")
LUT_D, ED_D = fit_lut(Ww_d, EFF_D)
print("Fitting LUT_U:")
LUT_U, ED_U = fit_lut(Ww_u, EFF_U)

assert np.isfinite(LUT_D).all() and np.isfinite(ED_D).all() and (LUT_D>=0).all() and LUT_D.max()<0.10
assert np.isfinite(LUT_U).all() and np.isfinite(ED_U).all() and (LUT_U>=0).all() and LUT_U.max()<0.10
sd={k:v.cpu().numpy().astype(np.float32) for k,v in net.state_dict().items()}
OUT_NPZ=f"{B}/train_es/bounds_assets_{a.tag}.npz"
np.savez_compressed(OUT_NPZ,LUT_D=LUT_D,ED_D=ED_D,LUT_U=LUT_U,ED_U=ED_U,w=np.int32(W),alpha=np.float32(a.alpha),
                    mu=np.float32(ck["mu"]),sd_=np.float32(ck["sd_"]),
                    **{"w_"+k:v for k,v in sd.items()})"""

text = text.replace(fit_old, fit_new)
text = text.replace('open(f"{B}/train_es/submission_shift.py","rb").read()', 'open(f"{B}/train_es/submission_asym.py","rb").read()')
text = text.replace('"lut_med":[float(np.median(LUT[:,0])),float(np.median(LUT[:,1]))]', '"lut_med_d":[float(np.median(LUT_D[:,0])),float(np.median(LUT_D[:,1]))]')

with open('/SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es/build_asym.py', 'w') as f:
    f.write(text)
