import numpy as np, json, sys, time
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; LH=f"{B}/local_harness"
sys.path.insert(0,B)
X=np.load(f"{LH}/tr_frames.npy")
m=json.load(open(f"{LH}/tr_meta.json")); off,lens,names=m["off"],m["lens"],m["names"]
RE=np.array([int(n.split("_")[0]) for n in names]); AOA=np.array([int(n.split("_")[1].split(".")[0]) for n in names])
traj_of=np.zeros(len(X),int)
for i in range(len(names)): traj_of[off[i]:off[i]+lens[i]]=i

def feats(starts):
    F=[]
    for c in range(0,len(starts),256):
        w=np.stack([X[s:s+20] for s in starts[c:c+256]]).astype(np.float32)
        u=w[...,0]; v=w[...,1]
        nz=((u!=0)|(v!=0)); valid=nz.mean(1)>0.5
        cnt=valid.sum((1,2)).clip(1)
        um=u.mean(1); vm=v.mean(1)
        s=(um*valid).sum((1,2))/cnt
        vb=(vm*valid).sum((1,2))/cnt
        su=u.std(1); sv=v.std(1)
        ti=np.sqrt(((su**2+sv**2)*valid).sum((1,2))/cnt)
        zf=1-valid.mean((1,2))
        vf=v[:,:,:,40:]-v[:,:,:,40:].mean(1,keepdims=True)
        P=(np.abs(np.fft.rfft(vf,axis=1))**2).mean((2,3))[:,1:]
        Pn=P/P.sum(1,keepdims=True).clip(1e-20)
        fc=(Pn*np.arange(1,11)).sum(1)
        pool=lambda a: a.reshape(a.shape[0],8,4,16,4).mean((2,4)).reshape(a.shape[0],-1)
        sc=s[:,None,None]
        f=np.concatenate([np.stack([np.log(s),zf,vb/s,ti/s,fc],1),Pn,
                          pool(um/sc),pool(vm/sc),pool(np.sqrt(su**2+sv**2)/sc),pool(valid.astype(np.float32))],1)
        F.append(f)
    return np.concatenate(F,0)

def lda_fit(F,y,classes,lam=0.1):
    mu=F.mean(0); sd=F.std(0)+1e-8; Z=(F-mu)/sd
    M=np.stack([Z[y==c].mean(0) for c in classes])
    R_=Z-M[np.searchsorted(classes,y)]
    C=R_.T@R_/len(Z); C=(1-lam)*C+lam*np.eye(C.shape[0])*np.trace(C)/C.shape[0]
    Ci=np.linalg.inv(C); W=Ci@M.T; b=-0.5*np.sum(M@Ci*M,1)
    return dict(mu=mu,sd=sd,W=W,b=b,classes=np.array(classes))

allst=np.array([t0 for i in range(len(names)) for t0 in range(off[i],off[i]+lens[i]-39)])
trs=allst[::5]
Ftr=feats(trs); ytr_t=traj_of[trs]
ytr=AOA[ytr_t]

classes=[0,5,10,15,20]
P=lda_fit(Ftr,ytr,classes,lam=0.1)
np.savez(f"{B}/agents/moe/router_lda.npz", **P)
print("Baked router lda.npz")
