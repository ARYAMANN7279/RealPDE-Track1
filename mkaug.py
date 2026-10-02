p = "/SML_DISK_24TB/rajeshr/Aryamann/UGP/ffno_train.py"
s = open(p).read()
if "--noise" in s:
    print("already patched"); raise SystemExit
s = s.replace('ap.add_argument("--tmodes",type=int,default=20)',
              'ap.add_argument("--noise",type=float,default=0.0)\n'
              'ap.add_argument("--maskp",type=float,default=0.0)\n'
              'ap.add_argument("--tmodes",type=int,default=20)')
old = "        x,y=batch(Xs,perm[i:i+a.bs]); p=fwd(x)"
new = """        x,y=batch(Xs,perm[i:i+a.bs])
        if a.noise>0:                                  # official: x += x*randn*noise_scale
            x=x+x*torch.randn_like(x)*a.noise; y=y+y*torch.randn_like(y)*a.noise
        if a.maskp>0 and np.random.rand()<a.maskp:     # official: mask unmeasured modality
            x=x.clone(); x[...,2]=0.0
        p=fwd(x)"""
assert old in s, "sim-phase anchor not found"
s = s.replace(old, new, 1)
open(p, "w").write(s)
print("patched --noise / --maskp into the SIM phase only")
