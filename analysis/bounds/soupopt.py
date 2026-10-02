"""Same argmax check for the SOUP calibration, plus the corrected-LUT candidate."""
import numpy as np
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; SIG = 0.0563870259; SUBS = 4
z = np.load(f"{B}/train_mvpe/runs/arrays2.npz")
A = np.load(f"{B}/_tmp/soupv1/head_assets.npz"); LUT = A["LUT"]; ED = A["ED"]; NB = LUT.shape[0]
srt, bins = {}, {}
for ci, nm in ((0,"u"),(1,"v")):
    h = z[f"held_{nm}"]
    err = z[f"err_{nm}"][h][::SUBS].astype(np.float64)
    for mk in ("mut","muf"):
        mu = z[f"{mk}_{nm}"][h][::SUBS]
        b = np.digitize(mu, ED[ci])
        bins[(nm,mk)] = [np.sort(err[b==k]) for k in range(NB)]
    srt[nm] = np.sort(err)
nu, nv = srt["u"].size, srt["v"].size; N = nu+nv
def Ech(nm,h,a):
    return np.exp(-2*h/SIG)*np.searchsorted(srt[nm], h/a, side="right")/srt[nm].size
def opt_h(nm,a):
    g = np.linspace(0.0005,0.06,2400); v=[Ech(nm,h,a) for h in g]
    i=int(np.argmax(v)); return float(g[i])
print("SOUP calibration -- where does each say the optimal CONSTANT pair is?")
print("  (live-board best known, sec5B, on the original ckpt: [0.0129, 0.0098])")
for tag, au, av in (("global_scale (shipped)", 2.3199, 1.1982),
                    ("refitted on 4 anchors ", 2.1500, 2.4250)):
    print("   %-24s [%.4f, %.4f]" % (tag, opt_h("u",au), opt_h("v",av)))

print("\ncorrected-LUT candidate (refit calibration a_u 2.150 / a_v 2.425):")
au, av = 2.1500, 2.4250
def E_lut(lut, mk):
    t=0.0
    for nm,ci,a in (("u",0,au),("v",1,av)):
        for k,s in enumerate(bins[(nm,mk)]):
            if s.size==0: continue
            h=float(lut[k,ci]); t += s.size*np.exp(-2*h/SIG)*np.searchsorted(s,h/a,side="right")/s.size
    return t/N
def refit(mk):
    nl=LUT.copy()
    for nm,ci,a in (("u",0,au),("v",1,av)):
        for k,s in enumerate(bins[(nm,mk)]):
            if s.size<200: continue
            v=a*s; kk=np.arange(1,v.size+1)/v.size
            nl[k,ci]=float(v[int(np.argmax(np.exp(-2*v/SIG)*kk))])
    return nl
W=0.684593
base = E_lut(LUT,"mut")
for mk,label in (("mut","time-avg (shipped path)"),("muf","per-frame")):
    nl = refit(mk); e = E_lut(nl,mk)
    dE = e-base; print("   %-24s E %.4f   dE vs shipped %+.4f  -> dfinal %+.4f -> %.4f"
                       % (label, e, dE, 100*W*dE*0.217, 78.4566+100*W*dE*0.217))
    np.save(f"{B}/train_mvpe/runs/LUT_corrected_{mk}.npy", nl)
    print("        h_u med %.4f (was %.4f) | h_v med %.4f (was %.4f)"
          % (np.median(nl[:,0]), np.median(LUT[:,0]), np.median(nl[:,1]), np.median(LUT[:,1])))
