exec(open("disagreement.py").read().split("=== signal quality")[0].rsplit("print(",1)[0])
# SINGLE-CHECKPOINT measured uncertainty: perturb the input, measure output swing.
# Ensemble disagreement is impossible (3x201MB >> 256MB cap); this needs one ckpt.
rng=np.random.default_rng(0)
soup_sd=torch.load(f"{LH}/soup_final_candidate.pth",map_location=DEV)
def run_pert(noise=0.0,shift=0,scale=1.0):
    base.load_state_dict(soup_sd); m=base.to(DEV).eval(); o=[]
    with torch.no_grad():
        for i in range(0,len(Xin),32):
            x=Xin[i:i+32].copy()
            if noise>0: x[...,:2]+=rng.normal(0,noise,x[...,:2].shape).astype(np.float32)
            if shift: x=np.roll(x,shift,axis=3)
            if scale!=1.0: x[...,:2]*=scale
            xb=torch.from_numpy(np.ascontiguousarray(x)).to(DEV)
            y=(m((xb-mi)/si)*st+mt).cpu().numpy()
            if shift: y=np.roll(y,-shift,axis=3)
            if scale!=1.0: y[...,:2]/=scale
            o.append(y)
    return np.concatenate(o,0).astype(np.float32)[...,:C]

def quality(sig,tag):
    m=SCM[ev]
    a=np.log(sig[ev][m]+1e-8); b=np.log(ERR[ev][m]+1e-8)
    print("  %-30s corr %.3f   sd(log s) %.3f"%(tag,float(np.corrcoef(a,b)[0,1]),float(a.std())))
    return sig

P0=P_soup[...,:C]
print("=== single-checkpoint measured signals (bar: corr>0.48, sd>~1.2) ===")
cands={}
for nz in (0.002,0.005,0.01):
    s=np.abs(run_pert(noise=nz)-P0)+1e-8; cands["noise %.3f"%nz]=s; quality(s,"input noise %.3f"%nz)
s=np.abs(run_pert(shift=1)-P0)+1e-8; cands["shift1"]=s; quality(s,"spatial shift 1px")
s=np.abs(run_pert(scale=1.02)-P0)+1e-8; cands["scale1.02"]=s; quality(s,"input scale x1.02")
# average of two noise draws = lower-variance estimate
s=(cands["noise 0.005"]+np.abs(run_pert(noise=0.005)-P0))/2+1e-8; cands["noise2avg"]=s; quality(s,"noise 0.005 x2 avg")
print()
print("for reference: learned head corr 0.605 sd 0.811 | ensemble K=3 corr 0.558 sd 1.277")
np.savez_compressed(f"{LH}/sens_signals.npz",**{k:v.astype(np.float16) for k,v in cands.items()})
print("saved sens_signals.npz")
