
# =====================================================================================================
# FITBASE screen / build  (agents/fitbase, 11 Sep 2026)
# Lines 1-197 above are r50_build.py lines 1-197 VERBATIM (= r45_train.py's data / model / evaluate()).
# Invoke exactly like run_r50.sh did (the trainer args only select the 900 re_lohi monitor windows):
#   FB_MODE=screen CUDA_VISIBLE_DEVICES=0 $P fb_build.py --tag fb --split re_lohi --aug none --steps 1 --evalevery 1
#   FB_MODE=build FB_ARM=<arm label> FB_ZIPTAG=<tag>  <same args>  -> submissions/submission_FITBASE_<tag>.zip
# Screening rule (brief): local-negative => dead; advance only if 0 < dfinal < 0.20; FLAG leakage-signature
# if 9.5146*dcos0 >= dfinal; cos0 ingredient ceiling 0.8734 (SHARED_CONTEXT sec4.5).
# Evaluations are cached in screen_cache.json keyed on leaf (name, size, mtime).  BANKED is re-evaluated on
# every run and the whole cache is discarded if it does not reproduce to 1e-6.
# =====================================================================================================
import zipfile, hashlib, math, shutil, time as _time
LH=f"{B}/local_harness"; FB=f"{B}/agents/fitbase"; CK=f"{FB}/ckpt"
MODE=os.environ.get("FB_MODE","screen")
PRICE=dict(rel=0.6133,tke=0.1628,mvpe=0.1700)   # effective final per LOCAL point (SHARED_CONTEXT sec4.8)
LEAKSLOPE=9.5146                                  # sec128.3: dfinal = 9.5146*cos0 - 8.2480 (r2 0.9885)
COSGATE=0.8734                                    # sec127.3 ingredient ceiling (= soup_v2's cos0)
KITBASE=(95.4738,75.8957,96.0945)                 # r12_eval.BASE (selftest)
T0=_time.time()

bad=[i for i in range(3) if abs(base[i]-KITBASE[i])>1e-3]
print("\n=== HARNESS GATE: kit base %.4f / %.4f / %.4f  (r12 selftest wants %.4f / %.4f / %.4f)  ->  %s"
      %(base[0],base[1],base[2],KITBASE[0],KITBASE[1],KITBASE[2],"FAIL" if bad else "PASS"),flush=True)
if bad: raise SystemExit("HARNESS GATE FAILED - screen nothing")
print("  monitor windows: %d (re_lohi, rng 0)"%len(va_sub),flush=True)

# ---- r50_build.py helpers, verbatim ----
def load_sd(p):
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
def lincomb(terms):
    ref=terms[0][1]; out={}
    for k,v in ref.items():
        if "num_batches_tracked" in k:
            out[k]=v.clone()            # BN counter stored as float32 -> COPY, never average
        elif v.is_complex() or v.dtype.is_floating_point:
            dt=torch.complex64 if v.is_complex() else torch.float32
            acc=None
            for c,sd in terms:
                t=c*sd[k].to(dt); acc=t if acc is None else acc+t
            out[k]=acc.to(v.dtype)
        else: out[k]=v.clone()
    return out
def maxdiff(a,b):
    m=0.0
    for k in a:
        if "num_batches_tracked" in k: continue     # counters, not weights
        x,y=a[k],b[k]
        if x.is_complex(): x=torch.view_as_real(x.to(torch.complex64)); y=torch.view_as_real(y.to(torch.complex64))
        m=max(m,float((x.float()-y.float()).abs().max()))
    return m
# ---- FITBASE helpers ----
def reldist(a,b):
    """||a-b||_2 / ||b||_2 over every float/complex tensor except BN counters (complex as real pairs)."""
    num=den=0.0
    for k,y in b.items():
        if "num_batches_tracked" in k or not (y.is_complex() or y.dtype.is_floating_point): continue
        x=a[k]
        if y.is_complex(): x=torch.view_as_real(x.to(torch.complex64)); y=torch.view_as_real(y.to(torch.complex64))
        x=x.double(); y=y.double(); num+=float(((x-y)**2).sum()); den+=float((y*y).sum())
    return math.sqrt(num/max(den,1e-300))
REFK={k:v.detach().clone() for k,v in model.state_dict().items()}      # the KIT weights (model untouched so far)
KITSD={k:v.detach().clone().cpu() for k,v in REFK.items()}
def ev(sd,label):
    model.load_state_dict({k:sd[k].to(dtype=v.dtype) for k,v in REFK.items()},strict=True)
    model.eval(); r=evaluate()
    print("  %-42s rel_l2 %.4f  tke %.4f  mvpe %.4f | cos0 %.4f rho %.4f"%(label,r[0],r[1],r[2],LAST_COS,LAST_RHO),flush=True)
    return dict(rel=float(r[0]),tke=float(r[1]),mvpe=float(r[2]),cos0=float(LAST_COS),rho=float(LAST_RHO))
def pr_cached(label,r):
    print("  %-42s rel_l2 %.4f  tke %.4f  mvpe %.4f | cos0 %.4f rho %.4f  (cached)"%(label,r["rel"],r["tke"],r["mvpe"],r["cos0"],r["rho"]),flush=True)
def deltas(r,ref):
    d=dict(d_rel=r["rel"]-ref["rel"],d_tke=r["tke"]-ref["tke"],d_mvpe=r["mvpe"]-ref["mvpe"],d_cos0=r["cos0"]-ref["cos0"])
    d["d_final"]=PRICE["rel"]*d["d_rel"]+PRICE["tke"]*d["d_tke"]+PRICE["mvpe"]*d["d_mvpe"]
    d["leak"]=LEAKSLOPE*d["d_cos0"]
    d["frac_cos0"]=d["leak"]/d["d_final"] if abs(d["d_final"])>1e-12 else float("nan")
    return d

# ---- leaves + cache ----
OLD=[f"{LH}/ft_all_w15_lr1_final.pth",f"{LH}/ft_all_w15_lr3_final.pth",
     f"{LH}/ft_all_w33_lr1_final.pth",f"{LH}/ft_all_w33_lr3_final.pth"]
LEAF=dict(w15_lr1=OLD[0],w15_lr3=OLD[1],w33_lr1=OLD[2],w33_lr3=OLD[3],sv3=f"{B}/train_es/ftaug_sv3_3e5_10.pth")
BASE4=["w15_lr1","w15_lr3","w33_lr1","w33_lr3"]
CFG=json.load(open(f"{FB}/fb_arms.json"))
for k,p in CFG["leaves"].items(): LEAF[k]=p.format(B=B,LH=LH,CK=CK)
_SD={}
def leaf(n):
    if n not in _SD: _SD[n]=load_sd(LEAF[n])
    return _SD[n]
CACHEF=f"{FB}/screen_cache.json"
try: CACHE=json.load(open(CACHEF))
except Exception: CACHE={}
def save_cache(): json.dump(CACHE,open(CACHEF,"w"),indent=1)
def fkey(names): return "|".join("%s:%d:%d"%(n,os.path.getsize(LEAF[n]),int(os.path.getmtime(LEAF[n]))) for n in names)

print("\n=== GATE 1: mean(4 original ft_all_*) must equal soup_v2.pth bit-exactly ===",flush=True)
s2_file=load_sd(f"{LH}/soup_v2.pth")
s2_calc=lincomb([(0.25,leaf(n)) for n in BASE4])
d1=maxdiff(s2_file,s2_calc); print("  max|d| = %.6e  ->  %s"%(d1,"PASS" if d1<1e-6 else "FAIL"),flush=True)
if d1>=1e-6: raise SystemExit("GATE 1 FAILED")
print("\n=== GATE 2: 0.5*soup_v2 + 0.5*sv3 must equal the BANKED backbone ===",flush=True)
bank_file=load_sd(f"{B}/train_es/r19_screen_best.pth")
bank_calc=lincomb([(0.5,s2_file),(0.5,leaf("sv3"))])
TOL2=1e-3   # r19_screen_best.json souped soup_v2_FP16.pth, we use full-precision soup_v2.pth
d2=maxdiff(bank_file,bank_calc); print("  max|d| = %.6e  (tol %.0e, fp16-quantisation of the soup_v2 ingredient)  ->  %s"%(d2,TOL2,"PASS" if d2<TOL2 else "FAIL"),flush=True)
if d2>=TOL2: raise SystemExit("GATE 2 FAILED - the decomposition is wrong; do NOT build.")
print("  ** the ENTIRE banked artifact is reproduced from its 5 leaf checkpoints.",flush=True)
del s2_calc

SLOTS=dict(a="w15_lr1",b="w15_lr3",c="w33_lr1",d="w33_lr3",s="sv3")
def build(sl):
    half=lincomb([(0.25,leaf(sl[x])) for x in "abcd"])           # same two-stage construction as r50
    return half, lincomb([(0.5,half),(0.5,leaf(sl["s"]))])
PRIMARY="BANKED"
print("\n=== REFERENCES ===",flush=True)
REF={}
REF["BANKED"]=ev(bank_file,"BANKED r19_screen_best (live 79.484440)")
old=CACHE.get("REF:BANKED")
if old is not None and max(abs(old[k]-REF["BANKED"][k]) for k in ("rel","tke","mvpe","cos0","rho"))>1e-6:
    print("  !! BANKED no longer reproduces its cached value -> cache discarded",flush=True); CACHE={}
CACHE["REF:BANKED"]=dict(REF["BANKED"]); save_cache()
for nm,key,sdv,lab in (("BANKCALC","REF:BANKCALC|"+fkey(BASE4+["sv3"]),bank_calc,"BANKCALC 0.5*soup_v2+0.5*sv3 (fp32 leaves)"),
                       ("soup_v2","REF:soup_v2|"+fkey(BASE4),s2_file,"soup_v2 (banked soup half)")):
    if key in CACHE: REF[nm]=dict(CACHE[key]); pr_cached(lab,REF[nm])
    else: REF[nm]=ev(sdv,lab); CACHE[key]=dict(REF[nm]); save_cache()

print("\n=== MEMBERS (single leaf checkpoints; cos0 ingredient ceiling %.4f) ==="%COSGATE,flush=True)
MEM={}
for n,p in LEAF.items():
    if not os.path.exists(p): continue
    key="MEM|"+fkey([n]); cp=CFG["counterpart"].get(n)
    if key in CACHE:
        MEM[n]=dict(CACHE[key]); pr_cached("member %s"%n,MEM[n])
    else:
        MEM[n]=ev(leaf(n),"member %s"%n)
        MEM[n]["dist_kit"]=reldist(leaf(n),KITSD)
        if cp: MEM[n]["counterpart"]=cp; MEM[n]["dist_counterpart"]=reldist(leaf(n),leaf(cp))
        CACHE[key]=dict(MEM[n]); save_cache()
    msg="      |w-w_kit|/|w_kit| = %.4e"%MEM[n]["dist_kit"]
    if cp: msg+="   |w-w_%s|/|w_%s| = %.4e"%(cp,cp,MEM[n]["dist_counterpart"])
    print(msg,flush=True)
key="YARD|"+fkey(BASE4)
if key not in CACHE:
    CACHE[key]={"w15_lr3 vs w15_lr1":reldist(leaf("w15_lr3"),leaf("w15_lr1")),
                "w33_lr3 vs w33_lr1":reldist(leaf("w33_lr3"),leaf("w33_lr1")),
                "w15_lr3 vs w33_lr3":reldist(leaf("w15_lr3"),leaf("w33_lr3"))}; save_cache()
YARD=CACHE[key]
for k,v in YARD.items(): print("  yardstick |d|/|w| %-22s %.4e"%(k,v),flush=True)

def verdict(lab,d,ingr):
    if lab.startswith("C"): return "CONTROL (replicate: noise floor)"
    if d["d_final"]<=0: return "DEAD (local-negative)"
    if d["d_final"]>=0.20: return "REJECT (magnitude law: dfinal >= 0.20)"
    if d["leak"]>0 and d["leak"]>=d["d_final"]:
        return "FLAG leakage-signature (%.0f%% of dfinal explained by the cos0 rise) - not built"%(100*d["leak"]/d["d_final"])
    hi=["%s %.4f"%(k,c) for k,c in ingr.items() if c>COSGATE]
    if hi: return "REJECT (cos0 ingredient gate > %.4f: %s)"%(COSGATE,", ".join(hi))
    return "ADVANCE"

print("\n=== ARMS: backbone = 0.5*mean(4 ft_all slots) + 0.5*(sv3 slot); deltas vs %s ==="%PRIMARY,flush=True)
ARM={}
for lab,ov in CFG["arms"].items():
    sl=dict(SLOTS); sl.update(ov)
    names=[sl[x] for x in "abcds"]
    miss=[n for n in names if not os.path.exists(LEAF[n])]
    if miss: print("  %-42s SKIP (missing: %s)"%(lab,", ".join(miss)),flush=True); continue
    ch_soup=any(sl[x]!=SLOTS[x] for x in "abcd"); ch_sv3=(sl["s"]!=SLOTS["s"])
    ka="ARM|"+fkey(names); kh="HALF|"+fkey(names[:4])
    ingr={}
    if ka in CACHE and ((not ch_soup) or kh in CACHE):
        r=dict(CACHE[ka]); pr_cached(lab,r)
        if ch_soup: ingr["soup_half"]=CACHE[kh]["cos0"]; pr_cached("   ingredient: soup half",CACHE[kh])
    else:
        half,bb=build(sl)
        r=ev(bb,lab); r["dist_vs_bankcalc"]=reldist(bb,bank_calc); CACHE[ka]=dict(r)
        if ch_soup:
            hr=ev(half,"   ingredient: soup half"); CACHE[kh]=dict(hr); ingr["soup_half"]=hr["cos0"]
        save_cache(); del half,bb
    if ch_sv3: ingr["sv3_half"]=MEM[sl["s"]]["cos0"]
    newm={sl[x]:MEM[sl[x]]["cos0"] for x in "abcds" if sl[x]!=SLOTS[x]}
    d=deltas(r,REF[PRIMARY]); dc=deltas(r,REF["BANKCALC"])
    ARM[lab]=dict(slots=sl,score=r,d=d,d_vs_bankcalc=dc,ingredient_cos0=ingr,new_member_cos0=newm,
                  dist_vs_bankcalc=r.get("dist_vs_bankcalc"))
    ARM[lab]["verdict"]=verdict(lab,d,ingr)
    fr=("%.0f%%"%(100*d["frac_cos0"])) if d["frac_cos0"]==d["frac_cos0"] else "n/a"
    print("      d_rel %+.4f d_tke %+.4f d_mvpe %+.4f d_cos0 %+.4f | dfinal %+.4f (vs BANKCALC %+.4f) | 9.5146*dcos0 %+.4f = %s of dfinal"
          %(d["d_rel"],d["d_tke"],d["d_mvpe"],d["d_cos0"],d["d_final"],dc["d_final"],d["leak"],fr),flush=True)
    print("      |bb-bankcalc|/|bankcalc| %.4e | ingredient cos0 %s | new-member cos0 %s"
          %(ARM[lab]["dist_vs_bankcalc"],{k:round(v,4) for k,v in ingr.items()},{k:round(v,4) for k,v in newm.items()}),flush=True)
    print("      VERDICT: %s"%ARM[lab]["verdict"],flush=True)

R0=REF[PRIMARY]
print("\n=== SUMMARY vs %s: rel %.4f tke %.4f mvpe %.4f cos0 %.4f ==="%(PRIMARY,R0["rel"],R0["tke"],R0["mvpe"],R0["cos0"]),flush=True)
print("  %-28s %8s %8s %8s %8s %8s | %8s %8s %8s %8s | %8s %8s %6s | %s"%("arm","rel","tke","mvpe","cos0","rho","d_rel","d_tke","d_mvpe","d_cos0","dfinal","9.51dcos","frac","verdict"))
for lab,A_ in ARM.items():
    s=A_["score"]; d=A_["d"]; fr=("%5.0f%%"%(100*d["frac_cos0"])) if d["frac_cos0"]==d["frac_cos0"] else "   n/a"
    print("  %-28s %8.4f %8.4f %8.4f %8.4f %8.4f | %+8.4f %+8.4f %+8.4f %+8.4f | %+8.4f %+8.4f %6s | %s"
          %(lab,s["rel"],s["tke"],s["mvpe"],s["cos0"],s["rho"],d["d_rel"],d["d_tke"],d["d_mvpe"],d["d_cos0"],d["d_final"],d["leak"],fr,A_["verdict"]))
cand=[A_ for l,A_ in ARM.items() if not l.startswith("C")]
if len(cand)>=3:
    x=np.array([A_["d"]["d_cos0"] for A_ in cand]); y=np.array([A_["d"]["d_final"] for A_ in cand])
    rr=float(np.corrcoef(x,y)[0,1]); sl_=float(np.polyfit(x,y,1)[0])
    print("  corr(dfinal, dcos0) over %d candidate arms = %+.3f (r2 %.3f), OLS slope %.3f  [sec128.4.3 check]"%(len(cand),rr,rr*rr,sl_),flush=True)
OUTJ=dict(ref=REF,members=MEM,arms=ARM,yard=YARD,primary=PRIMARY,gates=dict(gate1=d1,gate2=d2,kit_base=list(base)),
          when=_time.strftime("%Y-%m-%d %H:%M:%S"))
json.dump(OUTJ,open(f"{FB}/screen_results.json","w"),indent=1,default=float)
shutil.copy(f"{FB}/screen_results.json",f"{FB}/screen_results_%s.json"%_time.strftime("%m%d_%H%M%S"))

if MODE=="build":
    lab=os.environ["FB_ARM"]; ztag=os.environ["FB_ZIPTAG"]
    if lab not in ARM: raise SystemExit("REFUSE: arm %s was not screened in this run"%lab)
    if ARM[lab]["verdict"]!="ADVANCE": raise SystemExit("REFUSE to build %s: %s"%(lab,ARM[lab]["verdict"]))
    DSTN=f"FITBASE_{ztag}"; DST=f"{B}/submissions/submission_{DSTN}.zip"
    if os.path.exists(DST): raise SystemExit("REFUSE: %s already exists - never overwrite a zip"%DST)
    _,new=build(ARM[lab]["slots"])
    packed,ck={},[]
    for k,v in new.items():
        if v.is_complex(): ck.append(k); packed[k]=torch.view_as_real(v.to(torch.complex64)).half()
        elif v.dtype.is_floating_point: packed[k]=v.half()
        else: packed[k]=v
    bp=f"{FB}/{DSTN}_fp16.pth"; torch.save({"state_fp16":packed,"complex_keys":ck},bp)
    raw=torch.load(bp,map_location="cpu"); cks=set(raw["complex_keys"]); mx=0.0
    for k,v in raw["state_fp16"].items():
        if "num_batches_tracked" in k: continue
        w=torch.view_as_complex(v.float()) if k in cks else v.float()
        mx=max(mx,(w-new[k].to(w.dtype)).abs().max().item())
    print("\npacked -> %s B (%d complex) | fp16 round-trip max|d| = %.3e"%(format(os.path.getsize(bp),","),len(ck),mx),flush=True)
    r16=ev(load_sd(bp),"packed fp16 CANDIDATE")
    zs=zipfile.ZipFile(f"{B}/submissions/submission_SCREEN.zip")
    sbp=f"{FB}/_SCREEN_backbone_fp16.pth"; open(sbp,"wb").write(zs.read("sim_real_fno_fp16.pth"))
    try:
        b16=ev(load_sd(sbp),"packed fp16 BANKED (SCREEN zip)"); d16=deltas(r16,b16)
        print("  fp16-vs-fp16: d_rel %+.4f d_tke %+.4f d_mvpe %+.4f d_cos0 %+.4f | dfinal %+.4f"
              %(d16["d_rel"],d16["d_tke"],d16["d_mvpe"],d16["d_cos0"],d16["d_final"]),flush=True)
    except Exception as e: print("  (could not evaluate the SCREEN backbone: %r)"%(e,),flush=True)
    SRC=f"{B}/submissions/submission_SV2.zip"
    zin=zipfile.ZipFile(SRC); nb=open(bp,"rb").read()
    zo=zipfile.ZipFile(DST,"w",zipfile.ZIP_DEFLATED); n=0
    for it in zin.infolist():
        dd=nb if it.filename=="sim_real_fno_fp16.pth" else zin.read(it.filename)
        if it.filename=="sim_real_fno_fp16.pth": n+=1
        zi=zipfile.ZipInfo(it.filename,date_time=it.date_time)
        zi.compress_type=it.compress_type; zi.external_attr=it.external_attr
        zo.writestr(zi,dd)
    zo.close(); assert n==1
    zc=zipfile.ZipFile(DST); ext=sum(i.file_size for i in zc.infolist())
    same=all(zc.read(x)==zin.read(x) for x in zin.namelist() if x!="sim_real_fno_fp16.pth")
    same_screen=all(zc.read(x)==zs.read(x) for x in zs.namelist() if x!="sim_real_fno_fp16.pth")
    npyc=sum(1 for x in zc.namelist() if x.endswith(".pyc"))
    ok=os.system("unzip -tq "+DST+" >/dev/null")==0
    print("\n"+DST,flush=True)
    print("  other entries byte-identical to SV2: %s | to SCREEN: %s | entry list identical: %s | .pyc %d"
          %(same,same_screen,zc.namelist()==zin.namelist(),npyc),flush=True)
    print("  extracted %s B = %.2f%% of cap (< 268,435,456: %s) | unzip -t %s"%(format(ext,","),100*ext/268435456,ext<268435456,ok),flush=True)
    print("  backbone md5 %s"%hashlib.md5(zc.read("sim_real_fno_fp16.pth")).hexdigest(),flush=True)
    print("  zip md5      %s"%hashlib.md5(open(DST,"rb").read()).hexdigest(),flush=True)
    if not (same and same_screen and npyc==0 and ok and ext<268435456 and zc.namelist()==zin.namelist()):
        raise SystemExit("BUILD VERIFICATION FAILED")
print("\n[done] fitbase %s  %.0fs"%(MODE,_time.time()-T0),flush=True)
