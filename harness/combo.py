exec(open("disagreement.py").read().split("print(\"\\n=== E on held-out")[0])
import itertools
print("\n=== COMBINED signals: does disagreement ADD to the head? ===")
head_sig = np.exp(MU)
dis3 = sigs["disagree K=3"]; dis9 = sigs["disagree K=9"]

def best_h(v):
    v = np.sort(v)
    if v.size == 0: return 0.012
    k = np.arange(1, v.size+1)/v.size
    return float(v[np.argmax(np.exp(-2*v/SIG)*k)])

def E_const(hu,hv):
    eu=ERR[ev][...,0][SCM[ev][...,0]]; evv=ERR[ev][...,1][SCM[ev][...,1]]
    return float((np.exp(-2*hu/SIG)*(eu<=hu)).sum()+(np.exp(-2*hv/SIG)*(evv<=hv)).sum())/(eu.size+evv.size)

def E_1d(sig, NB=24):
    num=0.0; tot=0
    for ci in range(C):
        mf=SCM[...,ci]&fit[:,None,None,None]; me=SCM[...,ci]&ev[:,None,None,None]
        sf=sig[...,ci][mf]; ef=ERR[...,ci][mf]
        q=np.quantile(sf,np.linspace(0,1,NB+1)[1:-1]); b=np.digitize(sf,q)
        LUT=np.array([best_h(ef[b==k][::3]) if (b==k).sum()>200 else 0.012 for k in range(NB)],np.float32)
        h=LUT[np.digitize(sig[...,ci][me],q)]; ee=ERR[...,ci][me]
        num+=float((np.exp(-2*h/SIG)*(ee<=h)).sum()); tot+=ee.size
    return num/tot

def E_2d(s1, s2, NB=8):
    num=0.0; tot=0
    for ci in range(C):
        mf=SCM[...,ci]&fit[:,None,None,None]; me=SCM[...,ci]&ev[:,None,None,None]
        a_f=s1[...,ci][mf]; b_f=s2[...,ci][mf]; ef=ERR[...,ci][mf]
        qa=np.quantile(a_f,np.linspace(0,1,NB+1)[1:-1]); qb=np.quantile(b_f,np.linspace(0,1,NB+1)[1:-1])
        ia=np.digitize(a_f,qa); ib=np.digitize(b_f,qb); cell=ia*NB+ib
        LUT=np.full(NB*NB,0.012,np.float32)
        for k in range(NB*NB):
            s=(cell==k)
            if s.sum()>200: LUT[k]=best_h(ef[s][::3])
        ia_e=np.digitize(s1[...,ci][me],qa); ib_e=np.digitize(s2[...,ci][me],qb)
        h=LUT[ia_e*NB+ib_e]; ee=ERR[...,ci][me]
        num+=float((np.exp(-2*h/SIG)*(ee<=h)).sum()); tot+=ee.size
    return num/tot

ec=E_const(0.0129,0.0098)
print("  %-34s E %.4f"%("constants",ec))
r={}
r["head alone"]=E_1d(head_sig)
r["disagree K=3 alone"]=E_1d(dis3)
r["geo-mean(head,dis3)"]=E_1d(np.sqrt(head_sig*dis3))
r["2D LUT (head x dis3)"]=E_2d(head_sig,dis3)
r["2D LUT (head x dis9)"]=E_2d(head_sig,dis9)
for k,v in r.items():
    print("  %-34s E %.4f  (%+.4f vs constants, %+.4f vs head)"%(k,v,v-ec,v-r["head alone"]))
W=0.684538
print("\n  gain over head, IF it transfers 1:1 -> final:")
for k,v in r.items():
    if k!="head alone":
        print("    %-32s %+.3f"%(k,(v-r["head alone"])*100*W*0.217))
