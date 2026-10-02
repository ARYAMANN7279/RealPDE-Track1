import re
with open('/Users/aryamannsrivastava/Desktop/IMPORTANT/UGP/build_asym.py', 'r') as f:
    content = f.read()

# Replace the fit_lut function
new_fit_lut = """
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
            
            # Conformal Prediction: Use 95th percentile
            idx = int(0.95 * v.size)
            if idx >= v.size: idx = v.size - 1
            LUT[k,ci] = float(v[idx])
            
        print("   ch%d %d elem | LUT min %.5f med %.5f max %.5f | %d/%d non-monotone"
              %(ci,s.size,LUT[:,ci].min(),np.median(LUT[:,ci]),LUT[:,ci].max(),
                int((np.diff(LUT[:,ci])<0).sum()),NB-1),flush=True)
    return LUT, ED
"""

content = re.sub(r'def fit_lut\(Ww, EFF\):.*?return LUT, ED', new_fit_lut, content, flags=re.DOTALL)
content = content.replace('a.tag', 'a.tag+"_q95"')
with open('/Users/aryamannsrivastava/Desktop/IMPORTANT/UGP/build_asym_conf_q95.py', 'w') as f:
    f.write(content)

content = content.replace('int(0.95 * v.size)', 'int(0.99 * v.size)')
content = content.replace('a.tag+"_q95"', 'a.tag+"_q99"')
with open('/Users/aryamannsrivastava/Desktop/IMPORTANT/UGP/build_asym_conf_q99.py', 'w') as f:
    f.write(content)

content = content.replace('int(0.99 * v.size)', 'int(0.90 * v.size)')
content = content.replace('a.tag+"_q99"', 'a.tag+"_q90"')
with open('/Users/aryamannsrivastava/Desktop/IMPORTANT/UGP/build_asym_conf_q90.py', 'w') as f:
    f.write(content)
