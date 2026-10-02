import re
with open('/Users/aryamannsrivastava/Desktop/IMPORTANT/UGP/build_asym.py', 'r') as f:
    content = f.read()

replacement = """
        for k in range(NB):
            v=np.sort(es[cut[k]:cut[k+1]])
            if v.size<200: LUT[k,ci]=0.012; continue
            kk=np.arange(1,v.size+1)/v.size
            LUT[k,ci]=float(v[np.argmax(np.exp(-3*v/SIG)*kk)])
"""

content = re.sub(r'        for k in range\(NB\):.*?LUT\[k,ci\]=float\(v\[np.argmax\(np.exp\(-2\*v/SIG\)\*kk\)\]\)', replacement, content, flags=re.DOTALL)

with open('/Users/aryamannsrivastava/Desktop/IMPORTANT/UGP/build_asym_squeeze_fixed.py', 'w') as f:
    f.write(content)
