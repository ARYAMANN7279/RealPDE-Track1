import os
p = "/SML_DISK_24TB/rajeshr/Aryamann/UGP/train_mvpe/errnet2.py"
s = open(p).read()
if "--pert" in s:
    print("already patched"); raise SystemExit
s = s.replace('ap.add_argument("--rich", action="store_true")',
              'ap.add_argument("--rich", action="store_true")\nap.add_argument("--pert", action="store_true")')
old = "CI = 80 + (RF.shape[-1]*20 if RF is not None else 0)"
new = '''PT = None
if a.pert:
    pc = np.load(f"{B}/train_mvpe/runs/pert_cache.npz")
    PT = np.concatenate([np.log(pc["DN"]+1e-12), np.log(pc["DT"]+1e-12)], -1).astype(np.float32)
    m2, s2 = PT[TR].reshape(-1, PT.shape[-1]).mean(0), PT[TR].reshape(-1, PT.shape[-1]).std(0)+1e-6
    PT = ((PT-m2)/s2).astype(np.float32)
    print("[feat] pert channels", PT.shape, flush=True)
CI = 80 + (RF.shape[-1]*20 if RF is not None else 0) + (PT.shape[-1]*20 if PT is not None else 0)'''
assert old in s, "anchor missing"
s = s.replace(old, new)
s = s.replace("    if RF is not None: parts.append(flat(RF[ix]))",
              "    if RF is not None: parts.append(flat(RF[ix]))\n    if PT is not None: parts.append(flat(PT[ix]))")
open(p, "w").write(s)
print("patched --pert into errnet2.py")
