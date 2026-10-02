import os
LH = "/SML_DISK_24TB/rajeshr/Aryamann/UGP/local_harness"
src = open(os.path.join(LH, "re_extrap.py")).read()
old = '''RE = np.array([int(names[i].split("_")[0]) for i in wt])
uniq_re = sorted(set(RE.tolist()))
lo_re, hi_re = uniq_re[:3], uniq_re[-3:]
EXTRAP = np.isin(RE, lo_re + hi_re)            # unseen regimes
TRAIN = ~EXTRAP'''
assert old in src, "anchor block not found"
new = '''SHIFT = os.environ.get("SHIFT", "re")
RE  = np.array([int(names[i].split("_")[0]) for i in wt])
AOA = np.array([int(names[i].split("_")[1].replace(".h5", "")) for i in wt])
uniq_re  = sorted(set(RE.tolist()))
uniq_aoa = sorted(set(AOA.tolist()))
if SHIFT == "re":
    lo_re, hi_re = uniq_re[:3], uniq_re[-3:]
    EXTRAP = np.isin(RE, lo_re + hi_re)
    DESC = "Re EXTRAPOLATION: hold out %s + %s" % (lo_re, hi_re)
elif SHIFT == "aoa_edge":       # lowest + highest angle: the AoA analogue of re
    EXTRAP = np.isin(AOA, [uniq_aoa[0], uniq_aoa[-1]])
    DESC = "AoA EXTRAPOLATION: hold out %s + %s" % (uniq_aoa[0], uniq_aoa[-1])
elif SHIFT == "aoa_mid":        # one interior angle: interpolation control
    mid = uniq_aoa[len(uniq_aoa) // 2]
    EXTRAP = AOA == mid
    DESC = "AoA INTERPOLATION control: hold out %s" % mid
else:
    raise SystemExit("bad SHIFT")
TRAIN = ~EXTRAP
print("angles present: %s | Re present: %s" % (uniq_aoa, uniq_re))
print(DESC)'''
src = src.replace(old, new)
src = src.replace('"Re-EXTRAP split (real)"', '"HELD-OUT REGIME"')
src = src.replace('''print("Re extrapolation split: train on %d interior Re, test on %s + %s"
      % (len(uniq_re) - 6, lo_re, hi_re))
''', '')
src = src.replace('"  train windows %d | extrapolation windows %d | (old random split ev=%d)"',
                  '"  train windows %d | held-out windows %d | (old random split ev=%d)"')
out = os.path.join(LH, "shift_extrap.py")
open(out, "w").write(src)
print("wrote", out)
