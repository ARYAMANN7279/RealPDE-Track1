"""Create variant directories $D/var_<name>/ that each hold a generated submission.py (from
sub_template.py with a fixed flag set) and symlinks to the UNCHANGED SCREEN payload.

  $P mkvariants.py base=  v1=V1 v2=V2 v3=V3 v4=V4 v5=V5 all=V1,V2,V3,V4,V5
('base' = the template with no flags; it must reproduce SCREEN bit-for-bit.)
No GPU work.  Prints md5 of every generated submission.py.
"""
import os, sys, hashlib

D = os.path.dirname(os.path.abspath(__file__))
SCREEN = os.path.join(D, "screen")
PAYLOAD = ["sim_real_fno_fp16.pth", "bounds_assets.npz", "load_baseline.py", "rpde_baselines",
           "_vendor"]
tmpl = open(os.path.join(D, "sub_template.py")).read()
MARK = "_FLAGS = frozenset()   # @@FLAGS@@"
assert tmpl.count(MARK) == 1, "flag marker missing"

for spec in sys.argv[1:]:
    name, flags = spec.split("=", 1)
    fl = [f for f in flags.split(",") if f]
    vd = os.path.join(D, "var_" + name)
    os.makedirs(vd, exist_ok=True)
    for p in PAYLOAD:
        dst = os.path.join(vd, p)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(SCREEN, p), dst)
    src = tmpl.replace(MARK, "_FLAGS = frozenset(%r)" % (sorted(fl),))
    with open(os.path.join(vd, "submission.py"), "w") as f:
        f.write(src)
    md5 = hashlib.md5(src.encode()).hexdigest()
    print(f"var_{name:6s} flags={fl} submission.py md5 {md5}")
