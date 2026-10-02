"""Package the SOUP submission.

Unlike train/06_build_submission.py this REPLACES the checkpoint rather than
appending to a base zip. Appending a name that already exists would leave two
entries called sim_real_fno_fp16.pth in the archive, and which one wins on
extract is undefined -- exactly the kind of silent mismatch that has cost slots
before. The archive is rebuilt entry by entry instead, which also lets us drop
__pycache__/*.pyc (the Aug 19 soup zips carried them; ROBUST and v7 do not).
"""
import os, sys, zipfile
from importlib.machinery import SourceFileLoader
HERE = os.path.dirname(os.path.abspath(__file__))
C = SourceFileLoader("c", os.path.join(HERE, "00_config.py")).load_module()
OUT  = os.path.join(C.WORK, "submission.zip")
CK   = os.path.join(C.WORK, "sim_real_fno_fp16.pth")   # packed soup
BASE = os.environ.get("BASE_ZIP", os.path.join(C.ROOT, "submissions/submission_fno_plain_sps.zip"))
REPLACE = {"submission.py", "head_assets.npz", "sim_real_fno_fp16.pth"}

assert os.path.exists(CK), "packed soup checkpoint missing: %s" % CK
zin = zipfile.ZipFile(BASE)
if os.path.exists(OUT):
    os.remove(OUT)
zout = zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED)
kept = dropped = 0
for it in zin.infolist():
    n = it.filename
    if n in REPLACE:
        continue
    if "__pycache__" in n or n.endswith(".pyc"):
        dropped += 1
        continue
    zout.writestr(it, zin.read(n)); kept += 1
zout.write(os.path.join(HERE, "submission.py"), "submission.py")
zout.write(os.path.join(C.WORK, "head_assets.npz"), "head_assets.npz")
zout.write(CK, "sim_real_fno_fp16.pth")
zout.close()

z = zipfile.ZipFile(OUT)
names = [i.filename for i in z.infolist()]
assert names.count("sim_real_fno_fp16.pth") == 1, "duplicate checkpoint entry"
assert names.count("submission.py") == 1, "duplicate submission.py entry"
assert not [n for n in names if n.endswith(".pyc")], "pyc leaked into archive"
tot = sum(i.file_size for i in z.infolist())
print("kept %d entries, dropped %d pycache" % (kept, dropped))
print("built %s  (%.1f MB extracted, cap 256)" % (OUT, tot / 1048576))
assert tot / 1048576 < 256, "OVER THE 256 MB CAP"
