"""Package: official fp16 checkpoint + kit loaders + submission.py + head_assets.npz.
The checkpoint is the Drive baseline, byte-identical -- no weights are trained here."""
import os, shutil, sys, zipfile
from importlib.machinery import SourceFileLoader
HERE = os.path.dirname(os.path.abspath(__file__))
C = SourceFileLoader("c", os.path.join(HERE, "00_config.py")).load_module()
OUT = os.path.join(C.WORK, "submission.zip")
BASE = os.environ.get("BASE_ZIP", os.path.join(C.ROOT, "submissions/submission_fno_plain_sps.zip"))
shutil.copy(BASE, OUT)
z = zipfile.ZipFile(OUT, "a", zipfile.ZIP_DEFLATED)
z.write(os.path.join(HERE, "submission.py"), "submission.py")
z.write(os.path.join(C.WORK, "head_assets.npz"), "head_assets.npz")
z.close()
tot = sum(i.file_size for i in zipfile.ZipFile(OUT).infolist())
print("built %s  (%.1f MB extracted, cap 256)" % (OUT, tot/1048576))
