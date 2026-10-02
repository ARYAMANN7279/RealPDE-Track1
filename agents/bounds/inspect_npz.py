"""List keys/shapes/dtypes/ranges of npz files. Arg forms: path.npz  or  zip:path.zip::member.npz"""
import io
import os
import sys
import zipfile

import numpy as np


def _open(spec):
    if spec.startswith("zip:"):
        zp, mem = spec[4:].split("::")
        with zipfile.ZipFile(zp) as zf:
            return np.load(io.BytesIO(zf.read(mem)), allow_pickle=False)
    return np.load(spec, allow_pickle=False, mmap_mode="r")


for spec in sys.argv[1:]:
    print("==", spec)
    try:
        z = _open(spec)
    except Exception as ex:  # noqa: BLE001
        print("   cannot open:", ex)
        continue
    files = z.files
    print("   n_keys", len(files))
    for k in files:
        a = z[k]
        rng = ""
        if a.size and a.dtype.kind in "fiu" and a.size < 5e7:
            rng = "min %.6g max %.6g" % (float(np.min(a)), float(np.max(a)))
        elif a.size and a.dtype.kind in "fiu":
            rng = "(large)"
        print("   %-28s %-22s %-8s %s" % (k, str(a.shape), a.dtype, rng))
