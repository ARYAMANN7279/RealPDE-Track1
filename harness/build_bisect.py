"""Build bisect artifacts by surgically editing the PROVEN SOUP_v1 zip.

Everything in MAXSOUP has been verified structurally identical to SOUP_v1 (which
scored 78.45), so the failure cannot be localised by inspection any further. These
zips change exactly ONE component each, so a submission result identifies the cause:

  BISECT_CKPT : SOUP_v1 zip, only sim_real_fno_fp16.pth swapped to the new soup.
                Also a legitimate candidate in its own right: new model accuracy,
                v1's already-proven bounds head.
  BISECT_HEAD : SOUP_v1 zip, only head_assets.npz swapped to the new one.

Every other byte -- submission.py, load_baseline.py, rpde_baselines/, _vendor/,
compression, entry order -- is copied through untouched from the working archive.
"""
import zipfile, shutil, os, hashlib, sys

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
SRC = f"{B}/submissions/submission_SOUP_v1.zip"
NEW = f"{B}/submissions/submission_MAXSOUP_v2.zip"

def md5(b):
    return hashlib.md5(b).hexdigest()

def rebuild(out, swap):
    """Copy SRC entry-by-entry, replacing the named entries from NEW."""
    zin = zipfile.ZipFile(SRC)
    znew = zipfile.ZipFile(NEW)
    if os.path.exists(out):
        os.remove(out)
    zout = zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED)
    swapped = []
    for info in zin.infolist():
        if info.filename in swap:
            data = znew.read(info.filename)
            swapped.append(info.filename)
        else:
            data = zin.read(info.filename)
        # preserve the original entry metadata (date, compress type, attrs)
        ni = zipfile.ZipInfo(info.filename, date_time=info.date_time)
        ni.compress_type = info.compress_type
        ni.external_attr = info.external_attr
        ni.internal_attr = info.internal_attr
        ni.create_system = info.create_system
        zout.writestr(ni, data)
    zout.close()
    print("  %s: swapped %s" % (os.path.basename(out), swapped))
    return out

for name, swap in [("submission_BISECT_CKPT.zip", {"sim_real_fno_fp16.pth"}),
                   ("submission_BISECT_HEAD.zip", {"head_assets.npz"})]:
    out = f"{B}/submissions/{name}"
    rebuild(out, swap)
    z = zipfile.ZipFile(out)
    bad = z.testzip()
    tot = sum(i.file_size for i in z.infolist())
    print("     entries=%d  integrity=%s  extracted=%.1f MB" % (
        len(z.infolist()), "OK" if bad is None else "CORRUPT:" + str(bad), tot / 1e6))
    # confirm the intended component actually differs and everything else does not
    zs, zn = zipfile.ZipFile(SRC), zipfile.ZipFile(NEW)
    diffs = []
    for i in z.infolist():
        a = zs.read(i.filename); b = z.read(i.filename)
        if md5(a) != md5(b):
            diffs.append(i.filename)
    print("     differs from SOUP_v1 in exactly: %s" % diffs)
    assert diffs == list(swap), "unexpected extra differences: %s" % diffs
print("\nboth bisect zips built and verified")
