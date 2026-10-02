"""Build $B/submissions/submission_SPEED_<tag>.zip = submission_SCREEN.zip with ONLY
submission.py replaced.  Refuses to overwrite anything.  Verifies: testzip clean, entry list
identical (names AND order) to SCREEN, 0 .pyc, extracted size < 268,435,456 B, payload md5s
(checkpoint and bounds unchanged; submission.py = the new file)."""
import argparse, hashlib, os, sys, time, zipfile

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--sub", required=True, help="new submission.py")
ap.add_argument("--out", required=True)
args = ap.parse_args()

if os.path.exists(args.out) or os.path.exists(args.out + ".partial"):
    raise SystemExit(f"refusing to overwrite {args.out}")
new_sub = open(args.sub, "rb").read()


def md5(b):
    return hashlib.md5(b).hexdigest()


with zipfile.ZipFile(args.src) as zi, zipfile.ZipFile(args.out + ".partial", "w") as zo:
    for info in zi.infolist():
        data = zi.read(info.filename)
        dt = info.date_time
        if info.filename == "submission.py":
            data = new_sub
            dt = time.localtime()[:6]
        ni = zipfile.ZipInfo(info.filename, date_time=dt)
        ni.compress_type = info.compress_type
        ni.external_attr = info.external_attr
        ni.create_system = info.create_system
        zo.writestr(ni, data)
os.rename(args.out + ".partial", args.out)

with zipfile.ZipFile(args.src) as a, zipfile.ZipFile(args.out) as b:
    na = [i.filename for i in a.infolist()]
    nb = [i.filename for i in b.infolist()]
    bad = b.testzip()
    size = sum(i.file_size for i in b.infolist())
    pyc = [n for n in nb if n.endswith(".pyc") or "__pycache__" in n]
    rows = {}
    for n in ("sim_real_fno_fp16.pth", "bounds_assets.npz", "submission.py"):
        rows[n] = (md5(a.read(n)), md5(b.read(n)))
    others_same = all(a.read(n) == b.read(n) for n in na if n != "submission.py")
print(f"out {args.out}")
print(f"testzip: {'OK' if bad is None else 'BAD ' + str(bad)}")
print(f"entries: {len(nb)}  identical names+order to SCREEN: {na == nb}")
print(f"all non-submission.py entries byte-identical to SCREEN: {others_same}")
print(f"pyc/__pycache__ entries: {len(pyc)}")
print(f"extracted size {size} B  (< 268435456: {size < 268435456})")
for n, (x, y) in rows.items():
    print(f"  {n:24s} SCREEN {x}  NEW {y}  {'same' if x == y else 'CHANGED'}")
print(f"new submission.py md5 (file) {md5(new_sub)}")
print(f"zip md5 {md5(open(args.out, 'rb').read())}  size {os.path.getsize(args.out)} B")
