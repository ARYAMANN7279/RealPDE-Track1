"""FITBASE pre-check (CPU only): is the SV2 template identical to the BANKED SCREEN zip except the backbone?
Also records the env versions (for replicate / env-drift notes) and extracts SV2's submission.py for reading."""
import zipfile, hashlib, os, sys
import torch
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
SUB = f"{B}/submissions"
OUT = f"{B}/agents/fitbase"
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.version.cuda,
      "| cudnn", torch.backends.cudnn.version())
Z = {n: zipfile.ZipFile(f"{SUB}/submission_{n}.zip") for n in ["SV2", "SCREEN", "RECIPE2"]}
names = {n: z.namelist() for n, z in Z.items()}
print("n entries:", {n: len(v) for n, v in names.items()})
print("entry list identical  SV2==SCREEN: %s   SV2==RECIPE2: %s"
      % (names["SV2"] == names["SCREEN"], names["SV2"] == names["RECIPE2"]))
alln = sorted(set().union(*names.values()))
ndiff = 0
for x in alln:
    row = []
    for n, z in Z.items():
        row.append(hashlib.md5(z.read(x)).hexdigest() if x in names[n] else "-")
    same = len(set(row)) == 1
    if not same: ndiff += 1
    if (not same) or x in ("sim_real_fno_fp16.pth", "bounds_assets.npz", "submission.py"):
        print("  %-32s SV2 %s | SCREEN %s | RECIPE2 %s  %s" % (x, row[0][:12], row[1][:12], row[2][:12],
                                                                 "SAME" if same else "DIFF"))
print("entries differing across the 3 zips: %d of %d" % (ndiff, len(alln)))
for n in Z:
    print("zip md5 %-8s %s" % (n, hashlib.md5(open(f"{SUB}/submission_{n}.zip", "rb").read()).hexdigest()))
open(f"{OUT}/sv2_submission_py.txt", "wb").write(Z["SV2"].read("submission.py"))
print("[zipcmp done]")
