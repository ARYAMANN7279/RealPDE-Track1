import zipfile, os, sys, hashlib
import numpy as np

zip_path = "/SML_DISK_24TB/rajeshr/Aryamann/UGP/submissions/submission_SV2.zip"

print(f"=== CODALAB PRE-FLIGHT CHECK ===")
print(f"Target: {os.path.basename(zip_path)}\n")

# 1. FILE SIZE CHECK
size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print(f"1. File Size: {size_mb:.2f} MB")
if size_mb > 256.0:
    print("  [FAIL] ZIP exceeds CodaLab 256MB limit!")
else:
    print("  [PASS] Size is within limits.")

# 2. ZIP STRUCTURE CHECK
with zipfile.ZipFile(zip_path, 'r') as z:
    file_list = z.namelist()
    print(f"\n2. ZIP Contents ({len(file_list)} files):")
    has_submission = False
    has_fno = False
    has_pyc = False
    
    extracted_size_mb = sum(info.file_size for info in z.infolist()) / (1024 * 1024)
    print(f"   Uncompressed Size: {extracted_size_mb:.2f} MB")
    
    for f in file_list:
        if f == "submission.py": has_submission = True
        if f == "sim_real_fno_fp16.pth": has_fno = True
        if ".pyc" in f or "__pycache__" in f:
            has_pyc = True
            print(f"  [FAIL] Contains forbidden file: {f}")

    if not has_submission:
        print("  [FAIL] Missing submission.py at root!")
    elif not has_fno:
        print("  [FAIL] Missing sim_real_fno_fp16.pth at root!")
    elif has_pyc:
        print("  [FAIL] Contains .pyc or __pycache__ (CodaLab will crash).")
    elif extracted_size_mb > 256.0:
        print("  [FAIL] Uncompressed size exceeds 256MB! The CodaLab builder will crash during extraction.")
    else:
        print("  [PASS] Structure is perfectly compliant.")

# 3. MD5 HASH
md5 = hashlib.md5(open(zip_path, 'rb').read()).hexdigest()
print(f"\n3. MD5 Checksum: {md5}")

print("\n=== PRE-FLIGHT COMPLETE ===")
