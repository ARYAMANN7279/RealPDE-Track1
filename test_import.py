import sys, os, subprocess

zip_path = "/SML_DISK_24TB/rajeshr/Aryamann/UGP/submissions/submission_SV2.zip"
d = "/SML_DISK_24TB/rajeshr/Aryamann/UGP/_tmp/test_import_sv2"
os.makedirs(d, exist_ok=True)
subprocess.run(["unzip", "-qo", zip_path, "-d", d], check=True)

sys.path.insert(0, d)
try:
    import submission as M
    print("SUCCESS: submission.py imported cleanly.")
    if hasattr(M, "predict"):
        print("SUCCESS: M.predict exists.")
    else:
        print("FAIL: M.predict missing.")
except Exception as e:
    print(f"FAIL: Failed to import submission: {e}")
