import subprocess
import os

zips = [
    "submission_SOUP_v1.zip",
    "submission_SV2.zip",
    "submission_ASYM_W96_a85.zip",
    "submission_ASYM_W96_v3_a85.zip",
    "submission_e2e_sv3_final.zip"
]

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP/submissions"

for z in zips:
    if os.path.exists(f"{B}/{z}"):
        print(f"Running gate23 on {z}...")
        try:
            r = subprocess.run(["python3", "/SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es/gate23_es.py", z], capture_output=True, text=True)
            for line in r.stdout.split('\n'):
                if "h_u ratio med" in line or "time  best of 3" in line or "max |pred diff|" in line:
                    print("  " + line.strip())
        except Exception as e:
            print(f"Error on {z}: {e}")
    else:
        print(f"File {z} not found.")

