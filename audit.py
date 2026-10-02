import os, subprocess, glob

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
subs_dir = f"{B}/submissions"
scratch = f"{B}/scratch"
os.makedirs(scratch, exist_ok=True)

targets = ["submission_W73.zip", "submission_CORNER3.zip", "submission_LONG80.zip"]
expected_zip_md5 = {
    "submission_W73.zip": "3bdf77b97c3ef5b5c5c0e79865982d98",
    "submission_CORNER3.zip": "7b6b125dfe27d7453ca82e4753530ea8",
    "submission_LONG80.zip": "6bae48b8b5ba1478614baf95fa10c6fc"
}
expected_pth_md5 = {
    "submission_W73.zip": "8d112b78550e87fa91241c15af8eae84",
    "submission_CORNER3.zip": "fd4a4231a7402982d06c3917b96e05c8",
    "submission_LONG80.zip": "43c7823d7004c9a4eb4f895e82d5be6a"
}

def get_md5(cmd):
    try: return subprocess.check_output(cmd, shell=True).decode().strip().split()[0]
    except: return ""

print("--- 3.1 Fingerprints ---")
for t in targets:
    zpath = f"{subs_dir}/{t}"
    if not os.path.exists(zpath):
        print(f"MISSING: {t}")
        continue
    zmd5 = get_md5(f"md5sum {zpath}")
    pmd5 = get_md5(f"unzip -p {zpath} sim_real_fno_fp16.pth 2>/dev/null | md5sum")
    print(f"{t}: Zip: {zmd5} (Expected: {expected_zip_md5[t]} {'[MATCH]' if zmd5 == expected_zip_md5[t] else '[MISMATCH]'})")
    print(f"  Pth: {pmd5} (Expected: {expected_pth_md5[t]} {'[MATCH]' if pmd5 == expected_pth_md5[t] else '[MISMATCH]'})")

print("\n--- 3.2 Duplicate Check (CRC32-accelerated) ---")
all_zips = glob.glob(f"{subs_dir}/*.zip")
w73_path = f"{subs_dir}/submission_W73.zip"

def get_crcs(zpath):
    try:
        out = subprocess.check_output(f"unzip -v {zpath}", shell=True).decode()
        crcs = {}
        for line in out.strip().split('\n')[3:-2]:
            parts = line.split()
            if len(parts) >= 8:
                crcs[parts[-1]] = parts[6] # The CRC-32 column
        return crcs
    except: return {}

w73_crcs = get_crcs(w73_path)
w73_crc_pth = w73_crcs.get("sim_real_fno_fp16.pth", "")
w73_crc_bnd = w73_crcs.get("bounds_assets.npz", "")
w73_crc_sub = w73_crcs.get("submission.py", "")

w73_md5_pth = expected_pth_md5["submission_W73.zip"]
w73_md5_bnd = get_md5(f"unzip -p {w73_path} bounds_assets.npz 2>/dev/null | md5sum")
w73_md5_sub = get_md5(f"unzip -p {w73_path} submission.py 2>/dev/null | md5sum")

dup_pth_list = []
dup_all_list = []

for z in all_zips:
    name = os.path.basename(z)
    if name == "submission_W73.zip": continue
    
    crcs = get_crcs(z)
    if crcs.get("sim_real_fno_fp16.pth", "") == w73_crc_pth:
        # CRC matches, verify with MD5
        m_pth = get_md5(f"unzip -p {z} sim_real_fno_fp16.pth 2>/dev/null | md5sum")
        if m_pth == w73_md5_pth:
            dup_pth_list.append(name)
            
            # Check all 3
            if crcs.get("bounds_assets.npz", "") == w73_crc_bnd and crcs.get("submission.py", "") == w73_crc_sub:
                m_bnd = get_md5(f"unzip -p {z} bounds_assets.npz 2>/dev/null | md5sum")
                m_sub = get_md5(f"unzip -p {z} submission.py 2>/dev/null | md5sum")
                if m_bnd == w73_md5_bnd and m_sub == w73_md5_sub:
                    dup_all_list.append(name)

print(f"Shared backbone md5 with W73: {dup_pth_list}")
print(f"Shared all 3 md5s with W73: {dup_all_list}")

def get_entries(zpath):
    try:
        out = subprocess.check_output(f"unzip -l {zpath}", shell=True).decode()
        lines = out.strip().split('\n')
        entries = []
        size = 0
        for line in lines[3:-2]:
            parts = line.split()
            if len(parts) >= 4:
                entries.append(parts[-1])
                try: size += int(parts[0])
                except: pass
        return entries, size
    except: return [], 0

sv2_entries, _ = get_entries(f"{subs_dir}/submission_SV2.zip")
sv2_crcs = get_crcs(f"{subs_dir}/submission_SV2.zip")

print("\n--- 3.3 Structural Verification ---")
results = {}
for t in targets:
    print(f"\nChecking {t}...")
    zpath = f"{subs_dir}/{t}"
    if not os.path.exists(zpath):
        results[t] = "MISSING"
        continue
    
    try:
        t_out = subprocess.check_output(f"unzip -t {zpath}", shell=True, stderr=subprocess.STDOUT).decode()
        if "No errors detected" in t_out: t_res = "clean"
        else: t_res = "errors"
    except Exception as e: t_res = f"error"
        
    entries, tot_size = get_entries(zpath)
    entry_match = (set(entries) == set(sv2_entries))
    pyc_count = sum(1 for x in entries if x.endswith(".pyc"))
    
    cap = 268435456
    size_pct = tot_size / cap * 100
    
    npz_keys = 0
    os.system(f"unzip -p {zpath} bounds_assets.npz > {scratch}/x.npz 2>/dev/null")
    try:
        py_cmd = f"import numpy; print(len(numpy.load('{scratch}/x.npz').files))"
        k_out = subprocess.check_output([f"/SML_DISK_24TB/rajeshr/Aryamann/env/bin/python", "-c", py_cmd]).decode().strip()
        npz_keys = int(k_out)
    except: npz_keys = -1
        
    identical = True
    bad_files = []
    my_crcs = get_crcs(zpath)
    for e in entries:
        if e != "sim_real_fno_fp16.pth" and not e.endswith('/') and e in sv2_crcs:
            if my_crcs.get(e, "") != sv2_crcs.get(e, ""):
                m1 = get_md5(f"unzip -p {zpath} {e} 2>/dev/null | md5sum")
                m2 = get_md5(f"unzip -p {subs_dir}/submission_SV2.zip {e} 2>/dev/null | md5sum")
                if m1 != m2:
                    identical = False
                    bad_files.append(e)
                
    fingerprint_match = get_md5(f"md5sum {zpath}") == expected_zip_md5[t] and get_md5(f"unzip -p {zpath} sim_real_fno_fp16.pth 2>/dev/null | md5sum") == expected_pth_md5[t]
    
    go = True
    fail_reasons = []
    if not fingerprint_match: go=False; fail_reasons.append("Fingerprint mismatch")
    if t_res != "clean": go=False; fail_reasons.append(f"unzip -t {t_res}")
    if not entry_match: go=False; fail_reasons.append("Entry list mismatch vs SV2")
    if pyc_count != 0: go=False; fail_reasons.append(f"{pyc_count} .pyc files found")
    if tot_size >= cap: go=False; fail_reasons.append("Extracted size exceeds cap")
    if npz_keys != 186: go=False; fail_reasons.append(f"Bounds keys={npz_keys} (!=186)")
    if t == "submission_W73.zip" and dup_pth_list: go=False; fail_reasons.append(f"Backbone shared with {dup_pth_list}")
    if not identical: go=False; fail_reasons.append(f"Files not identical to SV2: {bad_files}")
    
    results[t] = "GO" if go else f"NO-GO: {', '.join(fail_reasons)}"
    
    print(f"  unzip -t: {t_res}")
    print(f"  Entry list == SV2: {entry_match}")
    print(f"  .pyc count: {pyc_count}")
    print(f"  Extracted size: {tot_size} B ({size_pct:.2f}%)")
    print(f"  Bounds keys: {npz_keys}")
    print(f"  Identical to SV2: {identical}")
    print(f"  Status: {results[t]}")

print("\n--- GO/NO-GO SUMMARY ---")
for t, st in results.items():
    print(f"{t}: {st}")
