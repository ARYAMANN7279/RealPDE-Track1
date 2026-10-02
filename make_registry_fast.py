import os
import json
import glob
import subprocess
import hashlib

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"

pth_files = []
for d in [f"{B}/train_es", f"{B}/local_harness", f"{B}/data/comp_real"]:
    pth_files.extend(glob.glob(f"{d}/*.pth"))

pth_files = [f for f in pth_files if os.path.getsize(f) > 3e8]

def get_sidecar(pth_file):
    base = os.path.basename(pth_file)
    dir_name = os.path.dirname(pth_file)
    name = os.path.splitext(base)[0]
    
    candidates = [
        f"{dir_name}/ftaug_{name}.json",
        f"{dir_name}/{name}_result.json",
        f"{dir_name}/ftaug_{name.replace('ftaug_', '')}.json",
        f"{dir_name}/{name.replace('ftaug_', '')}_result.json",
    ]
    if name.startswith("r13_"):
        candidates.append(f"{dir_name}/{name}.json")
    
    for c in candidates:
        if os.path.exists(c):
            try:
                return json.load(open(c))
            except:
                pass
    return None

def get_args(sidecar):
    if sidecar is None:
        return {}
    if "args" in sidecar:
        if isinstance(sidecar["args"], dict):
            return sidecar["args"]
        elif isinstance(sidecar["args"], str):
            try:
                return json.loads(sidecar["args"].replace("'", '"'))
            except:
                return {"_raw": sidecar["args"]}
    return sidecar

scripts = glob.glob(f"{B}/*.py") + glob.glob(f"{B}/local_harness/*.py")
script_content = {}
for s in scripts:
    with open(s, "r") as f:
        script_content[s] = f.read()

def find_script_source(filename):
    for s, content in script_content.items():
        if filename in content:
            if "make_init.py" in s and filename == "init_sv4.pth":
                return "soup_fno_fp16.pth"
    return None

data = []
basename_to_path = {os.path.basename(p): p for p in pth_files}
known_soups = ["soup_fno_fp16.pth", "soup_v1.pth", "soup_v2.pth", "soup_v3.pth", "soup_v4.pth"]

def trace_init(init_str):
    if not init_str or init_str == "kit":
        return "KIT"
    
    chain = [init_str]
    curr = init_str
    visited = set([curr])
    
    while True:
        is_soup = False
        for s in known_soups:
            if s in curr:
                chain.append(f"SOUP({s})")
                is_soup = True
                break
        if is_soup:
            break
            
        script_src = find_script_source(curr)
        if script_src:
            curr = script_src
            chain.append(curr)
            continue
            
        curr_path = basename_to_path.get(curr)
        if not curr_path:
            found = False
            for d in [f"{B}/train_es", f"{B}/local_harness", f"{B}/data/comp_real", B]:
                if os.path.exists(f"{d}/{curr}"):
                    curr_path = f"{d}/{curr}"
                    found = True
                    break
            if not found:
                chain.append("UNKNOWN")
                break
                
        side = get_sidecar(curr_path)
        if side:
            a = get_args(side)
            if not isinstance(a, dict): a = {}
            nxt = a.get("init", "")
            if not nxt:
                chain.append("KIT")
                break
            else:
                nxt_base = os.path.basename(nxt)
                if nxt_base in visited:
                    chain.append("CYCLE")
                    break
                curr = nxt_base
                chain.append(curr)
                visited.add(curr)
        else:
            break
            
    return " -> ".join(chain)

subs = glob.glob(f"{B}/submissions/submission_*.zip")
scored_artifacts = ["submission_SV2.zip", "submission_FP16.zip", "submission_TMEAN.zip", 
                   "submission_LUTCAL.zip", "submission_SOUP_v1.zip", "submission_SCREEN.zip", 
                   "submission_BLIND.zip", "submission_LONG80.zip"]

# Calculate fast md5 (first 1MB only)
def get_fast_md5_cmd(path):
    try:
        out = subprocess.check_output(f"head -c 1048576 {path} | md5sum", shell=True).decode().strip().split()[0]
        return out
    except:
        return ""

zip_md5s = {}
for sub in subs:
    sub_base = os.path.basename(sub)
    if sub_base in scored_artifacts:
        try:
            out = subprocess.check_output(f"unzip -p {sub} sim_real_fno_fp16.pth 2>/dev/null | head -c 1048576 | md5sum", shell=True).decode().strip().split()[0]
            if out and out != "d41d8cd98f00b204e9800998ecf8427e":
                zip_md5s[sub_base] = out
        except:
            pass

pth_md5s = {}
for p in pth_files:
    pth_md5s[p] = get_fast_md5_cmd(p)

for p in pth_files:
    fname = os.path.basename(p)
    size = os.path.getsize(p)
    
    sidecar = get_sidecar(p)
    args_dict = get_args(sidecar)
    if not isinstance(args_dict, dict):
        args_dict = {}
        
    split = args_dict.get("split", "UNKNOWN") if sidecar else "UNKNOWN"
    steps = args_dict.get("steps", "UNKNOWN") if sidecar else "UNKNOWN"
    lr = args_dict.get("lr", "UNKNOWN") if sidecar else "UNKNOWN"
    wtke = args_dict.get("wtke", "UNKNOWN") if sidecar else "UNKNOWN"
    ema = args_dict.get("ema", "UNKNOWN") if sidecar else "UNKNOWN"
    aug = args_dict.get("aug", "UNKNOWN") if sidecar else "UNKNOWN"
    seed = args_dict.get("seed", "UNKNOWN") if sidecar else "UNKNOWN"
    
    init_raw = args_dict.get("init", "kit") if sidecar else "UNKNOWN"
    if init_raw == "": init_raw = "kit"
    if init_raw is None: init_raw = "kit"
    
    if init_raw != "UNKNOWN":
        init_root = trace_init(os.path.basename(str(init_raw)) if init_raw != "kit" else "kit")
    else:
        init_root = "UNKNOWN"
        
    script_used_every5 = False
    if "soup_v3" in fname:
        script_used_every5 = True
        
    leakage = "UNKNOWN"
    is_all_data = str(split).lower() == "none" or str(split) == ""
    is_re_lohi = str(split).lower() == "re_lohi"
    is_every5 = "every5" in str(split).lower() or script_used_every5
    
    root_is_kit = init_root.endswith("KIT")
    root_is_soup = "SOUP" in init_root
    
    if is_re_lohi and root_is_kit:
        leakage = "HONEST"
    elif is_every5:
        leakage = "EVERY5"
    elif is_all_data:
        leakage = "ALL-DATA"
    elif is_re_lohi and root_is_soup:
        leakage = "INIT-LEAK"
    
    if sidecar is None and "soup" in fname.lower():
        leakage = "ALL-DATA"
    if sidecar is None and "ftaug_" not in fname and fname.startswith("sim_real"):
        leakage = "KIT" # presumably
    
    my_md5 = pth_md5s[p]
    in_scored = "no"
    for z, zmd5 in zip_md5s.items():
        if my_md5 == zmd5:
            in_scored = f"yes ({z})"
            break
            
    if in_scored == "no":
        for s in scripts:
            if fname in script_content[s]:
                if "sv4" in s:
                    in_scored = "member of sv4"
                    break
                if "sv3" in s:
                    in_scored = "member of sv3"
                    break
    
    data.append({
        "file": fname,
        "bytes": size,
        "split": split,
        "init": init_raw,
        "init_root": init_root,
        "steps": steps,
        "lr": lr,
        "wtke": wtke,
        "ema": ema,
        "aug": aug,
        "seed": seed,
        "leakage": leakage,
        "scored": in_scored
    })

md = "# CHECKPOINT PROVENANCE REGISTRY\n\n"

counts = {"HONEST": 0, "EVERY5": 0, "ALL-DATA": 0, "INIT-LEAK": 0, "UNKNOWN": 0, "KIT": 0}
for d in data:
    counts[d["leakage"]] = counts.get(d["leakage"], 0) + 1

md += "## Summary Counts\n"
for k, v in counts.items():
    if v > 0:
        md += f"- **{k}**: {v}\n"

md += "\n## HONEST Checkpoints\n"
for d in data:
    if d["leakage"] == "HONEST":
        md += f"- `{d['file']}`\n"

md += "\n## INIT-LEAK Checkpoints (Dangerous)\n"
for d in data:
    if d["leakage"] == "INIT-LEAK":
        md += f"- `{d['file']}`\n"

md += "\n## Registry\n\n"
md += "| file | bytes | split | init (raw) | init ROOT | steps | lr | wtke | ema | aug | seed | LEAKAGE | in a scored artifact? |\n"
md += "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"

for d in data:
    md += f"| {d['file']} | {d['bytes']} | {d['split']} | {d['init']} | {d['init_root']} | {d['steps']} | {d['lr']} | {d['wtke']} | {d['ema']} | {d['aug']} | {d['seed']} | **{d['leakage']}** | {d['scored']} |\n"

with open(f"{B}/CHECKPOINT_REGISTRY.md", "w") as f:
    f.write(md)
print("DONE")
