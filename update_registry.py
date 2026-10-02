import os, json

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
reg_path = f"{B}/CHECKPOINT_REGISTRY.md"

if not os.path.exists(reg_path):
    print("Registry not found!")
    exit(1)

lines = open(reg_path).read().split('\n')
header_idx = -1
for i, l in enumerate(lines):
    if "| file | bytes |" in l:
        header_idx = i
        break

if header_idx == -1:
    print("Table not found!")
    exit(1)

table_lines = lines[header_idx+2:]
parsed = []
for l in table_lines:
    if not l.strip(): continue
    parts = [p.strip() for p in l.split('|')][1:-1]
    if len(parts) >= 12:
        parsed.append(parts)

def get_sidecar(name):
    base = name
    name_noext = os.path.splitext(base)[0]
    name_noext2 = name_noext.replace('_best', '').replace('_final', '').replace('ftaug_', '')
    for d in [f"{B}/train_es", f"{B}/local_harness", f"{B}/data/comp_real", B]:
        candidates = [
            f"{d}/ftaug_{name_noext}.json",
            f"{d}/{name_noext}_result.json",
            f"{d}/ftaug_{name_noext2}.json",
            f"{d}/{name_noext2}_result.json",
        ]
        if name_noext.startswith("r13_"):
            candidates.append(f"{d}/{name_noext}.json")
        for c in candidates:
            if os.path.exists(c):
                try: return json.load(open(c))
                except: pass
    return None

def match_base(base, target):
    if len(base) != 3: return False
    return all(abs(float(base[i]) - float(target[i])) <= 0.01 for i in range(3))

moved = []

for row in parsed:
    fname = row[0]
    leakage = row[11].replace('**', '')
    scored = row[12]
    
    sc = get_sidecar(fname)
    base = None
    if sc and "base" in sc:
        base = sc["base"]
    
    if leakage == "UNKNOWN":
        old_leak = leakage
        if base:
            if match_base(base, [95.4738, 75.8957, 96.0945]):
                init_root = row[4]
                if init_root.endswith("KIT") or init_root == "KIT":
                    leakage = "HONEST"
                elif "SOUP" in init_root:
                    leakage = "INIT-LEAK"
                elif "EVERY5" in init_root:
                    leakage = "EVERY5"
                elif "ALL-DATA" in init_root:
                    leakage = "ALL-DATA"
                else:
                    leakage = "HONEST"
            elif match_base(base, [96.325, 75.264, 96.861]):
                leakage = "EVERY5"
            elif match_base(base, [95.781, 77.026, 96.14]):
                leakage = "EVERY9"
        
        if leakage != old_leak:
            moved.append((fname, leakage, str(base)))
                
    if fname == "ftaug_sv3_3e5_10.pth":
        scored = "member of submission_SCREEN.zip"
    if fname == "soup_v2_fp16.pth":
        scored = "member of submission_SV2.zip and submission_SCREEN.zip"
    if fname == "ft_long_w15lr3_best.pth":
        scored = "member of submission_LONG80.zip"
    if fname in ["ftaug_r29_ema100_d999.pth", "ftaug_r29_ema100_d9999.pth"]:
        scored = "member of submission_BLIND.zip"
        
    row[11] = f"**{leakage}**"
    row[12] = scored

for row in parsed:
    leak = row[11].replace('**', '')
    usable = "UNKNOWN"
    
    if leak in ["EVERY5", "ALL-DATA", "EVERY9"]:
        usable = "YES"
    elif leak == "INIT-LEAK" or "ftaug_r24_a100_" in row[0] or "ftaug_r29_ema100_" in row[0]:
        usable = "NO"
    elif leak == "HONEST":
        usable = "HONEST-ONLY"
        
    if len(row) == 13:
        row.append(usable)
    else:
        row[13] = usable

md = "# CHECKPOINT PROVENANCE REGISTRY\n\n"
md += "## Summary Counts\n"
counts = {}
for row in parsed:
    lk = row[11].replace('**', '')
    counts[lk] = counts.get(lk, 0) + 1
for k, v in counts.items():
    md += f"- **{k}**: {v}\n"

unknown_remain = counts.get("UNKNOWN", 0)

md += "\n## HONEST Checkpoints\n"
for row in parsed:
    if row[11].replace('**', '') == "HONEST":
        md += f"- `{row[0]}`\n"
        
md += "\n## INIT-LEAK Checkpoints (Dangerous)\n"
for row in parsed:
    if row[11].replace('**', '') == "INIT-LEAK":
        md += f"- `{row[0]}`\n"

md += "\n## Registry\n\n"
md += "| file | bytes | split | init (raw) | init ROOT | steps | lr | wtke | ema | aug | seed | LEAKAGE | in a scored artifact? | USABLE AS BLEND PARTNER? |\n"
md += "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"

for row in parsed:
    md += "| " + " | ".join(row) + " |\n"

open(reg_path, "w").write(md)

print("MOVED:")
for m in moved:
    print(f"{m[0]}: moved to {m[1]} via base {m[2]}")
print(f"UNKNOWN_REMAIN: {unknown_remain}")
