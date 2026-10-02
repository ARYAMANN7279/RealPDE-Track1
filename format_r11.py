import json
import os

with open("/Users/aryamannsrivastava/.gemini/antigravity/brain/c91671e0-0fad-4525-887c-4ecb73348b55/.system_generated/tasks/task-10527.log") as f:
    lines = f.readlines()

json_str = ""
in_json = False
for line in lines:
    if line.startswith("{"):
        in_json = True
    if in_json:
        json_str += line
    if in_json and line.startswith("}"):
        break

try:
    data = json.loads(json_str)
    
    out = "## 74. ROUND 11 RAW RESULTS (Gemini, executed)\n\n"
    out += "| LUT Variant | Eval Mode | SPS `g` | `R = g / g_shipped` | Coverage | Mean Width |\n"
    out += "|---|---|---|---|---|---|\n"
    
    eff = data["effects"]
    out += f"| Shipped W96 | Raw Local | {eff['shipped']['raw']['g']:.4f} | 1.0000 | - | - |\n"
    out += f"| Shipped W96 | b=1.8 Emulated | {eff['shipped']['emul']['g']:.4f} | 1.0000 | - | - |\n"
    
    for b in ["b1.0", "b1.4", "b1.8", "b2.2"]:
        row = eff[b]
        out += f"| {b} Refit | Raw Local | {row['raw']['g']:.4f} | {row['raw']['R']:.4f} | {row['coverage']:.4f} | {row['mean_h']:.4f} |\n"
        out += f"| {b} Refit | b=1.8 Emulated | {row['emul']['g']:.4f} | {row['emul']['R']:.4f} | - | - |\n"
        
    out += "\n**Gates:**\n"
    g1 = data["G1"]
    out += f"- **G1 (Majority Wider)**: `u={g1['u']:.4f}`, `v={g1['v']:.4f}`. Pass = `{g1['pass']}`. (b=1.8 narrower than b=1.0 on u)\n"
    g2 = data["G2"]
    out += f"- **G2 (Stability)**: `u={g2['u']:.4f}`, `v={g2['v']:.4f}`. Pass = `{g2['pass']}`.\n\n"
    out += "Conclusion: G1 FAILED! The theory that inflation maps to wider bins is falsified on the `u` channel. Refitting on the emulated scale actually produced narrower optimal bounds than refitting on raw local residuals. Consequently, the archive `submission_LUTEMU.zip` was **not** built.\n"
    
    print(out)
    
    with open("/Users/aryamannsrivastava/Desktop/sem7/UGP/project_memory.md", "a") as out_f:
        out_f.write("\n" + out + "\n")
        
    os.system("sshpass -p 'ACAL@2026' scp -o StrictHostKeyChecking=no /Users/aryamannsrivastava/Desktop/sem7/UGP/project_memory.md rajeshr@172.31.100.2:/SML_DISK_24TB/rajeshr/Aryamann/UGP/project_memory.md")

except Exception as e:
    print(e)
