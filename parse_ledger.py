import re

text = open('/Users/aryamannsrivastava/Desktop/sem7/UGP/project_memory.md', 'r').read()

zips = ["submission_SOUP_v1.zip", "submission_LUTFIX.zip", "submission_WIDE125.zip", 
        "submission_ASYM_W96_safe_arcsinh.zip", "submission_SHIFT_v2_W96_a85.zip", 
        "submission_ENSEMBLE_FAST_v3.zip", "submission_TMEAN.zip", "submission_LUTCAL.zip", 
        "submission_FP16.zip", "submission_SV2.zip", "submission_SCREEN.zip", 
        "submission_BLIND.zip", "submission_W73.zip"]

print("--- ZIP MENTIONS ---")
for z in zips:
    print(f"\n{z}:")
    for i, line in enumerate(text.split('\n')):
        if z in line:
            # print surrounding lines
            start = max(0, i-2)
            end = min(len(text.split('\n')), i+3)
            print(f"L{i}: " + "\n".join(text.split('\n')[start:end]))
            print("-" * 20)
