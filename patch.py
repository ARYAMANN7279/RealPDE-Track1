import re
with open("submission.py", "r") as f:
    code = f.read()

code = re.sub(
    r"prediction = np\.empty\(\(n, 20\) \+ x\.shape\[2:\], dtype=np\.float32\)",
    "prediction_t = torch.empty((n, 20) + x.shape[2:], dtype=torch.float32, pin_memory=(device==\"cuda\"))\n    prediction = prediction_t.numpy()",
    code
)

code = re.sub(
    r"prediction\[i:i \+ _BATCH\] = yb\.float\(\)\.cpu\(\)\.numpy\(\)",
    "prediction_t[i:i + _BATCH].copy_(yb, non_blocking=True)",
    code
)

if "prediction_t" in code:
    code = code.replace(
        "if fallback_from is not None:",
        "if device == \"cuda\":\n        torch.cuda.synchronize()\n    if fallback_from is not None:"
    )
    with open("submission.py", "w") as f:
        f.write(code)
    print("Patched successfully!")
else:
    print("Patch failed!")
