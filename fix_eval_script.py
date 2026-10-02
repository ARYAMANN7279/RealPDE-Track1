with open("/SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es/eval_anchors_fast.py", "r") as f:
    lines = f.read()
import sys
lines = lines.replace(
    "zips = [", 
    "import sys\nzips = sys.argv[1:] if len(sys.argv) > 1 else ["
)
# Wait, if zips = sys.argv[1:], it expects zipnames or paths?
# The script does: zipname = ...
# f"{B}/submissions/{zipname}"
# So we just pass the zip name (e.g., "submission_DISTILLED_v3.zip")

lines = lines.replace(
    'f"{B}/submissions/{zipname}"',
    'f"{B}/submissions/{zipname}" if not "/" in zipname else zipname'
)
# But wait, it also does: d_dir = f"{B}/_tmp/eval_{zipname}"
# If zipname contains a path, it will crash mkdir.
# So let's just make sure we pass the basename!

lines = lines.replace(
    'd_dir = f"{B}/_tmp/eval_{zipname}"',
    'd_dir = f"{B}/_tmp/eval_{os.path.basename(zipname)}"'
)
with open("/SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es/eval_anchors_fast2.py", "w") as f:
    f.write(lines)
