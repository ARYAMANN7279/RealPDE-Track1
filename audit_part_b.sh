B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
while [ ! -f $B/submissions/submission_RECIPE.zip ]; do
  sleep 10
done
echo "Found submission_RECIPE.zip"
/SML_DISK_24TB/rajeshr/Aryamann/env/bin/python -u /SML_DISK_24TB/rajeshr/Aryamann/UGP/audit_recipe.py
