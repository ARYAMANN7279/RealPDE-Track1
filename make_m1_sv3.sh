#!/bin/bash
set -e
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
cd $B/submissions

echo "Swapping to m1 FNO..."
rm -rf build_super_sv3
mkdir -p build_super_sv3
cd build_super_sv3
unzip -qo ../submission_SV2.zip

# convert m1 to fp16 packed
python3 $B/train_es/to_fp16.py $B/train_es/ftaug_sv3_proper_m1.pth m1_fp16.pth
cp m1_fp16.pth sim_real_fno_fp16.pth

echo "Zipping M1_SV3..."
zip -q -0 -r ../submission_M1_SV3.zip ./*
echo "Done!"
