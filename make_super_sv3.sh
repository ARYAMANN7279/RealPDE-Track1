#!/bin/bash
set -e
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
cd $B/submissions

echo "Extracting SV2..."
mkdir -p build_super_sv3
cd build_super_sv3
unzip -qo ../submission_SV2.zip

echo "Swapping FNO..."
cp $B/train_es/soup_sv3_fp16.pth sim_real_fno_fp16.pth

echo "Zipping SUPER_SV3..."
zip -q -0 -r ../submission_SUPER_SV3.zip ./*

echo "Done!"
