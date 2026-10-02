#!/bin/bash
cd /SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es
echo "--- CPU BENCHMARK ---"
for i in {1..9}; do
    echo "Rep $i FP16"
    CUDA_VISIBLE_DEVICES="" python3 gate23_es.py ../submissions/submission_FP16.zip | grep -i "time"
    echo "Rep $i SV2"
    CUDA_VISIBLE_DEVICES="" python3 gate23_es.py ../submissions/submission_SV2.zip | grep -i "time"
done
