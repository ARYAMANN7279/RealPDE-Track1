#!/bin/bash
cd /SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es

echo "Starting GPU 0 jobs..."
CUDA_VISIBLE_DEVICES=0 python3 ftmv.py --split none --aug phase --wtke 0.08 --lr 1e-5 --seed 301 --tag sv3_proper_m1 --gpu 0 > sv3_proper_m1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 python3 ftmv.py --split none --aug phase --wtke 0.08 --lr 1e-5 --seed 305 --tag sv3_proper_m5 --gpu 0 > sv3_proper_m5.log 2>&1 &

echo "Starting GPU 2 jobs..."
CUDA_VISIBLE_DEVICES=2 python3 ftmv.py --split none --aug phase --wtke 0.08 --lr 1e-5 --seed 303 --tag sv3_proper_m3 --gpu 0 > sv3_proper_m3.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python3 ftmv.py --split none --aug phase --wtke 0.08 --lr 1e-5 --seed 306 --tag sv3_proper_m6 --gpu 0 > sv3_proper_m6.log 2>&1 &

wait
echo "SV3 proper completed!"
