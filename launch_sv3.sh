#!/bin/bash
cd /SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es

echo "Starting GPU 0 jobs..."
CUDA_VISIBLE_DEVICES=0 python3 ftmv.py --split none --lr 1e-5 --wd 1e-6 --steps 16000 --wtke 0.08 --aug none --seed 201 --tag sv3_1e5_08 --gpu 0 > sv3_mem1.log 2>&1 &
PID0=$!

echo "Starting GPU 1 jobs..."
CUDA_VISIBLE_DEVICES=1 python3 ftmv.py --split none --lr 1e-5 --wd 1e-6 --steps 16000 --wtke 0.05 --aug none --seed 202 --tag sv3_1e5_05 --gpu 0 > sv3_mem2.log 2>&1 &
PID1=$!

echo "Starting GPU 2 jobs..."
CUDA_VISIBLE_DEVICES=2 python3 ftmv.py --split none --lr 1e-5 --wd 1e-6 --steps 16000 --wtke 0.10 --aug none --seed 203 --tag sv3_1e5_10 --gpu 0 > sv3_mem3.log 2>&1 &
PID2=$!

echo "Starting GPU 3 jobs..."
CUDA_VISIBLE_DEVICES=3 python3 ftmv.py --split none --lr 3e-5 --wd 1e-6 --steps 16000 --wtke 0.08 --aug none --seed 204 --tag sv3_3e5_08 --gpu 0 > sv3_mem4.log 2>&1 &
PID3=$!

wait $PID0
CUDA_VISIBLE_DEVICES=0 python3 ftmv.py --split none --lr 3e-5 --wd 1e-6 --steps 16000 --wtke 0.05 --aug none --seed 205 --tag sv3_3e5_05 --gpu 0 > sv3_mem5.log 2>&1 &

wait $PID1
CUDA_VISIBLE_DEVICES=1 python3 ftmv.py --split none --lr 3e-5 --wd 1e-6 --steps 16000 --wtke 0.10 --aug none --seed 206 --tag sv3_3e5_10 --gpu 0 > sv3_mem6.log 2>&1 &

wait $PID2
CUDA_VISIBLE_DEVICES=2 python3 ftmv.py --split none --lr 1e-5 --wd 1e-6 --steps 16000 --wtke 0.08 --aug none --seed 207 --tag sv3_1e5_08_b --gpu 0 > sv3_mem7.log 2>&1 &

wait
echo "All 7 SV3 models finished!"
