#!/bin/bash
cd /SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es

echo "Starting GPU 0 jobs..."
CUDA_VISIBLE_DEVICES=0 python3 ftmv.py --lr 1e-5 --wd 1e-6 --steps 16000 --wtke 0.08 --aug none --seed 101 --tag mem1_1e5_08 --gpu 0 > mem1.log 2>&1 &
PID0=$!

echo "Starting GPU 1 jobs..."
CUDA_VISIBLE_DEVICES=1 python3 ftmv.py --lr 1e-5 --wd 1e-6 --steps 16000 --wtke 0.05 --aug none --seed 102 --tag mem2_1e5_05 --gpu 0 > mem2.log 2>&1 &
PID1=$!

echo "Starting GPU 2 jobs..."
CUDA_VISIBLE_DEVICES=2 python3 ftmv.py --lr 1e-5 --wd 1e-6 --steps 16000 --wtke 0.10 --aug none --seed 103 --tag mem3_1e5_10 --gpu 0 > mem3.log 2>&1 &
PID2=$!

echo "Starting GPU 3 jobs..."
CUDA_VISIBLE_DEVICES=3 python3 ftmv.py --lr 3e-5 --wd 1e-6 --steps 16000 --wtke 0.08 --aug none --seed 104 --tag mem4_3e5_08 --gpu 0 > mem4.log 2>&1 &
PID3=$!

wait $PID0
CUDA_VISIBLE_DEVICES=0 python3 ftmv.py --lr 3e-5 --wd 1e-6 --steps 16000 --wtke 0.05 --aug none --seed 105 --tag mem5_3e5_05 --gpu 0 > mem5.log 2>&1 &

wait $PID1
CUDA_VISIBLE_DEVICES=1 python3 ftmv.py --lr 3e-5 --wd 1e-6 --steps 16000 --wtke 0.10 --aug none --seed 106 --tag mem6_3e5_10 --gpu 0 > mem6.log 2>&1 &

wait $PID2
CUDA_VISIBLE_DEVICES=2 python3 ftmv.py --lr 1e-5 --wd 1e-6 --steps 16000 --wtke 0.08 --aug none --seed 107 --tag mem7_1e5_08 --gpu 0 > mem7.log 2>&1 &

wait
echo "All 7 models finished!"
