#!/bin/bash
cd /SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es

echo "Starting GPU 0 jobs..."
CUDA_VISIBLE_DEVICES=0 python3 ftmv.py --split re_lohi --aug phase --wtke 1.0 --lr 1e-5 --seed 401 --tag sv4_m1 --gpu 0 --init init_sv4.pth --steps 6000 --evalevery 1000 > sv4_m1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 ftmv.py --split re_lohi --aug phase --wtke 1.0 --lr 1e-5 --seed 402 --tag sv4_m2 --gpu 0 --init init_sv4.pth --steps 6000 --evalevery 1000 > sv4_m2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python3 ftmv.py --split re_lohi --aug phase --wtke 1.0 --lr 1e-5 --seed 403 --tag sv4_m3 --gpu 0 --init init_sv4.pth --steps 6000 --evalevery 1000 > sv4_m3.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python3 ftmv.py --split re_lohi --aug phase --wtke 1.0 --lr 1e-5 --seed 404 --tag sv4_m4 --gpu 0 --init init_sv4.pth --steps 6000 --evalevery 1000 > sv4_m4.log 2>&1 &

wait
echo "SV4 Phase-Augmented Models Completed!"
