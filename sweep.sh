#!/bin/bash
cd /SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es
CUDA_VISIBLE_DEVICES=0 python3 ftmv.py --steps 8000 --aug phase --wtke 0.08 --lr 3e-5 --seed 201 --tag sweep_1 --gpu 0 > sweep_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 ftmv.py --steps 8000 --aug phase --wtke 0.08 --lr 1e-5 --seed 202 --tag sweep_2 --gpu 0 > sweep_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python3 ftmv.py --steps 8000 --aug phase --wtke 0.12 --lr 3e-5 --seed 203 --tag sweep_3 --gpu 0 > sweep_3.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python3 ftmv.py --steps 8000 --aug none  --wtke 0.08 --lr 3e-5 --seed 204 --tag sweep_4 --gpu 0 > sweep_4.log 2>&1 &
wait
echo "Sweep done"
