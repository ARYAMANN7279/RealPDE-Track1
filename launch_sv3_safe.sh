#!/bin/bash
(CUDA_VISIBLE_DEVICES=0 python3 ftmv.py --split none --wtke 0.08 --lr 5e-4 --seed 42 --tag sv3_m1 > sv3_mem1.log 2>&1; CUDA_VISIBLE_DEVICES=0 python3 ftmv.py --split none --wtke 0.08 --lr 5e-4 --seed 46 --tag sv3_m5 > sv3_mem5.log 2>&1) &
(CUDA_VISIBLE_DEVICES=1 python3 ftmv.py --split none --wtke 0.08 --lr 5e-4 --seed 43 --tag sv3_m2 > sv3_mem2.log 2>&1; CUDA_VISIBLE_DEVICES=1 python3 ftmv.py --split none --wtke 0.08 --lr 5e-4 --seed 47 --tag sv3_m6 > sv3_mem6.log 2>&1) &
(CUDA_VISIBLE_DEVICES=2 python3 ftmv.py --split none --wtke 0.08 --lr 5e-4 --seed 44 --tag sv3_m3 > sv3_mem3.log 2>&1; CUDA_VISIBLE_DEVICES=2 python3 ftmv.py --split none --wtke 0.12 --lr 5e-4 --seed 48 --tag sv3_m7 > sv3_mem7.log 2>&1) &
(CUDA_VISIBLE_DEVICES=3 python3 ftmv.py --split none --wtke 0.08 --lr 5e-4 --seed 45 --tag sv3_m4 > sv3_mem4.log 2>&1) &
wait
