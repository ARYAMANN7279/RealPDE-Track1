#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
nohup python3 $B/agents/moe/m13_train_lora_v6_parallel.py AoA_0 0 0 > $B/agents/moe/train_AoA_0.log 2>&1 &
nohup python3 $B/agents/moe/m13_train_lora_v6_parallel.py AoA_5 5 1 > $B/agents/moe/train_AoA_5.log 2>&1 &
nohup python3 $B/agents/moe/m13_train_lora_v6_parallel.py AoA_10 10 2 > $B/agents/moe/train_AoA_10.log 2>&1 &
nohup python3 $B/agents/moe/m13_train_lora_v6_parallel.py AoA_15 15 3 > $B/agents/moe/train_AoA_15.log 2>&1 &
nohup python3 $B/agents/moe/m13_train_lora_v6_parallel.py AoA_20 20 0 > $B/agents/moe/train_AoA_20.log 2>&1 &
echo "Launched 5 experts in parallel."
