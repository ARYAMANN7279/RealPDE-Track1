#!/bin/bash
# Wait for the current job on GPU 0 to finish
while nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; do
    sleep 60
done

# Job 2: Arm A seed 1
echo "Starting Arm A Seed 1"
env CUDA_VISIBLE_DEVICES=0 /SML_DISK_24TB/rajeshr/Aryamann/env/bin/python r13_train.py --tag r13_control_s1 --split re_lohi --aug none --lr 3e-5 --wtke 0.15 --bs 16 --steps 6000 --evalevery 500 --seed 1 > train_es/r13_control_s1.log 2>&1

# Job 3: Arm E seed 0
echo "Starting Arm E Seed 0"
env CUDA_VISIBLE_DEVICES=0 /SML_DISK_24TB/rajeshr/Aryamann/env/bin/python r13_train.py --tag r13_wd1e3_s0 --split re_lohi --aug none --lr 3e-5 --wtke 0.15 --bs 16 --steps 6000 --evalevery 500 --seed 0 --wd 1e-3 > train_es/r13_wd1e3_s0.log 2>&1

# Job 4: Arm E seed 1
echo "Starting Arm E Seed 1"
env CUDA_VISIBLE_DEVICES=0 /SML_DISK_24TB/rajeshr/Aryamann/env/bin/python r13_train.py --tag r13_wd1e3_s1 --split re_lohi --aug none --lr 3e-5 --wtke 0.15 --bs 16 --steps 6000 --evalevery 500 --seed 1 --wd 1e-3 > train_es/r13_wd1e3_s1.log 2>&1

# Job 5: Arm C seed 0
echo "Starting Arm C Seed 0"
env CUDA_VISIBLE_DEVICES=0 /SML_DISK_24TB/rajeshr/Aryamann/env/bin/python r13_train.py --tag r13_ema9999_s0 --split re_lohi --aug none --lr 3e-5 --wtke 0.15 --bs 16 --steps 6000 --evalevery 500 --seed 0 --ema 0.9999 > train_es/r13_ema9999_s0.log 2>&1

# Job 6: Arm C seed 1
echo "Starting Arm C Seed 1"
env CUDA_VISIBLE_DEVICES=0 /SML_DISK_24TB/rajeshr/Aryamann/env/bin/python r13_train.py --tag r13_ema9999_s1 --split re_lohi --aug none --lr 3e-5 --wtke 0.15 --bs 16 --steps 6000 --evalevery 500 --seed 1 --ema 0.9999 > train_es/r13_ema9999_s1.log 2>&1

echo "Queue finished!"
