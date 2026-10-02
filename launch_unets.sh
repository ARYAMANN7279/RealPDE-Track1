#!/bin/bash
cd /SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es

echo "Starting 12 U-Net trainings on 4 GPUs..."
for i in {0..2}; do
    CUDA_VISIBLE_DEVICES=0 python3 joint.py --cache cache_soup.npz --width 96 --epochs 25 --tag unet_m${i}_0 --gpu 0 > unet_m${i}_0.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 python3 joint.py --cache cache_soup.npz --width 96 --epochs 25 --tag unet_m${i}_1 --gpu 0 > unet_m${i}_1.log 2>&1 &
    CUDA_VISIBLE_DEVICES=2 python3 joint.py --cache cache_soup.npz --width 96 --epochs 25 --tag unet_m${i}_2 --gpu 0 > unet_m${i}_2.log 2>&1 &
    CUDA_VISIBLE_DEVICES=3 python3 joint.py --cache cache_soup.npz --width 96 --epochs 25 --tag unet_m${i}_3 --gpu 0 > unet_m${i}_3.log 2>&1 &
    wait
done

echo "All U-Nets finished!"
