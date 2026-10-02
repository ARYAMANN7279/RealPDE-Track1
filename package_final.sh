#!/bin/bash
set -ex
cd /SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es
PY=/SML_DISK_24TB/rajeshr/Aryamann/env/bin/python

# 1. Repackage UNet
$PY -c '
import torch
d = torch.load("e2e_het_final_best.pth")
torch.save({"sd": d["unet"], "w": 96, "mu": -6.0, "sd_": 1.5}, "joint_het.pth")
'

# 2. Make cache for w15lr3 (if not exists, actually just make it)
$PY mkcache2.py --ckpt ../local_harness/ft_long_w15lr3_best.pth --tag w15lr3 || true

# 3. Build shift
$PY build_shift.py --joint joint_het.pth --ckpt w15lr3 --cache cache_w15lr3.npz --alpha 0.0 --tag het_submission
