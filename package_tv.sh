#!/bin/bash
set -ex
cd /SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es
PY=/SML_DISK_24TB/rajeshr/Aryamann/env/bin/python

# 1. Unpack FNO and UNet
$PY << 'PYEOF'
import torch
d = torch.load("e2e_tv_best.pth", map_location="cpu")
torch.save(d["fno"], "e2e_tv_fno.pth")
torch.save({"sd": d["unet"], "w": 64, "mu": -6.0, "sd_": 1.5}, "e2e_tv_unet.pth")
PYEOF

# 2. Pack FNO to FP16
$PY ../starting_kit_v9/realpde_t1_starting_kit_v9/pack_ckpt_fp16.py e2e_tv_fno.pth e2e_tv_fno_fp16.pth

# 3. Cache
$PY mkcache2.py --ckpt e2e_tv_fno.pth --tag e2e_tv

# 4. Build Shift using custom builder
$PY build_tv_custom.py --unet e2e_tv_unet.pth --fno_fp16 e2e_tv_fno_fp16.pth --cache cache_e2e_tv.npz --alpha 0.75 --tag tv_final
