import torch
import sys

ckpt = torch.load('/SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es/e2e_sv3_v2.pth', map_location='cpu')
fno_sd = ckpt['fno']
unet_sd = ckpt['unet']

def to_fp16(sd):
    new_sd = {}
    for k, v in sd.items():
        if v.dtype == torch.complex64:
            new_sd[k] = v
        else:
            new_sd[k] = v.half()
    return new_sd

torch.save(to_fp16(fno_sd), '/SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es/fno_sv3_e2e_fp16.pth')
torch.save(to_fp16(unet_sd), '/SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es/unet_sv3_e2e_fp16.pth')
print("Successfully split and converted to FP16!")
