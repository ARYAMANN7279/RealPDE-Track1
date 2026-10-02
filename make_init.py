import torch
import sys

DEV = "cpu"
sd = torch.load("/SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es/soup_fno_fp16.pth", map_location=DEV)
if 'state_fp16' in sd:
    ck = set(sd.get('complex_keys', []))
    sd_fp16 = sd['state_fp16']
    for k, v in sd_fp16.items():
        if hasattr(v, 'float'):
            v = v.float()
        if k in ck:
            v = torch.view_as_complex(v)
        sd_fp16[k] = v
    sd = sd_fp16

torch.save(sd, "/SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es/init_sv4.pth")
print("Saved init_sv4.pth!")
