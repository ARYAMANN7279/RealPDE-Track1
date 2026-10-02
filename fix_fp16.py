import torch
import sys

ckpt = torch.load('/SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es/e2e_sv3_v2.pth', map_location='cpu')
fno_sd = ckpt['fno']
unet_sd = ckpt['unet']

def pack_fno(sd):
    state = {}
    complex_keys = []
    for k, v in sd.items():
        if v.is_complex():
            state[k] = torch.view_as_real(v).half()
            complex_keys.append(k)
        else:
            state[k] = v.half()
    return {"state_fp16": state, "complex_keys": complex_keys}

def pack_unet(sd):
    return {k: v.half() for k, v in sd.items()}

torch.save(pack_fno(fno_sd), '/SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es/fno_sv3_e2e_fp16_packed.pth')
torch.save(pack_unet(unet_sd), '/SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es/unet_sv3_e2e_fp16_packed.pth')
print("Successfully packed as FP16 with complex_keys!")
