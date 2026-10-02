import torch
import os

paths = [
    '/SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es/ftaug_sv3_proper_m1.pth',
    '/SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es/ftaug_sv3_proper_m3.pth',
    '/SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es/ftaug_sv3_proper_m5.pth',
    '/SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es/ftaug_sv3_proper_m6.pth'
]

print('Loading SV3 models...')
sds = [torch.load(p, map_location='cpu') for p in paths]
soup_sd = {}
for k in sds[0].keys():
    soup_sd[k] = sum(sd[k] for sd in sds) / len(sds)

torch.save(soup_sd, '/SML_DISK_24TB/rajeshr/Aryamann/UGP/train_es/soup_sv3.pth')
print('soup_sv3.pth saved successfully!')
