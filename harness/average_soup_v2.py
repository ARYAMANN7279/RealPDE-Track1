import torch, sys
paths = [
    'ft_all_w15_lr1_final.pth',
    'ft_all_w15_lr3_final.pth',
    'ft_all_w33_lr1_final.pth',
    'ft_all_w33_lr3_final.pth'
]
print('Loading models...')
sds = [torch.load(p, map_location='cpu') for p in paths]
soup_sd = {}
for k in sds[0].keys():
    soup_sd[k] = sum(sd[k] for sd in sds) / len(sds)

torch.save(soup_sd, 'soup_v2.pth')
print('soup_v2.pth saved!')
