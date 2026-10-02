import numpy as np
z = np.load('/SML_DISK_24TB/rajeshr/Aryamann/UGP/_tmp/eval_submission_SV2/bounds_assets.npz')
j = 0
while f'e{j}_w' in z.files:
    j += 1
print(f'Total extra ensemble nets: {j}')
print('Center weights:', z['cw'] if 'cw' in z.files else 'None')
