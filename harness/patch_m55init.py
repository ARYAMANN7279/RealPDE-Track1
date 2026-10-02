p = "/SML_DISK_24TB/rajeshr/Aryamann/UGP/local_harness/finetune_m55init.py"
s = open(p).read()
s = s.replace(
    'ap.add_argument("--holdout_stride",type=int,default=5)',
    'ap.add_argument("--holdout_stride",type=int,default=5)\n'
    'ap.add_argument("--init_ckpt",default="sim_real_fno.pth")'
)
s = s.replace(
    'model,_=load_baseline(f"{B}/data/comp_real/sim_real_fno.pth",device=DEV)',
    'model,_=load_baseline(f"{B}/data/comp_real/{a.init_ckpt}",device=DEV)'
)
open(p, "w").write(s)
print("patched:", "init_ckpt" in s, "{a.init_ckpt}" in s)
