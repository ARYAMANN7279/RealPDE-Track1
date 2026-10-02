p = "/SML_DISK_24TB/rajeshr/Aryamann/UGP/local_harness/finetune_moredata.py"
s = open(p).read()
s = s.replace(
    'ap.add_argument("--wtke",type=float,default=0.33); ap.add_argument("--tag",default="a")',
    'ap.add_argument("--wtke",type=float,default=0.33); ap.add_argument("--tag",default="a")\n'
    'ap.add_argument("--holdout_stride",type=int,default=5)'
)
s = s.replace(
    "vidx=set(range(0,ntraj,5))            # every 5th trajectory held out",
    "vidx=set(range(0,ntraj,a.holdout_stride))  # configurable holdout for more-data experiments"
)
open(p, "w").write(s)
print("patched")
