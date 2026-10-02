p = "/SML_DISK_24TB/rajeshr/Aryamann/UGP/train_mvpe/errnet2.py"
s = open(p).read()
bad = '''torch.save({"sd":net.state_dict(),"mode":a.mode,"w":a.width,"rich":a.rich,
                                 "mu":mu_,"sd":sd_,"taus":TAUS}, f"{B}/train_mvpe/runs/errnet_{a.tag}.pth"); flag=" *"'''
good = '''torch.save({"sd":net.state_dict(),"mode":a.mode,"w":a.width,"rich":a.rich,
                                 "pert":a.pert,"mu":mu_,"sd_":sd_,"taus":TAUS},
                                f"{B}/train_mvpe/runs/errnet_{a.tag}.pth"); flag=" *"'''
assert bad in s, "save block not found as expected"
open(p, "w").write(s.replace(bad, good))
print("FIXED: duplicate 'sd' key was overwriting net.state_dict() with a float scalar")
