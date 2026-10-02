exec(open("disagreement.py").read().split("def quality")[0])
# Is a 3-member ENSEMBLE MEAN as accurate as the 9-member soup?
# If yes we need only 3 forward passes (prediction AND disagreement from the same
# runs). If no we must also run the soup = 4 passes, costing more time subscore.
def acc(P,m):
    dm=S.rel_l2_per_sample(P[m],Y[m],C); tk=S.tke_rel_l2_per_sample(P[m],Y[m],C); mv=S.mvpe_rel_l2_per_sample(P[m],Y[m])
    return (S.score_error(float(dm.mean())),S.score_error(float(tk.mean())),S.score_error(float(mv.mean())))
P3d=np.zeros_like(P_soup); 
idx3=np.linspace(0,len(PS)-1,3).astype(int)
P3d[...,:C]=PS[idx3].mean(axis=0)
P5d=np.zeros_like(P_soup); idx5=np.linspace(0,len(PS)-1,5).astype(int)
P5d[...,:C]=PS[idx5].mean(axis=0)
P9d=np.zeros_like(P_soup); P9d[...,:C]=PS.mean(axis=0)
print("%-26s %7s %7s %7s"%("prediction source","rel_l2","tke","mvpe"))
for tag,P_ in [("9-member SOUP (shipped)",P_soup),("ensemble mean K=3",P3d),("ensemble mean K=5",P5d),("ensemble mean K=9",P9d)]:
    a=acc(P_,ev); print("%-26s %7.2f %7.2f %7.2f"%(tag,a[0],a[1],a[2]))
print()
print("members used for K=3:",[MEM[i] for i in idx3])
