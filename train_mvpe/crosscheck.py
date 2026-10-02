"""Independent check of the v-scale claim, without using the fit.

The two CONSTANT anchors are two linear equations in four real CDF values:
  0.5[e(hu)Fu(0.030 ) + e(hv)Fv(0.010 )] = 0.4399
  0.5[e(hu)Fu(0.0129) + e(hv)Fv(0.0098)] = 0.4876       e(h)=exp(-2h/sigma)
Fv(0.010) ~ Fv(0.0098) to within the local slope, so the system pins the real
Fu/Fv values almost independently of any distribution family. Compare those to
what the LOCAL data gives under each candidate scaling.
"""
import numpy as np
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; SIG = 0.0563870259
z = np.load(f"{B}/train_mvpe/runs/arrays.npz")
eu = np.sort(z["err_u"][z["held_u"]][::4].astype(np.float64))
ev = np.sort(z["err_v"][z["held_v"]][::4].astype(np.float64))
nu, nv = eu.size, ev.size
e = lambda h: np.exp(-2*h/SIG)
F = lambda s, t: np.searchsorted(s, t, side="right")/s.size
wu, wv = nu/(nu+nv), nv/(nu+nv)
print("scored elements: u %d  v %d  (weights %.4f / %.4f)" % (nu, nv, wu, wv))

# solve the two anchor equations, treating Fv(0.010) == Fv(0.0098) == y
# wu*e(0.030 )*Fu(0.030 ) + wv*e(0.010 )*y = 0.4399
# wu*e(0.0129)*Fu(0.0129) + wv*e(0.0098)*y = 0.4876
print("\nimplied REAL CDF values (solved from the two anchors):")
for Fu30 in (1.00, 0.98, 0.95):
    y = (0.4399 - wu*e(0.030)*Fu30) / (wv*e(0.010))
    Fu129 = (0.4876 - wv*e(0.0098)*y) / (wu*e(0.0129))
    print("   assuming Fu(0.030)=%.2f ->  Fv(0.0098) = %.4f   Fu(0.0129) = %.4f" % (Fu30, y, Fu129))

print("\nwhat the LOCAL data gives under each scaling (err_real = a*err_local):")
print("   %-34s %8s %8s %8s" % ("", "Fu(.030)", "Fu(.0129)", "Fv(.0098)"))
for tag, au, av in (("CURRENT global_scale  2.320/1.198", 2.3199, 1.1982),
                    ("refitted scalar       2.150/2.425", 2.1500, 2.4250),
                    ("refitted power (a only shown)    ", 3.6667, 3.1833)):
    print("   %-34s %8.4f %8.4f %8.4f"
          % (tag, F(eu, 0.030/au), F(eu, 0.0129/au), F(ev, 0.0098/av)))
print("\n   (power row uses b_u=1.10, b_v=1.05, so its thresholds differ slightly;")
print("    shown for scale only -- distfit2 evaluates it exactly.)")
