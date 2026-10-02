import numpy as np, json, os, sys
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
X=np.load(f"{B}/local_harness/tr_frames.npy", mmap_mode="r")
m=json.load(open(f"{B}/local_harness/tr_meta.json"))
off,lens,names=m["off"],m["lens"],m["names"]
RE=np.array([int(n.split("_")[0]) for n in names])
AOA=np.array([int(n.split("_")[1].split(".")[0]) for n in names])

# Load banked results
res = np.load(f"{B}/agents/moe/res/banked.npz")
dm = res['dm'] # rel_l2 error?
tk = res['tk'] # tke error?
mv = res['mv'] # mvpe error?

# Load router features
feats = np.load(f"{B}/agents/moe/router_feats.npz")
Fev = feats['Fev'] # (900, dim)
sub = feats['sub']  # window starts

# Map windows to trajectories
traj_of = np.zeros(len(X), int)
for i in range(len(names)): traj_of[off[i]:off[i]+lens[i]] = i
y_traj = traj_of[sub]

# Regimes
re_traj = RE[y_traj]
aoa_traj = AOA[y_traj]

print("Regime Error Analysis (Banked Model):")
print("-" * 60)
print(f"{'Regime':<20} {'Count':<10} {'rel_l2':<12} {'tke':<12} {'mvpe':<12}")
for r in np.unique(re_traj):
    mask = (re_traj == r)
    print(f"Re={r:<15} {mask.sum():<10} {dm[mask].mean():<12.4f} {tk[mask].mean():<12.4f} {mv[mask].mean():<12.4f}")

print("\n" + "-" * 60)
for a in np.unique(aoa_traj):
    mask = (aoa_traj == a)
    print(f"AoA={a:<14} {mask.sum():<10} {dm[mask].mean():<12.4f} {tk[mask].mean():<12.4f} {mv[mask].mean():<12.4f}")
