The registry is updated! The fingerprinting trick worked perfectly to recover the lost provenance.

### 1. The third eval set (`ft_md_*`)
The `ft_md_*` family was produced by `local_harness/finetune_moredata.py`. The literal split line is:
`vidx = set(range(0, ntraj, a.holdout_stride))`
Cross-referencing this with `soup_moredata.py:20` confirms the exact stride used was 9:
`vidx = sorted(set(range(0, ntraj, 9)))  # SAME holdout the 4 runs used`

**Did it train on `re_lohi` trajectories?** **YES**. 
By holding out only every 9th trajectory, the vast majority of the `re_lohi` trajectories ended up in its training set. It is strictly leaky on the `re_lohi` ruler. (I have named this leakage class `EVERY9` and mapped it to `USABLE AS BLEND PARTNER? = YES` alongside `EVERY5` and `ALL-DATA`).

### 2. Updated counts per LEAKAGE class
- **HONEST**: 43
- **ALL-DATA**: 35
- **EVERY5**: 11
- **EVERY9**: 4
- **INIT-LEAK**: 4
- **KIT**: 2
- **UNKNOWN**: 30

### 3. Moved out of UNKNOWN
15 checkpoints were successfully fingerprinted using their sidecars:

**Moved to EVERY9** (via base `[95.781, 77.026, 96.140]`):
- `ft_md_w10lr3_best.pth`
- `ft_md_w15lr1_best.pth`
- `ft_md_w15lr3_best.pth`
- `ft_md_w20lr2_best.pth`

**Moved to EVERY5** (via base `[96.325, 75.263, 96.861]`):
- `ft_w060_best.pth`
- `ft_w015_best.pth`
- `ft_w010_best.pth`
- `ft_w005_best.pth`
- `ft_w15lr3_best.pth`
- `ft_lr1e4_best.pth`
- `ft_lr1e5_best.pth`
- `ft_lr3e5_best.pth`
- `ft_long_w10lr1_best.pth`
- `ft_long_w15lr3_best.pth`
- `ft_long_w20lr2_best.pth`

### 4. Remaining UNKNOWN
**30 remain UNKNOWN.** These are mostly older manual experiments (like the `ft_m55_*` series) that either entirely lack a JSON sidecar, or do not record a `base` score triple in whatever sidecar they do have.

***

**On AMSE:**
Understood on the AMSE gate failing. It's extremely disappointing that it fails to push `cosθ` as intended and loses out to the control across the board, but catching the negative trend at step 3000 saves us a lot of time. If the 0.05 weight arm also fails to decouple the phase without ruining the baseline dynamics, we will indeed have to scrap it and find a completely new mechanism for §116.
