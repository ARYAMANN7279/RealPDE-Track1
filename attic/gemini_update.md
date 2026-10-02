
## 65. I am Gemini 3.1 Pro

### 65.1 The True Honest Baseline and the NEW_SOUP Tie
The previously projected `+0.44` live gain was a phantom caused by incomplete leakage math. Correcting the shipped soup (`soup_v1`) for its leakage means converting all three leaked metrics into `d_acc` units:
- rel_l2 (+0.201 * 0.669 = +0.1345)
- tke (+2.727 * 0.157 = +0.4281)
- mvpe (+0.171 * 0.170 = +0.0291)
- Total Leakage = **+0.5917**

The shipped-corrected score is therefore `1.3684 - 0.5917 = +0.7767`. 
My `NEW_SOUP` (7 members trained strictly without `re_lohi`) evaluated to `+0.7637` on the honest `starts_664` protocol. **This is a statistical tie (-0.013).**
While it doesn't give us a free 79.8+ route, it produces something much more valuable: **a clean, honest measuring stick (`re_lohi` soup) that gives us a reliable ruler for the first time.**

### 65.2 Dead Ends: SV3, Bounds, and Time-Mean
- **SV3 (100% Data) is unfalsifiable:** Training with `--split none` provides zero validation windows. Without a holdout set, the members cannot be checkpoint-selected or validated. It's exactly the same trap as `soup_v2`. The SV3 jobs were killed to free up the GPUs.
- **Time-Mean is saturated:** The time-mean head is already shipped and only captured `f=0.049` (yielding `+0.025` live). The optimistic `+0.20 to +0.33` projection was detached from historical reality.
- **Bounds are solved:** Training an SE-Block U-Net (`joint_asym_attn.py`) was a waste of time. The SPS policy space is a closed 4-DOF problem completely handled by the LUT calibration. The FNO base model is the ONLY remaining lever that lifts W and E in the SPS.

### 65.3 The Path to 79.8+ (Honest Hyperparameter Sweep)
Since an honestly-measured gain transfers at ~2x, the strategy is now cleanly defined: **find a recipe that beats `+0.7767` on the honest `re_lohi` split.**
I launched a 4-axis sweep across all 4 GPUs for 8000 steps testing the core `ftmv.py` levers:
1. `aug=phase` vs `aug=none`
2. `lr=3e-5` vs `lr=1e-5`
3. `wtke=0.08` vs `wtke=0.12`

### 65.4 Infrastructure Fixes
1. **FP32 to FP16 Cap Bypass:** `NEW_SOUP` (FP32) was 390 MB, which blew past CodaLab's 256 MB cap during `build_asym.py`. I utilized `to_fp16.py` to shrink it to 193 MB (`soup_honest_fp16.pth`) and patched the build scripts to handle it natively.
2. **OOM Killer on Caches:** Attempting to load the 23.4 GB `cache_honest_stride3.npz` crashed the VM's CPU RAM (OOM killer killed all running Pythons). Memory mapping (`mmap_mode="r"`) is absolutely required for operations moving forward.
