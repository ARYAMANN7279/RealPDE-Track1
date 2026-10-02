"""Shared config. Every path points at the official release only."""
import os, numpy as np, random
ROOT   = os.environ.get("REALPDE_ROOT", "/SML_DISK_24TB/rajeshr/Aryamann/UGP")
TRAIN_REAL = os.path.join(ROOT, "data/comp_real/train_real")      # Drive release
CKPT_FP32  = os.path.join(ROOT, "data/comp_real/sim_real_fno.pth") # Drive baseline
KIT        = os.path.join(ROOT, "starting_kit_v9/realpde_t1_starting_kit_v9")
WORK       = os.path.join(ROOT, "train_work")
SEED = 1234
EXCLUDE = {"7575_0.h5"}      # duplicate of 6300_0.h5 (forum, benslash2 2026-08-06)
IN_STEP = OUT_STEP = 20
SUB_S = 2                    # raw PIV 64x128 -> eval 32x64
def seed_all():
    random.seed(SEED); np.random.seed(SEED)
    try:
        import torch; torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass
    print("[seed] %d" % SEED)
os.makedirs(WORK, exist_ok=True)
