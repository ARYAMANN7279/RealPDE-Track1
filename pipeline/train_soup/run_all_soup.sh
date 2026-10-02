#!/bin/bash
# SOUP pipeline. Note `set -o pipefail`: train/run_all.sh lacks it, and because
# each stage is piped through `grep -v Warning` the pipeline's exit status was
# grep's, not python's. On Aug 21 a NameError in 05_build_lut.py was swallowed
# that way and the script printed "PIPELINE COMPLETE" while silently reusing a
# stale head_assets.npz. Do not remove this line.
set -e
set -o pipefail

B=/SML_DISK_24TB/rajeshr/Aryamann/UGP
cd $B/train_soup
E=$B/../env/bin/python3
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

echo "########## pack soup -> fp16 ##########"
$E $B/starting_kit_v9/realpde_t1_starting_kit_v9/pack_ckpt_fp16.py \
   $B/local_harness/soup_best.pth $B/train_work_soup/sim_real_fno_fp16.pth
echo "[ok] pack"

# 03_fit_spectral.py is deliberately NOT run: the spectral correction is dropped.
for s in 02_cache_preds.py 04_train_head.py 05_build_lut.py 06_build_submission.py; do
  echo "########## $s ##########"
  $E -u $s 2>&1 | grep -v Warning
  echo "[ok] $s"
done
echo "PIPELINE COMPLETE"
