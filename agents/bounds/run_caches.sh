#!/bin/bash
# Sequential anchor-cache builds on GPU 2 only. Launched with setsid nohup by the caller.
B=/SML_DISK_24TB/rajeshr/Aryamann/UGP
P=/SML_DISK_24TB/rajeshr/Aryamann/env/bin/python
S=$B/submissions
cd $B/agents/bounds
for spec in \
  SHIFT85:submission_SHIFT_v2_W96_a85.zip \
  ENSFASTv3:submission_ENSEMBLE_FAST_v3.zip \
  TMEAN:submission_TMEAN.zip \
  TIER2D:submission_TIER2D.zip \
  LUTCAL:submission_LUTCAL.zip \
  FP16:submission_FP16.zip \
  SV2:submission_SV2.zip \
  SCREEN:submission_SCREEN.zip \
  BLIND:submission_BLIND.zip \
  W73:submission_W73.zip \
  RECIPE2:submission_RECIPE2.zip \
  ARCSINH:submission_ASYM_W96_safe_arcsinh.zip \
  SOUPv1:submission_SOUP_v1.zip ; do
  tag=${spec%%:*}; z=${spec#*:}
  echo "=== $tag $z $(date +%T)"
  nvidia-smi --query-gpu=index,temperature.gpu,clocks_throttle_reasons.sw_thermal_slowdown --format=csv,noheader -i 2
  CUDA_VISIBLE_DEVICES=2 $P mk_anchor_cache.py --zip $S/$z --tag $tag 2>&1 | grep -v -E "UserWarning|h\[:, :, ci\]|hd\[:, :, ci\]"
done
echo "[alldone] $(date +%T)"
