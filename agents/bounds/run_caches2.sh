#!/bin/bash
# Second batch: the two 13-feature-head anchors whose own predictions _agent2 did not keep.
B=/SML_DISK_24TB/rajeshr/Aryamann/UGP
P=/SML_DISK_24TB/rajeshr/Aryamann/env/bin/python
S=$B/submissions
cd $B/agents/bounds
for spec in LUTFIX:submission_LUTFIX.zip WIDE125:submission_WIDE125.zip ; do
  tag=${spec%%:*}; z=${spec#*:}
  echo "=== $tag $z $(date +%T)"
  nvidia-smi --query-gpu=index,temperature.gpu,clocks_throttle_reasons.sw_thermal_slowdown --format=csv,noheader -i 2
  CUDA_VISIBLE_DEVICES=2 $P mk_anchor_cache.py --zip $S/$z --tag $tag 2>&1 | grep -v -E "UserWarning|h\[:, :, ci\]|hd\[:, :, ci\]"
done
echo "[alldone2] $(date +%T)"
