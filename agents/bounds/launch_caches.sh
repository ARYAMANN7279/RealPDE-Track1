#!/bin/bash
# Thermal + occupancy gate, then launch run_caches.sh detached on GPU 2 only.
B=/SML_DISK_24TB/rajeshr/Aryamann/UGP
cd $B/agents/bounds || exit 1
Q=$(nvidia-smi -i 2 --query-gpu=temperature.gpu,clocks_throttle_reasons.sw_thermal_slowdown --format=csv,noheader,nounits)
T=$(echo "$Q" | cut -d, -f1 | xargs)
S=$(echo "$Q" | cut -d, -f2 | xargs)
APPS=$(nvidia-smi -i 2 --query-compute-apps=pid,used_memory --format=csv,noheader)
echo "GPU2 temp=$T sw_thermal='$S' apps='$APPS'"
if [ "$T" -ge 90 ] || [ "$S" = "Active" ]; then echo "THERMAL GATE: not launching"; exit 2; fi
if [ -n "$APPS" ]; then echo "GPU2 OCCUPIED: not launching"; exit 3; fi
setsid nohup env CUDA_VISIBLE_DEVICES=2 bash run_caches.sh < /dev/null > run_caches.log 2>&1 & disown
sleep 2
echo "launched; log head:"; head -3 run_caches.log
