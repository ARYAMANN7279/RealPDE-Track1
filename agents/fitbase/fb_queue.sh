#!/bin/bash
# FITBASE sequential job queue -- GPU 0 ONLY, exactly one job at a time (SHARED_CONTEXT sec6).
# usage (from $D):  setsid nohup bash fb_queue.sh <queuefile> < /dev/null > logs/queue_<name>.log 2>&1 & disown
# queue file lines:  TAG|script.py args...      (lines starting with # are skipped; TAG.done marks completion)
B=/SML_DISK_24TB/rajeshr/Aryamann/UGP; P=/SML_DISK_24TB/rajeshr/Aryamann/env/bin/python
D=$B/agents/fitbase; L=$D/logs; Q=$1
cd "$D" || exit 1
G0UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==0{print $2}')
[ -z "$G0UUID" ] && { echo "[queue] STOP: cannot read GPU0 uuid"; exit 5; }
echo "[queue] $(date '+%F %T') start $Q  (pid $$, GPU0 $G0UUID)"
thermal_ok () {
  local s t sw
  s=$(nvidia-smi -i 0 --query-gpu=temperature.gpu,clocks.sm,clocks_throttle_reasons.sw_thermal_slowdown --format=csv,noheader)
  t=${s%%,*}; sw=${s##*, }
  echo "[thermal] $(date +%T) GPU0: $s"
  [ -n "$t" ] && [ "$t" -lt 90 ] && [ "$sw" = "Not Active" ]
}
gpu0_busy () { nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -q "$G0UUID"; }
while IFS='|' read -r TAG CMD; do
  [ -z "$TAG" ] && continue
  case "$TAG" in \#*) continue;; esac
  if [ -f "$L/$TAG.done" ]; then echo "[queue] skip $TAG (already done)"; continue; fi
  n=0; until thermal_ok; do n=$((n+1)); [ $n -ge 60 ] && { echo "[queue] STOP: GPU0 hot/throttled for 60 min"; exit 2; }; sleep 60; done
  n=0
  while gpu0_busy; do
    n=$((n+1)); echo "[queue] GPU0 has a compute process:"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep "$G0UUID"
    [ $n -ge 4 ] && { echo "[queue] STOP: GPU0 occupied by another process -- report, do not move GPUs"; exit 3; }
    sleep 30
  done
  echo "[queue] $(date '+%F %T') launch $TAG :: $CMD"
  env CUDA_VISIBLE_DEVICES=0 $P $CMD < /dev/null > "$L/$TAG.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && grep -q '\[done\]' "$L/$TAG.log"; then
    touch "$L/$TAG.done"; echo "[queue] $(date '+%F %T') OK $TAG"
  else
    echo "[queue] $(date '+%F %T') FAIL $TAG rc=$rc -- STOP"; tail -5 "$L/$TAG.log"; exit 4
  fi
  sleep 15
done < "$Q"
echo "[queue] $(date '+%F %T') [queue done] $Q"
