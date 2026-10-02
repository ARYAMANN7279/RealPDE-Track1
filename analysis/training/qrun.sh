#!/bin/bash
# qrun.sh <gpu> <tag:args> [tag:args ...]  -- run configs sequentially on one GPU
GPU=$1; shift
PY=/SML_DISK_24TB/rajeshr/Aryamann/env/bin/python3
cd /SML_DISK_24TB/rajeshr/Aryamann/UGP
for job in "$@"; do
  TAG="${job%%:*}"; ARGS="${job#*:}"
  echo "[qrun gpu$GPU] START $TAG  $ARGS  $(date -u +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=$GPU $PY -u train_mvpe/finetune2.py --gpu 0 --tag "$TAG" $ARGS \
      > train_mvpe/logs/$TAG.log 2>&1
  echo "[qrun gpu$GPU] DONE  $TAG  rc=$?  $(date -u +%H:%M:%S)"
  tail -2 train_mvpe/logs/$TAG.log
done
echo "[qrun gpu$GPU] QUEUE EMPTY"
