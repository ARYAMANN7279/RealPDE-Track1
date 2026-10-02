#!/bin/bash
set -e
set -o pipefail
B=/SML_DISK_24TB/rajeshr/Aryamann/UGP
cd $B/train_soup
E=$B/../env/bin/python3
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
echo "########## head A: calibrated target (== v7 machinery) ##########"
HEAD_TARGET=calibrated HEAD_OUT=head.pth $E -u 04_train_head.py 2>&1 | grep -v Warning
echo "########## head B: clean target ##########"
HEAD_TARGET=clean HEAD_OUT=head_clean.pth $E -u 04_train_head.py 2>&1 | grep -v Warning
echo "########## analysis ##########"
$E -u analyze.py 2>&1 | grep -v Warning
echo "HEADS COMPLETE"
