#!/bin/bash
# test-backend-ops for new (GPU0) and v16 (GPU1) builds, per op, in parallel
W=/root/work-20260925; mkdir -p $W; cd $W
NEW=/opt/llamacpp/tmp-rdnab/build3/bin; OLD=/opt/llamacpp/llama-cpp-mine-v16/build3/bin
for op in FLASH_ATTN_EXT MUL_MAT MUL_MAT_ID ADD; do
  (HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$NEW timeout 3000 $NEW/test-backend-ops -o $op -b ROCm0 > new_$op.log 2>&1; echo rc=$? >> new_$op.log) &
  (HIP_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=$OLD timeout 3000 $OLD/test-backend-ops -o $op -b ROCm0 > old_$op.log 2>&1; echo rc=$? >> old_$op.log) &
  wait
done
echo done > tbo.done
