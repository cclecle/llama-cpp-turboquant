#!/bin/bash
# step41.sh : v19 decode, the verify mat-vecs' launch shape on RDNA4 (step 40: at 5 columns every q8_0 matrix runs
# one 32-thread wave per row, ~55% of bandwidth on average; tall-K gives 4 waves per row only below 2048 rows and
# from 128 K blocks). Perf of the 8 decode shapes at 1 wave per row (TALL_K=0), the default, and 4 waves per row for
# every shape (TALL_K_ROWS=1000000 TALL_K_KBLOCKS=0).
cd /mnt/gguf/r9v/bench
T=/opt/llamacpp/tmp-v19
B=$T/build3/bin
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
echo "### build"
(cd $T && bash validation/v17-scripts/build-scratch.sh $T 2>&1 | head -6)
for E in GGML_CUDA_MMVQ_TALL_K=0 X=0 "GGML_CUDA_MMVQ_TALL_K_ROWS=1000000 GGML_CUDA_MMVQ_TALL_K_KBLOCKS=0"; do
  echo "# $E"
  (cd /tmp && env $E HIP_VISIBLE_DEVICES=0 timeout 900 $B/test-backend-ops perf -o MUL_MAT -b ROCm0 -p 'type_a=q8_0,type_b=f32,m=[0-9]+,n=5,k=' 2>&1 | grep -E 'MUL_MAT' | sed -E 's/.*m=([0-9]+),n=5,k=([0-9]+).* ([0-9.]+) us\/run.*/\1 x \2: \3 us/')
done
echo STEP41_DONE
