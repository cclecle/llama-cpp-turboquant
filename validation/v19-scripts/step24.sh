#!/bin/bash
# step24.sh : v19,
#   1) the fused unary -> mul (the o * sigmoid(g) gate of every attention / gated delta net output, 48 per verify
#      pass) writes the q8_1 copy of its output for the output projection (GGML_CUDA_FUSE_Q8_1); A = off, B = on,
#      every text must be the reference (8a0ec2a9ccd2816c6250db15e16181eb)
#   2) fa_sweep.sh: the RDNA mma flash-attention config for 256/256 and 16 columns (the sparse verify attention and
#      the MTP draft attention; today 64 threads, occupancy 2, 32-cell KV batches: 57 us for 5.6 MB of gathered K/V)
# Production stays stopped.
cd /mnt/gguf/r9v/bench
T=/opt/llamacpp/tmp-v19
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3

echo "### build"
(cd $T && bash validation/v17-scripts/build-scratch.sh $T 2>&1 | head -3)

echo "### throughput, warm, A B B A"
n=0
for arm in A B B A; do
  n=$((n + 1)); E="X=0"; [ $arm = A ] && E="GGML_CUDA_FUSE_Q8_1=0"
  FN_ENV="$E" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s24-$arm$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:'
done
python3 step_ms.py v19dev-s24- results.jsonl
echo "### generated text per load (md5; reference 8a0ec2a9ccd2816c6250db15e16181eb)"
md5sum v19dev-s24-*.answer.txt

echo "### flash-attention config sweep (nthreads occupancy nbatch_fa)"
bash $T/validation/v19-scripts/fa_sweep.sh "64 2 32" "64 4 32" "128 2 32" "128 2 64" "256 1 64" "128 1 64"
echo STEP24_DONE
