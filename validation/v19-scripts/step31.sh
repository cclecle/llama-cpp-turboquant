#!/bin/bash
# step31.sh : v19 prefill, three items.
#   a) mm_ids_helper (3 calls per MoE layer, ~200 us each at ubatch 1024: 0.94 s per GPU per 32k prefill) was bound by
#      the latency of one load per loop step; it now loads 8 steps ahead. Same output: the text must stay the
#      reference (8a0ec2a9ccd2816c6250db15e16181eb).
#   b) the F32 GEMMs (router 1.0 ms per call at 2.7 TFLOPS on a macro-tile 8x8 kernel, hc_inject, indexer scores):
#      rocBLAS's own Tensile kernels (ROCBLAS_USE_HIPBLASLT=0) vs hipBLASLt (=1) vs the default. A different GEMM may
#      round differently, so the prefill-written KV and then the text can change; the md5 is reported, not required.
#   c) S = LLAMA_QSA_SPARSE=1024: prefill attention through the per-query top-k lists (the decode path) instead of dense
#      FA over every cell with a -inf mask. Different tiling of the online softmax: the text may change.
# Production stays stopped.
cd /mnt/gguf/r9v/bench
T=/opt/llamacpp/tmp-v19
B=$T/build3/bin
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3

echo "### build"
(cd $T && bash validation/v17-scripts/build-scratch.sh $T 2>&1 | head -6)

echo "### test-backend-ops MUL_MAT_ID"
(cd /tmp && HIP_VISIBLE_DEVICES=0 timeout 1800 $B/test-backend-ops -o MUL_MAT_ID -b ROCm0 2>&1 | grep -E 'FAIL|passed' | tail -8)

echo "### throughput, ubatch 1024: D = default, R = ROCBLAS_USE_HIPBLASLT=0, L = ROCBLAS_USE_HIPBLASLT=1, S = LLAMA_QSA_SPARSE=1024"
n=0
for arm in D R L S S L R D; do
  n=$((n + 1))
  case $arm in D) E="X=0" ;; R) E="ROCBLAS_USE_HIPBLASLT=0" ;; L) E="ROCBLAS_USE_HIPBLASLT=1" ;; S) E="LLAMA_QSA_SPARSE=1024" ;; esac
  FN_ENV="$E" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s31-$arm$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:|memory:|died'
done
python3 step_ms.py v19dev-s31- results.jsonl
echo "### generated text per load (md5; reference 8a0ec2a9ccd2816c6250db15e16181eb)"
md5sum v19dev-s31-*.answer.txt
echo STEP31_DONE
