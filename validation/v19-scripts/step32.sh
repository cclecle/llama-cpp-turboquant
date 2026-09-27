#!/bin/bash
# step32.sh : v19 prefill, our own F32 GEMM on RDNA4 (sgemm.cu). hipBLASLt runs every prefill F32 GEMM on 8x8/16x16
# macro tiles (router 512 x 1024 x 2560 at 1.0 ms = 2.7 TFLOPS, 1.49 s per GPU per 32k prefill; hyper-connection
# projections M <= 24, K 10240 at 216 / 66 us). The new kernel: 64x64 tiles, 4x4 per thread, deterministic split-k
# for the short-wide shapes. GGML_CUDA_SGEMM=0: hipBLAS. A different GEMM rounds differently, so the text changes;
# correctness is test-backend-ops MUL_MAT f32 against the CPU. Production stays stopped.
#   1) build, 2) MUL_MAT f32 x f32 tests, 3) perf of the 4 shapes, new vs hipBLAS, 4) throughput G (new) vs H (hipBLAS)
cd /mnt/gguf/r9v/bench
T=/opt/llamacpp/tmp-v19
B=$T/build3/bin
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3

echo "### build"
(cd $T && bash validation/v17-scripts/build-scratch.sh $T 2>&1 | head -6)

echo "### test-backend-ops MUL_MAT f32 x f32"
(cd /tmp && HIP_VISIBLE_DEVICES=0 timeout 1800 $B/test-backend-ops -o MUL_MAT -b ROCm0 -p 'type_a=f32,type_b=f32' 2>&1 | grep -E 'FAIL|passed' | tail -8)

echo "### perf: new kernel, then hipBLAS"
for E in X=0 GGML_CUDA_SGEMM=0; do
  echo "# $E"
  (cd /tmp && env $E HIP_VISIBLE_DEVICES=0 timeout 900 $B/test-backend-ops perf -o MUL_MAT -b ROCm0 -p 'type_a=f32,type_b=f32,m=(512|24|4|16384),n=(1024|4096),k=(2560|10240|128)' 2>&1 | grep -E 'MUL_MAT' | sed 's/  */ /g' | cut -c1-160)
done

echo "### throughput, ubatch 1024: G = new sgemm, H = GGML_CUDA_SGEMM=0"
n=0
for arm in G H H G; do
  n=$((n + 1)); E="X=0"; [ $arm = H ] && E="GGML_CUDA_SGEMM=0"
  FN_ENV="$E" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s32-$arm$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:|memory:|died'
done
python3 step_ms.py v19dev-s32- results.jsonl
echo "### generated text per load (md5; H must be the reference 8a0ec2a9ccd2816c6250db15e16181eb)"
md5sum v19dev-s32-*.answer.txt
echo STEP32_DONE
