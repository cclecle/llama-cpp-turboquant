#!/bin/bash
# step33.sh : v19 prefill, the F32 GEMM with 16- and 32-row tiles for matrices with few rows (the hyper-connection
# projections, M 4 / 24, K 10240: step 32's 64-row tile wasted most FMAs and ran 148 / 155 us, memory bound is ~70).
#   1) build, 2) MUL_MAT f32 x f32 tests, 3) perf of the 4 shapes, 4) throughput, 2 loads (the text of step 32's G
#   arm is the reference for this GEMM: same k order per output, only the tile height changes)
cd /mnt/gguf/r9v/bench
T=/opt/llamacpp/tmp-v19
B=$T/build3/bin
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3

echo "### build"
(cd $T && bash validation/v17-scripts/build-scratch.sh $T 2>&1 | head -6)

echo "### test-backend-ops MUL_MAT f32 x f32"
(cd /tmp && HIP_VISIBLE_DEVICES=0 timeout 1800 $B/test-backend-ops -o MUL_MAT -b ROCm0 -p 'type_a=f32,type_b=f32' 2>&1 | grep -E 'FAIL|passed' | tail -8)

echo "### perf"
(cd /tmp && HIP_VISIBLE_DEVICES=0 timeout 900 $B/test-backend-ops perf -o MUL_MAT -b ROCm0 -p 'type_a=f32,type_b=f32,m=(512|24|4|16384),n=(1024|4096),k=(2560|10240|128)' 2>&1 | grep -E 'MUL_MAT' | sed 's/  */ /g' | cut -c1-160)

echo "### throughput, ubatch 1024"
for n in 1 2; do
  FN_ENV="X=0" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s33-G$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:|memory:|died'
done
python3 step_ms.py v19dev-s33- results.jsonl
echo "### generated text per load (md5; compare with step 32's G arm)"
md5sum v19dev-s33-*.answer.txt v19dev-s32-G*.answer.txt
echo STEP33_DONE
