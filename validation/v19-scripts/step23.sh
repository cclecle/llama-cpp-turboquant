#!/bin/bash
# step23.sh : v19, the producers write the q8_1 copy in the rows of the consumer's view (element e at block
# (e / w)*row_blocks + (e % w)/32, w the mat-vec's row width), so any reshape with rows of a multiple of 32 works:
# the gated delta net output gate [128, heads, T] read as [3072, T] (step 22: 48 quantize kernels per pass left
# behind the GLU). A = GGML_CUDA_FUSE_Q8_1=0, B = default; today's hot set. Every text must be the reference
# (8a0ec2a9ccd2816c6250db15e16181eb). Production stays stopped.
#   1) A B B A + md5, 2) trace of B: kernels per step and what still precedes a quantize
cd /mnt/gguf/r9v/bench
B=/opt/llamacpp/llama-cpp-mine-v19dev/build3/bin
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3

echo "### test-backend-ops"
for op in SWIGLU RMS_NORM; do
  echo "$op: $(HIP_VISIBLE_DEVICES=0 timeout 1800 $B/test-backend-ops -o $op -b ROCm0 2>&1 | grep -E 'tests passed|FAIL' | head -4 | tr '
' ' ')"
done

echo "### throughput, warm, A B B A"
n=0
for arm in A B B A; do
  n=$((n + 1)); E="X=0"; [ $arm = A ] && E="GGML_CUDA_FUSE_Q8_1=0"
  FN_ENV="$E" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s23-$arm$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:'
done
python3 step_ms.py v19dev-s23- results.jsonl
echo "### generated text per load (md5; reference 8a0ec2a9ccd2816c6250db15e16181eb)"
md5sum v19dev-s23-*.answer.txt

echo "### kernel trace: B"
DP_ARGS=fn-xl-tiered-r9vlike.args DP_PROMPT=prompt-32k.txt DP_TOKENS=512 DP_TAG=-s23 bash decode-profile.sh v19dev 2>&1 | grep -E 'timings|trace:'
d=prof-decode-v19dev-s23
f=$(find $d -name '*kernel_trace.csv' | head -1)
steps=$(python3 -c "import json; d=json.load(open('$d/answer.json'))['timings']; print(d['predicted_n'] - d['draft_n_accepted'])")
python3 torchtrace_anatomy.py rocprof $f $steps | head -4
timeout 600 python3 kernel_ngrams.py $f 2 80 2>&1 | grep -E 'passes|quantize_q8_1' | head -14
echo STEP22_DONE
