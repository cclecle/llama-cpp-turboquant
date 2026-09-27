#!/bin/bash
# step22.sh : v19, two more producers write the q8_1 copy of their output (GGML_CUDA_FUSE_Q8_1, step 19): the
# hyper-connection mix (dsv4_hc_pre: the input of every attention / gated delta net / FFN sublayer) and the gated
# unary ops (GLU, F32). Step 20 trace: 290 quantize kernels per verify pass left. Bit-identical by construction.
# A = GGML_CUDA_FUSE_Q8_1=0, B = default; today's hot set (the 44 GiB one is the user's decision). Every text must be
# the reference (8a0ec2a9ccd2816c6250db15e16181eb). Production stays stopped.
#   1) test-backend-ops GLU, 2) A B B A + md5, 3) trace of B: kernels per step and what still precedes a quantize
cd /mnt/gguf/r9v/bench
B=/opt/llamacpp/llama-cpp-mine-v19dev/build3/bin
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3

echo "### test-backend-ops"
echo "GLU: $(HIP_VISIBLE_DEVICES=0 timeout 1800 $B/test-backend-ops -o GLU -b ROCm0 2>&1 | grep -E 'tests passed|FAIL' | head -4 | tr '\n' ' ')"

echo "### throughput, warm, A B B A"
n=0
for arm in A B B A; do
  n=$((n + 1)); E="X=0"; [ $arm = A ] && E="GGML_CUDA_FUSE_Q8_1=0"
  FN_ENV="$E" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s22-$arm$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:'
done
python3 step_ms.py v19dev-s22- results.jsonl
echo "### generated text per load (md5; reference 8a0ec2a9ccd2816c6250db15e16181eb)"
md5sum v19dev-s22-*.answer.txt

echo "### kernel trace: B"
DP_ARGS=fn-xl-tiered-r9vlike.args DP_PROMPT=prompt-32k.txt DP_TOKENS=512 DP_TAG=-s22 bash decode-profile.sh v19dev 2>&1 | grep -E 'timings|trace:'
d=prof-decode-v19dev-s22
f=$(find $d -name '*kernel_trace.csv' | head -1)
steps=$(python3 -c "import json; d=json.load(open('$d/answer.json'))['timings']; print(d['predicted_n'] - d['draft_n_accepted'])")
python3 torchtrace_anatomy.py rocprof $f $steps | head -4
timeout 600 python3 kernel_ngrams.py $f 2 80 2>&1 | grep -E 'passes|quantize_q8_1' | head -14
echo STEP22_DONE
