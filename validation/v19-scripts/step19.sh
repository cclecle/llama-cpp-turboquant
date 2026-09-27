#!/bin/bash
# step19.sh : v19, the q8_1 quantization of mat-vec inputs written by their producers.
#   - GGML_CUDA_FUSE_Q8_1 (default on): rms norm (+ mul) and scale -> unary (-> scale), when a quantized mat-vec on
#     their output follows, also write its q8_1 copy (the math of quantize_q8_1, now a shared device function) into
#     the q8_1 reuse cache, so the mat-vec skips its quantize kernel (step 15 trace: 484 quantize kernels per verify
#     pass, ~195 behind these two producers: the hyper-connection norm and gate of every sublayer)
# Bit-identical by construction: A and B must both generate step 15 D's text (8a0ec2a9ccd2816c6250db15e16181eb).
# Production stays stopped. 1) test-backend-ops (the quantize kernel changed), 2) A B B A, 3) trace of B
cd /mnt/gguf/r9v/bench
B=/opt/llamacpp/llama-cpp-mine-v19dev/build3/bin
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3

echo "### test-backend-ops"
echo "MUL_MAT q8_0: $(HIP_VISIBLE_DEVICES=0 timeout 1800 $B/test-backend-ops -o MUL_MAT -b ROCm0 -p type_a=q8_0 2>&1 | grep -E 'tests passed|FAIL' | head -4 | tr '\n' ' ')"
for op in MUL_MAT_VEC_FUSION RMS_NORM; do
  echo "$op: $(HIP_VISIBLE_DEVICES=0 timeout 1800 $B/test-backend-ops -o $op -b ROCm0 2>&1 | grep -E 'tests passed|FAIL' | head -4 | tr '\n' ' ')"
done

echo "### throughput, warm, A B B A"
n=0
for arm in A B B A; do
  n=$((n + 1)); E="X=0"; [ $arm = A ] && E="GGML_CUDA_FUSE_Q8_1=0"
  FN_ENV="$E" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s19-$arm$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:'
done
python3 step_ms.py v19dev-s19- results.jsonl
echo "### generated text per load (md5; reference 8a0ec2a9ccd2816c6250db15e16181eb)"
md5sum v19dev-s19-*.answer.txt

echo "### kernel trace: B"
DP_ARGS=fn-xl-tiered-r9vlike.args DP_PROMPT=prompt-32k.txt DP_TOKENS=512 DP_TAG=-s19 bash decode-profile.sh v19dev 2>&1 | grep -E 'timings|trace:'
d=prof-decode-v19dev-s19
f=$(find $d -name '*kernel_trace.csv' | head -1)
steps=$(python3 -c "import json; d=json.load(open('$d/answer.json'))['timings']; print(d['predicted_n'] - d['draft_n_accepted'])")
python3 torchtrace_anatomy.py rocprof $f $steps | head -12
echo STEP19_DONE
