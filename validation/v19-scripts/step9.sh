#!/bin/bash
# step9.sh : v19 optimisation, the MTP draft loop. The trace of step 7 showed ~16 ms per step outside the verify
# pass: 9.3 small draft evaluations (9.9 ms) and 0.7 ms of GPU idle before each (the host). Under -sm tensor the
# draft sampler cannot run on the backend, so every draft token copied the logits and ran the generic CPU chain,
# which builds and sorts a candidate array of the whole 248k vocabulary.
#   - LLAMA_SPEC_FAST_TOPK (default on): the top 10 straight from the logits, same token and probability
#   - the scale -> silu / sigmoid (-> scale) and rms norm -> scale fusions of step 8 now run: ggml_cuda_can_fuse is a
#     whitelist and did not list them, so step 8 measured them off (its A/B was the P2P AllReduce, 37.6 vs RCCL 31 us)
#   - also in this build: mmvk (RDNA4 K-quant mat-vec) applies the SWIGLU_CLAMP gate (it silently skipped it)
# A = fast top-k and both fusions off, B = default (RCCL AllReduce in both). Production stays stopped.
#   1) test-backend-ops MUL_MAT_VEC_FUSION and ADD, 2) throughput A B B A (ms/step), 3) host profile of B at 32k
cd /mnt/gguf/r9v/bench
B=/opt/llamacpp/llama-cpp-mine-v19dev/build3/bin
ENV_A="LLAMA_SPEC_FAST_TOPK=0 GGML_CUDA_FUSE_SCALE_UNARY=0 GGML_CUDA_FUSE_RMS_NORM_SCALE=0"
ENV_B="X=0"
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3

echo "### test-backend-ops"
for op in MUL_MAT_VEC_FUSION ADD; do
  echo "$op: $(HIP_VISIBLE_DEVICES=0 timeout 1800 $B/test-backend-ops -o $op -b ROCm0 2>&1 | grep -E 'tests passed|FAIL' | head -4 | tr '\n' ' ')"
done

echo "### throughput, warm, A B B A"
n=0
for arm in A B B A; do
  n=$((n + 1)); E=$ENV_B; [ $arm = A ] && E=$ENV_A
  FN_ENV="$E" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s9-$arm$n bash flashnext.sh v19dev 2>&1 | grep -E '^\{|load:'
done
python3 step_ms.py v19dev-s9- results.jsonl

echo "### kernel trace: B"
DP_ARGS=fn-xl-tiered-r9vlike.args DP_PROMPT=prompt-32k.txt DP_TOKENS=512 DP_TAG=-s9 bash decode-profile.sh v19dev 2>&1 | grep -E 'timings|trace:'
f=$(find prof-decode-v19dev-s9 -name '*kernel_trace.csv' | head -1)
steps=$(python3 -c "import json; d=json.load(open('prof-decode-v19dev-s9/answer.json'))['timings']; print(d['predicted_n'] - d['draft_n_accepted'])")
python3 torchtrace_anatomy.py rocprof $f $steps | head -14

echo "### host profile: B at 32k"
DP_ARGS=fn-xl-tiered-r9vlike.args HP_PROMPT=prompt-32k.txt HP_WAIT=45 bash hostprof.sh v19dev 60 $ENV_B 2>&1 | tail -40
echo STEP9_DONE
