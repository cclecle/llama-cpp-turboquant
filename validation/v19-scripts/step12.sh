#!/bin/bash
# step12.sh : v19 optimisation, why step 11 lost ~2-4 ms/step, and the MoE reuse kernel without its serialization.
#   - GGML_META_ALLOC_DEPS (default on): the meta backend's allocation dependencies of step 11 (the fused MoE weighted
#     sum then runs in decode). Suspect: the CUDA dependency pass (a matcher over every node) now runs for every graph
#     the scheduler splits, on the host, and decode waits on the host.
#   - GGML_CUDA_MOE_REUSE (default on): the reuse kernel now computes all route dots without a branch; the ISA of step
#     11 showed the weight loads folded but each route's activation loads waited behind its own branch.
#   - GGML_CUDA_UVA_NONCOHERENT=1: cold experts in non-coherent (GPU-cacheable) host memory, so the duplicate reads of
#     the old kernel may hit L2 instead of PCIe. Speed only here; weights are written once at load, but correctness
#     needs a separate check before any use.
# Arms: A = reuse off + deps off (step 10), B = reuse off, C = default, D = reuse off + non-coherent. Production stays
# stopped. 1) test-backend-ops MUL_MAT_ID and MUL_MAT_VEC_FUSION, 2) throughput A B C D D C B A, 3) trace of the winner
cd /mnt/gguf/r9v/bench
B=/opt/llamacpp/llama-cpp-mine-v19dev/build3/bin
ENV_A="GGML_CUDA_MOE_REUSE=0 GGML_META_ALLOC_DEPS=0"
ENV_B="GGML_CUDA_MOE_REUSE=0"
ENV_C="X=0"
ENV_D="GGML_CUDA_MOE_REUSE=0 GGML_CUDA_UVA_NONCOHERENT=1"
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3

echo "### test-backend-ops"
for op in MUL_MAT_ID MUL_MAT_VEC_FUSION; do
  echo "$op: $(HIP_VISIBLE_DEVICES=0 timeout 3000 $B/test-backend-ops -o $op -b ROCm0 2>&1 | grep -E 'tests passed|FAIL' | head -4 | tr '\n' ' ')"
done

echo "### throughput, warm, A B C D D C B A"
n=0
for arm in A B C D D C B A; do
  n=$((n + 1)); eval E=\$ENV_$arm
  FN_ENV="$E" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s12-$arm$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:'
done
python3 step_ms.py v19dev-s12- results.jsonl

echo "### kernel trace: C"
DP_ARGS=fn-xl-tiered-r9vlike.args DP_PROMPT=prompt-32k.txt DP_TOKENS=512 DP_TAG=-s12 bash decode-profile.sh v19dev 2>&1 | grep -E 'timings|trace:'
f=$(find prof-decode-v19dev-s12 -name '*kernel_trace.csv' | head -1)
steps=$(python3 -c "import json; d=json.load(open('prof-decode-v19dev-s12/answer.json'))['timings']; print(d['predicted_n'] - d['draft_n_accepted'])")
python3 torchtrace_anatomy.py rocprof $f $steps | head -8
python3 mmv_breakdown.py $f $steps "" moe | head -10
echo STEP12_DONE
