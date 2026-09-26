#!/bin/bash
# step10.sh : v19 optimisation, mat-vec efficiency of verify batches, and the MoE reuse potential.
#   - GGML_CUDA_MMVQ_TALL_K (default on): on RDNA4 a verify batch through a matrix with few rows and a long K gets 4
#     warps per row instead of one (the hyper-connection down projection 10240 -> 320, 96 per step: ~240 GB/s)
#   - GGML_CUDA_MOE_STATS=1 (synchronizes, so CUDA graphs off): per MoE call of a verify batch, the (token, slot)
#     pairs, the distinct experts, and how many of each are cold (host memory): what reading every distinct expert
#     once per batch (R9V's reuse kernel) would save
# A = tall-K off, B = default. Production stays stopped.
cd /mnt/gguf/r9v/bench
B=/opt/llamacpp/llama-cpp-mine-v19dev/build3/bin
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3

echo "### test-backend-ops"
echo "MUL_MAT: $(HIP_VISIBLE_DEVICES=0 timeout 1800 $B/test-backend-ops -o MUL_MAT -b ROCm0 -p type_a=q8_0 2>&1 | grep -E 'tests passed|FAIL' | head -4 | tr '\n' ' ')"

echo "### throughput, warm, A B B A"
n=0
for arm in A B B A; do
  n=$((n + 1)); E=X=0; [ $arm = A ] && E=GGML_CUDA_MMVQ_TALL_K=0
  FN_ENV="$E" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s10-$arm$n bash flashnext.sh v19dev 2>&1 | grep -E '^\{|load:'
done
python3 step_ms.py v19dev-s10- results.jsonl

echo "### MoE sharing stats (one run, CUDA graphs off)"
FN_ENV="GGML_CUDA_MOE_STATS=1 GGML_CUDA_DISABLE_GRAPHS=1" FN_WARM=0 FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s10-stats bash flashnext.sh v19dev 2>&1 | grep -E 'load:'
grep "moe stats" v19dev-s10-stats.log | tail -3
echo STEP10_DONE
