#!/bin/bash
# step8.sh : v19 optimisation, kernel-count cuts on top of step 7, one build, env toggles:
#   - GGML_CUDA_Q8_1_REUSE: consecutive quantized mat-vecs on the same activations quantize them once
#   - GGML_CUDA_FUSE_SCALE_UNARY: scale -> silu / sigmoid (-> scale) in one kernel (the hyper-connection gates)
#   - GGML_CUDA_FUSE_RMS_NORM_SCALE: rms norm -> scale in one kernel (the gated delta net l2 norm of q and k)
#   - (no toggle) the conv-state rollback tails are copied straight from the strided view, without a cont first
#   - (no toggle) q8_0 mat-vec + mat-vec + GLU fuse at 2-8 columns too (the shared expert of a verify batch)
#   - GGML_CUDA_AR_HYBRID=32768: decode-sized AllReduces through the direct-P2P kernel (bit-identical to RCCL);
#     re-tested now that the MoE split is balanced (step 7), since the wait it measured was the uneven split
# A = reuse and fusions off (RCCL AllReduce), B = all on plus the P2P AllReduce. Production stays stopped.
#   1) test-backend-ops for the fused paths, 2) perplexity at ub 8 / 8k A vs B (bit-identical expected),
#   3) same-config throughput A B B A, 4) kernel trace of B and its per-step anatomy
cd /mnt/gguf/r9v/bench
B=/opt/llamacpp/llama-cpp-mine-v19dev/build3/bin
M=/mnt/gguf/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
HOT=$(readlink -f hot-r9v-uva16-uniform.txt)
OT_TIER='per_layer_token_embd=CPU,blk\.[0-9]+\.ffn_(gate|up|down)_exps\.weight=ROCm0_TIERED'
ENV_A="GGML_CUDA_Q8_1_REUSE=0 GGML_CUDA_FUSE_SCALE_UNARY=0 GGML_CUDA_FUSE_RMS_NORM_SCALE=0"
ENV_B="GGML_CUDA_AR_HYBRID=32768"
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3

echo "### test-backend-ops"
for op in ADD SCALE UNARY MUL_MAT_VEC_FUSION GLU; do
  echo "$op: $(HIP_VISIBLE_DEVICES=0 timeout 1800 $B/test-backend-ops -o $op -b ROCm0 2>&1 | grep -E 'tests passed|FAIL' | head -3 | tr '\n' ' ')"
done

echo "### perplexity at ub 8, ctx 8192: A vs B"
for arm in A B; do
  X="GGML_CUDA_MOE_HOT_FILE=$HOT"; [ $arm = A ] && X="$X $ENV_A" || X="$X $ENV_B"
  r=$(env $X HIP_VISIBLE_DEVICES=0,1 timeout 3600 $B/llama-perplexity -m $M -f /root/ppl.txt -c 8192 -b 8 -ub 8 --chunks 1 \
      -sm tensor -ngl 999 -ot "$OT_TIER" --n-cpu-moe 0 -fa on -ctk q8_0 -ctv q8_0 -t 12 -fit off 2>&1 | tee ppl-s8-$arm.log |
      grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' | tr '\n' ' ')
  echo "$arm: $r"
done

echo "### throughput, warm, A B B A"
n=0
for arm in A B B A; do
  n=$((n + 1)); E=$ENV_B; [ $arm = A ] && E=$ENV_A
  FN_ENV="$E" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s8-$arm$n bash flashnext.sh v19dev 2>&1 | grep -E '^\{|load:'
done
python3 step_ms.py v19dev-s8- results.jsonl

echo "### kernel trace: B"
env $ENV_B DP_ARGS=fn-xl-tiered-r9vlike.args DP_PROMPT=prompt-32k.txt DP_TOKENS=512 DP_TAG=-s8 bash decode-profile.sh v19dev 2>&1 | grep -E 'timings|trace:'
f=$(find prof-decode-v19dev-s8 -name '*kernel_trace.csv' | head -1)
steps=$(python3 -c "import json; d=json.load(open('prof-decode-v19dev-s8/answer.json'))['timings']; print(d['predicted_n'] - d['draft_n_accepted'])")
python3 torchtrace_anatomy.py rocprof $f $steps
echo STEP8_DONE
