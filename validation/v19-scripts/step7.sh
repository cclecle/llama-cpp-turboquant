#!/bin/bash
# step7.sh : v19 optimisation items 1b-1e on top of step 6 (sparse QSA attention):
#   - the QSA indexer from two fused ops, ggml_qsa_pool (gather + mean + RMS norm of the cached keys) and
#     ggml_qsa_expand (block score to cells, plus the mask); LLAMA_QSA_FUSED=0 restores the generic ops
#   - rope_multi packs several short rows into one block (the 8,192 pooled 128-wide keys per layer)
#   - BF16 mat-vec up to 5 columns on RDNA4 (the indexer projections left mul_mat_f)
#   - the expert and shared-expert FFN split 320/320 instead of 384/256 (the split unit halves to 64)
# The A arm is the step 6 build, copied to /opt/llamacpp/llama-cpp-mine-v19s6 and run with its own library path.
# Production stays stopped.
#   1) test-backend-ops for the touched ops, 2) perplexity at ub 8 / 8k: fused vs generic indexer (same split),
#   3) same-config throughput A (step 6) B (this) B A, 4) kernel trace of B and its per-step anatomy
cd /mnt/gguf/r9v/bench
B=/opt/llamacpp/llama-cpp-mine-v19dev/build3/bin
M=/mnt/gguf/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
HOT=$(readlink -f hot-r9v-uva16-uniform.txt)
OT_TIER='per_layer_token_embd=CPU,blk\.[0-9]+\.ffn_(gate|up|down)_exps\.weight=ROCm0_TIERED'
S6=/opt/llamacpp/llama-cpp-mine-v19s6/build3/bin
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3

echo "### test-backend-ops"
for op in QSA_POOL QSA_EXPAND ROPE FLASH_ATTN_EXT; do
  echo "$op: $(HIP_VISIBLE_DEVICES=0 timeout 1800 $B/test-backend-ops -o $op -b ROCm0 2>&1 | grep -E 'tests passed|FAIL' | head -3 | tr '\n' ' ')"
done
echo "MUL_MAT bf16: $(HIP_VISIBLE_DEVICES=0 timeout 1800 $B/test-backend-ops -o MUL_MAT -b ROCm0 -p type_a=bf16 2>&1 | grep -E 'tests passed|FAIL' | head -3 | tr '\n' ' ')"

echo "### perplexity at ub 8, ctx 8192: generic vs fused indexer"
for arm in generic fused; do
  X="GGML_CUDA_MOE_HOT_FILE=$HOT"; [ $arm = generic ] && X="$X LLAMA_QSA_FUSED=0"
  r=$(env $X HIP_VISIBLE_DEVICES=0,1 timeout 3600 $B/llama-perplexity -m $M -f /root/ppl.txt -c 8192 -b 8 -ub 8 --chunks 1 \
      -sm tensor -ngl 999 -ot "$OT_TIER" --n-cpu-moe 0 -fa on -ctk q8_0 -ctv q8_0 -t 12 -fit off 2>&1 | tee ppl-qsa7-$arm.log |
      grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' | tr '\n' ' ')
  echo "$arm: $r"
done

echo "### throughput, warm, A (step 6) B B A"
n=0
for arm in A B B A; do
  n=$((n + 1))
  if [ $arm = A ]; then
    FN_LD=$S6 FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s7-$arm$n bash flashnext.sh v19s6 2>&1 | grep -E '^\{|load:'
  else
    FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s7-$arm$n bash flashnext.sh v19dev 2>&1 | grep -E '^\{|load:'
  fi
done
head -c 300 v19dev-s7-B2.answer.txt; echo

echo "### kernel trace: B"
DP_ARGS=fn-xl-tiered-r9vlike.args DP_PROMPT=prompt-32k.txt DP_TOKENS=512 DP_TAG=-s7 bash decode-profile.sh v19dev 2>&1 | grep -E 'timings|trace:'
f=$(find prof-decode-v19dev-s7 -name '*kernel_trace.csv' | head -1)
steps=$(python3 -c "import json; d=json.load(open('prof-decode-v19dev-s7/answer.json'))['timings']; print(d['predicted_n'] - d['draft_n_accepted'])")
python3 torchtrace_anatomy.py rocprof $f $steps
echo STEP7_DONE
