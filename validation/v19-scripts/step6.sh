#!/bin/bash
# step6.sh : v19 optimisation item 1, sparse QSA attention for decode and verify batches: flash attention walks the
# indexer's top-k list of each query (ggml_flash_attn_ext_add_kv_idx) instead of a full-length mask over every cell.
# LLAMA_QSA_SPARSE=0 restores the dense masked form on the same build. Production stays stopped.
#   1) perplexity at ub 8 (the sparse path) and 8k context, so the indexer really selects (2,051 of up to 8,192
#      cells; at 2k it keeps every visible cell): dense vs sparse, within noise expected (another summation order)
#   2) same-config throughput (fn-xl-tiered-r9vlike.args: MTP 4, MTP drafts only), warm, A B B A over loads
#   3) kernel trace of the sparse arm (decode-profile.sh) and its per-step anatomy
cd /mnt/gguf/r9v/bench
B=/opt/llamacpp/llama-cpp-mine-v19dev/build3/bin
M=/mnt/gguf/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
HOT=$(readlink -f hot-r9v-uva16-uniform.txt)
OT_TIER='per_layer_token_embd=CPU,blk\.[0-9]+\.ffn_(gate|up|down)_exps\.weight=ROCm0_TIERED'
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3

echo "### perplexity at ub 8, ctx 8192: dense vs sparse"
for arm in dense sparse; do
  X="GGML_CUDA_MOE_HOT_FILE=$HOT"; [ $arm = dense ] && X="$X LLAMA_QSA_SPARSE=0"
  r=$(env $X HIP_VISIBLE_DEVICES=0,1 timeout 3600 $B/llama-perplexity -m $M -f /root/ppl.txt -c 8192 -b 8 -ub 8 --chunks 1 \
      -sm tensor -ngl 999 -ot "$OT_TIER" --n-cpu-moe 0 -fa on -ctk q8_0 -ctv q8_0 -t 12 -fit off 2>&1 | tee ppl-qsa-$arm.log |
      grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' | tr '\n' ' ')
  echo "$arm: $r"
done

echo "### throughput, warm, A B B A: dense (A) vs sparse (B)"
n=0
for arm in A B B A; do
  n=$((n + 1)); E=X=0; [ $arm = A ] && E=LLAMA_QSA_SPARSE=0
  FN_ENV=$E FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-qsa-$arm$n bash flashnext.sh v19dev 2>&1 | grep -E '^\{|load:'
done
head -c 300 v19dev-qsa-B2.answer.txt; echo

echo "### kernel trace: sparse"
DP_ARGS=fn-xl-tiered-r9vlike.args DP_PROMPT=prompt-32k.txt DP_TOKENS=512 DP_TAG=-qsa bash decode-profile.sh v19dev 2>&1 | grep -E 'timings|trace:'
f=$(find prof-decode-v19dev-qsa -name '*kernel_trace.csv' | head -1)
steps=$(python3 -c "import json; d=json.load(open('prof-decode-v19dev-qsa/answer.json'))['timings']; print(d['predicted_n'] - d['draft_n_accepted'])")
python3 torchtrace_anatomy.py rocprof $f $steps
echo STEP6_DONE
