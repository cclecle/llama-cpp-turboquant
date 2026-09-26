#!/bin/bash
# step11.sh : v19 optimisation, MoE expert reuse in verify batches.
#   - GGML_CUDA_MOE_REUSE (default on): mul_mat_vec_q_moe_reuse, one block per (token, slot) route; the first route
#     of an expert computes all its routes and the others exit, so each expert row is read once per batch. Step 10
#     counted 49.9 routes but 33.7 distinct experts per verify call (32%; 35% of the cold ones, read over PCIe from
#     uncached host memory). Same lane layout and reduction per route: bit-identical results expected.
# A = reuse off, B = default. Production stays stopped.
#   1) test-backend-ops MUL_MAT_ID and MUL_MAT_VEC_FUSION, test-moe-tiered, 2) perplexity at ub 8 / 8k A vs B
#   (bit-identical expected), 3) same-config throughput A B B A, 4) kernel trace of B: the MoE kernels and the anatomy
#   S11_SPEED_ONLY=1 skips 1) and 2). This build also has the meta backend's allocation dependencies (no toggle): the
#   trace shows whether the fused MoE weighted sum now runs in decode (moe_weighted_reduction_f32 per verify pass).
cd /mnt/gguf/r9v/bench
B=/opt/llamacpp/llama-cpp-mine-v19dev/build3/bin
M=/mnt/gguf/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
HOT=$(readlink -f hot-r9v-uva16-uniform.txt)
OT_TIER='per_layer_token_embd=CPU,blk\.[0-9]+\.ffn_(gate|up|down)_exps\.weight=ROCm0_TIERED'
ENV_A="GGML_CUDA_MOE_REUSE=0"
ENV_B="X=0"
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3

if [ "${S11_SPEED_ONLY:-0}" != 1 ]; then
echo "### test-backend-ops"
for op in MUL_MAT_ID MUL_MAT_VEC_FUSION; do
  echo "$op: $(HIP_VISIBLE_DEVICES=0 timeout 3000 $B/test-backend-ops -o $op -b ROCm0 2>&1 | grep -E 'tests passed|FAIL' | head -4 | tr '\n' ' ')"
done
echo "### test-moe-tiered"
for d in ROCm0 ROCm1; do
  (cd /tmp && HIP_VISIBLE_DEVICES=0,1 timeout 900 $B/test-moe-tiered $d > /mnt/gguf/r9v/bench/tmt-s11-$d.log 2>&1); echo "$d: $(tail -1 tmt-s11-$d.log) (fails: $(grep -c FAIL tmt-s11-$d.log))"
done

echo "### perplexity at ub 8, ctx 8192: A vs B"
for arm in A B; do
  X="GGML_CUDA_MOE_HOT_FILE=$HOT"; [ $arm = A ] && X="$X $ENV_A" || X="$X $ENV_B"
  r=$(env $X HIP_VISIBLE_DEVICES=0,1 timeout 3600 $B/llama-perplexity -m $M -f /root/ppl.txt -c 8192 -b 8 -ub 8 --chunks 1 \
      -sm tensor -ngl 999 -ot "$OT_TIER" --n-cpu-moe 0 -fa on -ctk q8_0 -ctv q8_0 -t 12 -fit off 2>&1 | tee ppl-s11-$arm.log |
      grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' | tr '\n' ' ')
  echo "$arm: $r"
done
fi

echo "### throughput, warm, A B B A"
n=0
for arm in A B B A; do
  n=$((n + 1)); E=$ENV_B; [ $arm = A ] && E=$ENV_A
  FN_ENV="$E" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s11-$arm$n bash flashnext.sh v19dev 2>&1 | grep -E '^\{|load:'
done
python3 step_ms.py v19dev-s11- results.jsonl

echo "### kernel trace: B"
DP_ARGS=fn-xl-tiered-r9vlike.args DP_PROMPT=prompt-32k.txt DP_TOKENS=512 DP_TAG=-s11 bash decode-profile.sh v19dev 2>&1 | grep -E 'timings|trace:'
f=$(find prof-decode-v19dev-s11 -name '*kernel_trace.csv' | head -1)
steps=$(python3 -c "import json; d=json.load(open('prof-decode-v19dev-s11/answer.json'))['timings']; print(d['predicted_n'] - d['draft_n_accepted'])")
python3 torchtrace_anatomy.py rocprof $f $steps | head -14
python3 - "$f" <<'PY'
import csv, sys, collections
n = collections.Counter(); d = collections.Counter()
for r in csv.DictReader(open(sys.argv[1])):
    k = r['Kernel_Name']
    if 'moe' in k:
        k = k[:80]; n[k] += 1; d[k] += int(r['End_Timestamp']) - int(r['Start_Timestamp'])
for k, v in n.most_common(8):
    print(f'{v:6d} {d[k] / 1e3 / v:7.1f} us  {k}')
PY
echo STEP11_DONE
