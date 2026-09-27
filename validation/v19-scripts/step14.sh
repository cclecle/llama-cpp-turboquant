#!/bin/bash
# step14.sh : v19, a deterministic one-kernel top-k for the QSA indexer.
#   - GGML_CUDA_TOP_K_ROW (default on): top_k_row, one block per row up to 64k columns: the row in registers, four radix
#     passes in shared memory, the list in ascending column order (ties: the lowest columns). The multi-block path
#     took 11 launches per call and wrote through atomic counters, so the list order and the tied tail changed from
#     run to run: every load of step 12/13 generated a different greedy text (divergence after ~15 tokens).
# A = the multi-block path, B = default. Production stays stopped.
#   1) test-backend-ops TOP_K, 2) throughput A B B A (the two B texts should now be identical), 3) trace of B with
#   the host's HIP calls: kernel anatomy and what the host does while the GPU idles (hostgap_anatomy.py)
cd /mnt/gguf/r9v/bench
B=/opt/llamacpp/llama-cpp-mine-v19dev/build3/bin
ENV_A="GGML_CUDA_TOP_K_ROW=0"
ENV_B="X=0"
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3

echo "### test-backend-ops"
echo "TOP_K: $(HIP_VISIBLE_DEVICES=0 timeout 1800 $B/test-backend-ops -o TOP_K -b ROCm0 2>&1 | grep -E 'tests passed|FAIL' | head -6 | tr '\n' ' ')"

echo "### throughput, warm, A B B A"
n=0
for arm in A B B A; do
  n=$((n + 1)); E=$ENV_B; [ $arm = A ] && E=$ENV_A
  FN_ENV="$E" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s14-$arm$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:'
done
python3 step_ms.py v19dev-s14- results.jsonl
echo "### generated text per load (md5)"
md5sum v19dev-s14-*.answer.txt

echo "### kernel + HIP trace: B"
DP_HIP=1 DP_ARGS=fn-xl-tiered-r9vlike.args DP_PROMPT=prompt-32k.txt DP_TOKENS=512 DP_TAG=-s14 bash decode-profile.sh v19dev 2>&1 | grep -E 'timings|trace:'
d=prof-decode-v19dev-s14
f=$(find $d -name '*kernel_trace.csv' | head -1)
steps=$(python3 -c "import json; d=json.load(open('$d/answer.json'))['timings']; print(d['predicted_n'] - d['draft_n_accepted'])")
python3 torchtrace_anatomy.py rocprof $f $steps | head -8
python3 mmv_breakdown.py $f $steps "" top_k | head -6
python3 hostgap_anatomy.py $d $steps
echo STEP14_DONE
