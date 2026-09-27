#!/bin/bash
# step18.sh : v19, graph re-capture bursts. Step 17: ~200 re-captures (every subgraph on both GPUs, ~160 us each on HIP)
# every ~256 generated tokens, when n_kv grows by its 256-cell pad and the graph is rebuilt and allocated again.
#   LLAMA_KV_N_PAD=2048: n_kv grows in steps of 2048 cells (the cells past the used ones are masked)
# A = pad 256 (as before), P = pad 2048. Then one diagnostic run of P: host timing per decode phase + re-captures.
# Production stays stopped.
cd /mnt/gguf/r9v/bench
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3

echo "### throughput, warm, A P P A"
n=0
for arm in A P P A; do
  n=$((n + 1)); E="X=0"; [ $arm = P ] && E="LLAMA_KV_N_PAD=2048"
  FN_ENV="$E" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s18-$arm$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:'
done
python3 step_ms.py v19dev-s18- results.jsonl
md5sum v19dev-s18-*.answer.txt

echo "### P: host timing + graph re-captures"
FN_ENV="LLAMA_KV_N_PAD=2048 LLAMA_HOST_TIMING=1 GGML_CUDA_GRAPH_DEBUG=1" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s18-T \
  bash flashnext.sh v19dev 2>&1 | grep -E 'load:'
python3 step_ms.py v19dev-s18-T results.jsonl | grep -v '^arm'
L=v19dev-s18-T.log
echo "--- host timing (steady decode)"
grep 'host timing' $L | tail -6 | sed -E 's/^.*host timing/host timing/' | cut -c1-260
echo "--- graph re-captures: $(grep -c 'cuda graph .* changed' $L) lines"
grep 'cuda graph .* changed' $L | sed -E 's/.*cuda graph 0x[0-9a-f]+ \(([0-9]+) nodes\): node [0-9]+ /\1 nodes: /; s/-[0-9]+'"'"' changed/'"'"' changed/' | sort | uniq -c | sort -rn | head -12
echo STEP18_DONE
