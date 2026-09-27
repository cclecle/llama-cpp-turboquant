#!/bin/bash
# step17.sh : v19, diagnostics of the host side, one run each (no A/B): LLAMA_HOST_TIMING=1 (per decode call, the host
# time of each phase and between calls, verify batches and single tokens apart) and GGML_CUDA_GRAPH_DEBUG=1 (every
# graph that has to be re-captured after warmup, with the first node that changed). Production stays stopped.
cd /mnt/gguf/r9v/bench
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3

echo "### host timing + graph re-captures"
FN_ENV="LLAMA_HOST_TIMING=1 GGML_CUDA_GRAPH_DEBUG=1" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s17-T bash flashnext.sh v19dev 2>&1 | grep -E 'load:'
python3 step_ms.py v19dev-s17-T results.jsonl | grep -v '^arm'
md5sum v19dev-s17-T.answer.txt
L=v19dev-s17-T.log
echo "--- host timing (the last reports: steady decode)"
grep 'host timing' $L | tail -6 | cut -c1-260
echo "--- graph re-captures: $(grep -c 'cuda graph .* changed' $L) lines"
grep 'cuda graph .* changed' $L | sed -E 's/.*cuda graph 0x[0-9a-f]+ \(([0-9]+) nodes\): node [0-9]+ /\1 nodes: /; s/-[0-9]+'"'"' changed/'"'"' changed/' | sort | uniq -c | sort -rn | head -20
echo "--- re-captures per second of the run (bursts vs steady)"
grep 'cuda graph .* changed' $L | awk '{print $1}' | cut -d. -f1-2 | uniq -c | head -40
echo STEP17_DONE
