#!/bin/bash
# step16.sh : v19, the HIP graph costs of the decode loop (step 15's HIP trace: 214 hipGraphLaunch per step at ~24 us
# of host time each, 6.4 hipGraphInstantiate and 3.8 hipGraphExecDestroy per step, 355 direct kernel launches).
#   1) GGML_CUDA_GRAPH_DEBUG=1 (diagnostic only): which graph nodes change after warmup and force a re-capture
#   2) DEBUG_CLR_GRAPH_PACKET_CAPTURE (the HIP runtime's pre-built graph packets), 0 and 1 against the default.
#      A runtime knob changes no math: every text must match step 15's D (8a0ec2a9ccd2816c6250db15e16181eb).
# Production stays stopped. Throughput B P0 P1 P1 P0 B.
cd /mnt/gguf/r9v/bench
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3

echo "### graph re-captures after warmup (GGML_CUDA_GRAPH_DEBUG=1)"
FN_ENV="GGML_CUDA_GRAPH_DEBUG=1" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s16-G bash flashnext.sh v19dev 2>&1 | grep -E 'load:'
grep -c 'cuda graph .* changed' v19dev-s16-G.log
grep 'cuda graph .* changed' v19dev-s16-G.log | sed -E 's/cuda graph 0x[0-9a-f]+ /cuda graph /; s/node [0-9]+ /node /' | sort | uniq -c | sort -rn | head -25
echo "--- per graph key"
grep -oE 'cuda graph 0x[0-9a-f]+ \([0-9]+ nodes\)' v19dev-s16-G.log | sort | uniq -c | sort -rn | head -10

echo "### throughput, warm, B P0 P1 P1 P0 B"
n=0
for arm in B P0 P1 P1 P0 B; do
  n=$((n + 1))
  case $arm in
    B)  E="X=0" ;;
    P0) E="DEBUG_CLR_GRAPH_PACKET_CAPTURE=0" ;;
    P1) E="DEBUG_CLR_GRAPH_PACKET_CAPTURE=1" ;;
  esac
  FN_ENV="$E" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s16-$arm-$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:'
done
python3 step_ms.py v19dev-s16-B results.jsonl | grep -v '^arm'
python3 step_ms.py v19dev-s16-P results.jsonl
echo "### generated text per load (md5; step 15 D: 8a0ec2a9ccd2816c6250db15e16181eb)"
md5sum v19dev-s16-*.answer.txt
echo STEP16_DONE
