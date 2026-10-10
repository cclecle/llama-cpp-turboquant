#!/bin/bash
# s4-q35-cache.sh : Qwen3.5-122B on tiered experts, hot set vs expert cache at about the same VRAM, untraced, through
# flashnext.sh (the 32k prompt, up to 2,048 tokens, greedy, results.jsonl / step_ms.py). Arms, interleaved twice:
#   h44     44 GiB hot, no cache
#   h40c16  40 GiB hot + 16 cache slots per layer and device
#   h44c8   44 GiB hot + 8 cache slots
# /proc/loadavg is logged before every run (the box runs a bursty side job). Production stays stopped.
cd /mnt/gguf/r9v/bench
T=/opt/llamacpp/tmp-v20
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
for G in 40 44; do
  hot=hot-q35-${G}g-uniform.txt
  [ -f $hot ] || /root/venv/bin/python3 $T/scripts/fleet/moe_hotset.py --model /mnt/gguf/Qwen3.5-122B-A10B-UD-Q5_K_M-00001-of-00003.gguf \
    --counts q35-imatrix.gguf --budget-gib $G --policy uniform --missing nearest -o $hot | tail -1
  [ -f q35-tiered-${G}g.args ] || { echo "missing q35-tiered-${G}g.args (s2-q35-tiered.sh makes it)"; \
    sed "s#hot-q35-44g-uniform.txt#$hot#" q35-tiered-44g.args > q35-tiered-${G}g.args; }
done
n=0
for arm in h44 h40c16 h44c8 h44c8 h40c16 h44; do
  n=$((n + 1)); E="X=0"; G=44
  case $arm in h40c16) G=40; E="GGML_CUDA_MOE_CACHE_SLOTS=16";; h44c8) E="GGML_CUDA_MOE_CACHE_SLOTS=8";; esac
  echo "### $arm-$n loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
  FN_ENV="$E" FN_ARGS=q35-tiered-${G}g.args FN_TAG=-s4-$arm-$n bash flashnext.sh v20dev 2>&1 | grep -E 'load:|died|memory:'
done
python3 step_ms.py v20dev-s4- results.jsonl
md5sum v20dev-s4-*.answer.txt
echo S4_DONE
