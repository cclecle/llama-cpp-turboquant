#!/bin/bash
# s7-loadab.sh : rebuild tmp-v20, then the tiered Qwen3.5-122B load time with and without the read-ahead of the tiered
# set_tensor (GGML_CUDA_TIERED_PREFETCH=0), interleaved on/off/on/off, load only, page cache and loadavg logged.
cd /mnt/gguf/r9v/bench
(cd /opt/llamacpp/tmp-v20 && bash validation/v17-scripts/build-scratch.sh /opt/llamacpp/tmp-v20 2>&1 | head -3)
for arm in on off on off; do
  E="X=0"; [ $arm = off ] && E="GGML_CUDA_TIERED_PREFETCH=0"
  echo "### $arm loadavg $(cut -d' ' -f1-3 /proc/loadavg) $(grep -E '^Cached|MemAvailable' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"
  FN_LOAD_ONLY=1 FN_ENV="$E" FN_ARGS=q35-tiered-44g.args FN_TAG=-s7-$arm bash flashnext.sh v20dev 2>&1 | grep -E 'load:|died'
done
echo S7_DONE
