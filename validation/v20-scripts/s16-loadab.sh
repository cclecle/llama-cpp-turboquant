#!/bin/bash
# s16-loadab.sh : load time without the whole-file mmap prefetch (default since c059def92) vs with it
# (LLAMA_MMAP_PREFETCH=1), interleaved, load only (flashnext.sh FN_LOAD_ONLY=1: seconds to /health, peak VRAM, host RAM).
# The rig's page cache cannot be dropped: arms alternate, and the log keeps loadavg / MemAvailable / Cached per load.
#   Qwen3.5-122B :XL tiered (prof-xl.args), Flash-Next :XL tiered (fn-xl-tiered-r9vlike.args, cached reloads),
#   Qwen3.6-35B-A3B (S16_SMALL args file: weights in VRAM, nothing tiered).
# Stops production; restarts it at the end (the user's go, 2026-10-10).
cd /mnt/gguf/r9v/bench
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 5
run() { # <tag> <args file> <ctx> <prefetch 0|1>
  E="X=0"; [ "$4" = 1 ] && E="LLAMA_MMAP_PREFETCH=1"
  echo "### $1 prefetch=$4 loadavg $(cut -d' ' -f1-3 /proc/loadavg) $(grep -E '^Cached|MemAvailable' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"
  FN_LOAD_ONLY=1 FN_CTX=$3 FN_ENV="$E" FN_ARGS=$2 FN_TAG=-s16-$1-p$4 bash flashnext.sh v20dev 2>&1 | grep -E 'load:|died|memory:'
}
for r in 1 2; do
  run q35 prof-xl.args 262144 0
  run q35 prof-xl.args 262144 1
done
for p in 0 1 0 1; do
  run fn fn-xl-tiered-r9vlike.args 65536 $p
done
if [ -n "${S16_SMALL:-}" ]; then
  for p in 0 1 0 1; do
    run small $S16_SMALL 65536 $p
  done
fi
systemctl start llamacpp-0 llamacpp-1 llamacpp-both; sleep 20
for port in 20000 20001 20002; do echo "production $port: $(curl -s -m 5 http://127.0.0.1:$port/health)"; done
echo S16_DONE
